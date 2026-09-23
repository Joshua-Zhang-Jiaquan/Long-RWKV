"""Development-only, repeated full-model inference in eight worker processes.

This checks the public-label wrapper against direct backbone gathers. It is
not a comparison against the official RWKV implementation or a proof for
unmeasured canvases. No confirmation conditions are accessed.
"""
import json
import os


def run(model,tok,predict,contract,checkpoint,out,rank,world):
    import torch
    from lrwkv_evidence.train04 import worker as W
    from lrwkv_evidence.train04binding import core as B
    from . import core as C
    if contract['version'] not in (4,5):
        raise ValueError('qualification currently requires public-label serialization')
    if os.environ.get('TRITON_F32_DEFAULT')!='ieee':
        raise ValueError('qualification requires explicit IEEE arithmetic')
    rows=[]
    with torch.inference_mode():
        for n in (2,4,8):
            for index in (0,1):
                ex=C.example('dev',51917,index,n)
                free=ex['information_set']
                for stage in (8,4):
                    visible={} if stage==8 else {i:ex['bits'][i] for i in free}
                    prefix=tok.encode(ex['prompt'])
                    canvas=dict(ids=prefix+[tok.binary_ids[visible[i]] if i in visible else tok.mask_id for i in range(n)],
                                prefix=len(prefix),stage=stage,masked=[i not in visible for i in range(n)],gold=[0]*n,oracle=[.5]*n)
                    decorated=B.decorate(ex,canvas,tok)
                    batch=B.pack([decorated],'cuda')
                    # Bypass SerialDenoiser and PackedDenoiser: call backbone directly.
                    direct=model.backbone(**{k:batch[k] for k in ('input_ids','doc_starts','attention_mask','codes','block_t','gather_idx')},
                                          block_size=1,R=1,force_forward=False).reshape(n,2).double().log_softmax(-1)
                    repeats=[predict(ex,visible,stage) for _ in range(3)]
                    reference=torch.tensor(repeats[0],dtype=torch.float64,device='cuda')
                    wrapper=float((direct-reference).abs().max())
                    repeat=max(float((torch.tensor(v,dtype=torch.float64,device='cuda')-reference).abs().max()) for v in repeats[1:])
                    rows.append(dict(n=n,index=index,stage=stage,log_probabilities=repeats[0],
                                     wrapper_max_log_probability_difference=wrapper,repeat_max_log_probability_difference=repeat))
    result=dict(rank=rank,world=world,checkpoint_sha256=W.file_sha(checkpoint),triton_f32_default=os.environ['TRITON_F32_DEFAULT'],
                torch_tf32=torch.backends.cuda.matmul.allow_tf32,rows=rows,execution_complete=True,
                scope='12 fixed development canvases, 3 repeats, same canvases across eight GPU processes; not official-model parity')
    W.atomic_json(out/f'numerical_rank{rank}.json',result)
    print(json.dumps(dict(numerical_qualification_complete=True,rank=rank)),flush=True)


def collect(root):
    """Reject partial, inconsistent, nonfinite, or numerically unstable evidence."""
    import math
    reports=[json.loads((root/f'numerical_rank{rank}.json').read_text()) for rank in range(8)]
    expected={(n,i,s) for n in (2,4,8) for i in (0,1) for s in (4,8)}
    reference={}
    repeat=wrapper=cross=0.
    for rank,report in enumerate(reports):
        if report['rank']!=rank or report['world']!=8 or not report['execution_complete']:
            raise ValueError('incomplete or wrong-rank qualification')
        if report['checkpoint_sha256']!=reports[0]['checkpoint_sha256'] or report['triton_f32_default']!='ieee' or report['torch_tf32']:
            raise ValueError('qualification provenance mismatch')
        rows=report['rows'];keys={(r['n'],r['index'],r['stage']) for r in rows}
        if keys!=expected or len(rows)!=len(expected):raise ValueError('wrong canvas panel')
        for row in rows:
            key=(row['n'],row['index'],row['stage']);lp=row['log_probabilities']
            if len(lp)!=row['n'] or any(len(v)!=2 for v in lp):raise ValueError('wrong output shape')
            values=[x for pair in lp for x in pair]
            errors=[row['wrapper_max_log_probability_difference'],row['repeat_max_log_probability_difference']]
            if not all(math.isfinite(v) for v in values+errors) or min(errors)<0:raise ValueError('nonfinite or invalid numeric evidence')
            if any(abs(sum(math.exp(v) for v in pair)-1)>1e-10 for pair in lp):raise ValueError('unnormalized output')
            if rank==0:reference[key]=values
            cross=max(cross,max(abs(a-b) for a,b in zip(values,reference[key])))
            wrapper=max(wrapper,errors[0]);repeat=max(repeat,errors[1])
    # Fixed before observing this qualification run; log-probability tolerances.
    passed=wrapper<=1e-5 and repeat<=1e-5 and cross<=1e-4
    return dict(passed=passed,checkpoint_sha256=reports[0]['checkpoint_sha256'],canvases=12,processes=8,repeats=3,
                wrapper_max_log_probability_difference=wrapper,repeat_max_log_probability_difference=repeat,
                cross_process_max_log_probability_difference=cross,
                thresholds=dict(wrapper=1e-5,repeat=1e-5,cross_process=1e-4),scope=reports[0]['scope'])


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=collect(args.root);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('numerical qualification failed')
