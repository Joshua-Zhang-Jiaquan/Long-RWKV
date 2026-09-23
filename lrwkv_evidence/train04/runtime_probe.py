"""Repeated-canvas and precision diagnostic for the frozen negative pilot."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import traceback
import torch
from . import tasks,worker as W
from .evaluate import support


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--model-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);variant=('bf16','bf16_no_tf32','fp32','bf16_deterministic')[rank%4]
    torch.set_num_threads(4);torch.cuda.set_device(local)
    if variant!='bf16':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    if variant=='bf16_deterministic':torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True,exist_ok=True);output=args.out/f'rank{rank}.json'
    if output.exists():raise ValueError('output exists')
    record=dict(rank=rank,variant=variant,scope='Repeated forward diagnostic; no training or model selection',checkpoint_sha256=W.file_sha(args.checkpoint),torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name())
    started=time.perf_counter()
    try:
        tok,bits,mask=W.load_tokenizer(args.base);model,identity,config=W.build_model(args.model_root,args.base,bits,mask,0)
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert payload['step']==500 and payload['provenance']['seed']==43
        model.load_state_dict(payload['model'],strict=True);del payload
        record['state_before']=W.state_digest(model)
        model=model.to('cuda').eval();contexts=[]
        for index in range(4):
            ex=tasks.make_example('test',73191,index);prefix=tok.encode(ex['prompt']);endpoint=support(ex)[0]
            for visible in ({},{i:endpoint[i] for i in ex['information_set']}):
                target=[bits[visible[i]] if i in visible else mask for i in range(8)]
                ids=torch.tensor([prefix+target],device='cuda',dtype=torch.long)
                contexts.append((ids,(len(prefix),len(prefix)+8),8-len(visible)))
        active=[None,None];reference={};layer_differences=[]
        def hook(layer):
            def collect(module,args,output):
                key=(active[0],layer);hidden=output[0].detach()
                if key not in reference:reference[key]=hidden.clone()
                else:layer_differences.append(dict(context=active[0],repeat=active[1],layer=layer,max_abs=float((hidden-reference[key]).abs().max())))
            return collect
        handles=[m.register_forward_hook(hook(i)) for i,m in enumerate(model.backbone.layers)]
        outputs=[]
        with torch.inference_mode():
            for repeat in range(6):
                # Alternating traversal changes the preceding canvas without changing the tested input.
                order=list(range(len(contexts)))
                if repeat%2:order.reverse()
                for ci in order:
                    active[:]=[ci,repeat];ids,span,stage=contexts[ci]
                    amp=nullcontext() if variant=='fp32' else torch.autocast('cuda',dtype=torch.bfloat16)
                    with amp:logits=model(ids,span,stage)
                    outputs.append(dict(context=ci,repeat=repeat,input_sha256=W.canonical_sha(ids.cpu().tolist()),logits=logits.float().cpu().tolist()))
        for h in handles:h.remove()
        by={r['context']:r for r in outputs if r['repeat']==0}
        diffs=[max(abs(x-y) for a,b in zip(r['logits'],by[r['context']]['logits']) for x,y in zip(a,b)) for r in outputs]
        record.update(supported=True,maximum_repeated_logit_difference=max(diffs),outputs=outputs,layer_differences=layer_differences,identity=identity,state_after=W.state_digest(model))
        record['parameters_unchanged']=record['state_before']==record['state_after']
    except Exception:
        record.update(supported=False,error=traceback.format_exc())
    record['elapsed_seconds']=time.perf_counter()-started
    W.atomic_json(output,record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('outputs','layer_differences','error')},allow_nan=False),flush=True)
if __name__=='__main__':main()
