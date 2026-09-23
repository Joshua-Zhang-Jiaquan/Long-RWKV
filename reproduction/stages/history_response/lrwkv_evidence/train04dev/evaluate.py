"""Development-only endpoint accounting, with independent marginal error sums."""
import argparse
import json
import math
import os
from pathlib import Path
import torch
from lrwkv_evidence.train04 import tasks,worker as W
from lrwkv_evidence.train04.evaluate import fixed_groups,support
from . import core as C
from .history_response import history_response


def endpoint_metrics(example,method,predict,cache=None):
    cache={} if cache is None else cache
    points=support(example);n=example['n'];logs=[];estimation=0.
    for bits in points:
        visible={};logq=0.
        for group in fixed_groups(example,method):
            stage=math.ceil(8*(n-len(visible))/n);key=(stage,tuple(sorted(visible.items())))
            if key not in cache:cache[key]=predict(example,visible,stage)
            lp=cache[key];p=tasks.oracle_marginals(example['matrix_rows'],example['syndrome'],visible,n)
            for i in group:
                logq+=lp[i][bits[i]]
                for bit,prob in enumerate((1-p[i],p[i])):
                    if prob:estimation+=prob*(math.log(prob)-lp[i][bit])/len(points)
            visible.update({i:bits[i] for i in group})
        logs.append(logq)
    maximum=max(logs);logmass=maximum+math.log(sum(math.exp(x-maximum) for x in logs))
    kl=-math.log(len(points))-sum(logs)/len(points)
    assignment={i:k for k,g in enumerate(fixed_groups(example,method)) for i in g}
    dependence=0.
    if example['family']=='paired_parity':
        dependence=sum(len({assignment[i] for i in range(n) if row>>i&1})==1 for row in example['matrix_rows'])*math.log(2)
    residual=kl-dependence-estimation
    if abs(residual)>1e-8 or logmass>1e-8 or kl+logmass < -1e-8:raise ValueError('decomposition failed')
    return dict(method=method,joint_kl_nats=kl,exact_valid_mass=math.exp(logmass),dependence_cost_nats=dependence,
                estimation_error_nats=estimation,decomposition_residual=residual,conditional_valid_kl_nats=kl+logmass,
                endpoint_log_probabilities=logs,valid_endpoints=points)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--model-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(4);torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    tok,bits,mask=W.load_tokenizer(args.base);model,identity,_=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);contract=payload['contract']
    allowed_steps={2:(300,700,1500),3:(300,700,1500),4:(300,700,1500),5:(1900,2500)}
    if payload['step'] not in allowed_steps.get(contract['version'],()):raise ValueError('wrong development checkpoint')
    identity_contract={k:v for k,v in contract.items() if k!='sources'}
    identity_contract['source_hashes']=sorted(contract['sources'].values())
    if W.canonical_sha(identity_contract)!=payload['contract_sha256']:raise ValueError('contract hash mismatch')
    if contract['base_sha256']!=W.file_sha(args.base/'model.safetensors'):raise ValueError('base mismatch')
    # Compare original recorded sources by their relative role, not stage prefix.
    here=Path(__file__).parent
    for name,digest in contract['sources'].items():
        original=Path(name)
        if '/model_source/' in name:current=args.model_root/'longrwkv'/original.name
        elif '/train04dev/' in name:current=here/original.name
        elif '/train04paired/' in name:current=here.parent/'train04paired'/original.name
        elif '/train04binding/' in name:current=here.parent/'train04binding'/original.name
        elif '/train04mix/' in name:current=here.parent/'train04mix'/original.name
        else:current=here.parent/'train04'/original.name
        if W.file_sha(current)!=digest:raise ValueError(f'source mismatch:{current}')
    step=payload['step'];model.load_state_dict(payload['model'],strict=True);del payload
    model=model.to('cuda').eval();calls=0
    if contract['version'] in (4,5):
        from lrwkv_evidence.train04binding import core as B
        if contract.get('serialization')!='public output labels beside value slots v1':raise ValueError('unknown binding serialization')
        serial=C.SerialDenoiser(model)
    @torch.inference_mode()
    def predict(ex,visible,stage):
        nonlocal calls
        prefix=tok.encode(ex['prompt']);ids=prefix+[bits[visible[i]] if i in visible else mask for i in range(ex['n'])]
        if contract['version'] in (4,5):
            canvas=dict(ids=ids,prefix=len(prefix),stage=stage,masked=[i not in visible for i in range(ex['n'])],gold=[0]*ex['n'],oracle=[.5]*ex['n'])
            decorated=B.decorate(ex,canvas,tok)
            logits=serial(B.pack([decorated],'cuda'))[0]
        else:
            logits=model(torch.tensor([ids],device='cuda'),(len(prefix),len(ids)),stage)
        calls+=1
        return logits.double().log_softmax(-1).cpu().tolist()
    args.out.mkdir(parents=True,exist_ok=True);dest=args.out/f'rank{rank}.jsonl'
    if os.environ.get('DEV_NUMERICAL_QUALIFICATION')=='1':
        from .numerical_qualification import run
        run(model,tok,predict,contract,args.checkpoint,args.out,rank,world)
        return
    if dest.exists():raise ValueError('output exists')
    with dest.open('w') as f:
        def emit(row):f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        emit(dict(kind='provenance',triton_f32_default=os.environ.get('TRITON_F32_DEFAULT','backend_default'),torch_tf32=False,checkpoint_sha256=W.file_sha(args.checkpoint),step=step,objective=contract['objective'],rank=rank,world=world,
                  history_coupling=contract.get('history_coupling'),serialization=contract.get('serialization','contiguous target suffix'),mask_law=contract.get('mask_law','standard corruption'),split='dev',seed=51917,conditions_per_dimension=16,dimensions=[2,4,8],scope='Development diagnostics, no confirmatory inference'))
        for n in (2,4,8):
            for index in range(rank,16,world):
                ex=C.example('dev',51917,index,n);cache={}
                for method in ('one','information_set','random_halves'):
                    emit(dict(kind='endpoint',n=n,index=index,family=ex['family'],**endpoint_metrics(ex,method,predict,cache)))
                emit(dict(kind='history_response',n=n,index=index,family=ex['family'],**history_response(ex,cache)))
        emit(dict(kind='complete',actual_audit_calls=calls))
if __name__=='__main__':main()
