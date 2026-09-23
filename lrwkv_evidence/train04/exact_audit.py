"""Post-primary exact endpoint audit of the two fixed two-call policies.

No training, checkpoint choice or sampler change. Enumerating all16valid endpoints
removes Monte Carlo error in validity on the existing32held-out conditions.
Memoization is only an audit optimization, never a generation speed claim.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
from . import tasks,worker as W
from .evaluate import Predictor,CONTRACT,fixed_groups,support


def endpoint_metrics(example,method,predict,cache):
    endpoints=support(example);logs=[]
    for bits in endpoints:
        visible={};logq=0.
        for group in fixed_groups(example,method):
            stage=math.ceil(8*(8-len(visible))/8);key=(stage,tuple(sorted(visible.items())))
            if key not in cache:cache[key]=predict.log_probs(example,visible,stage)
            logp=cache[key]
            for i in group:logq+=logp[i][bits[i]];visible[i]=bits[i]
        logs.append(logq)
    maximum=max(logs);logmass=maximum+math.log(sum(math.exp(v-maximum) for v in logs))
    kl=-math.log(16)-sum(logs)/16
    dependence=0.
    if example['family']=='paired_parity':
        assignment={i:k for k,g in enumerate(fixed_groups(example,method)) for i in g}
        dependence=sum(len({assignment[i] for i in range(8) if row>>i&1})==1 for row in example['matrix_rows'])*math.log(2)
    if logmass>1e-9 or kl+logmass < -1e-9 or kl<dependence-1e-9:raise ValueError('invalid endpoint decomposition')
    return dict(method=method,exact_valid_mass=math.exp(logmass),joint_kl_nats=kl,
                invalid_mass_log_loss=-logmass,conditional_valid_kl_nats=kl+logmass,
                dependence_cost_nats=dependence,estimation_error_nats=kl-dependence,
                valid_endpoints=endpoints,endpoint_log_probabilities=logs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--model-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8:raise ValueError('eight workers required')
    seed=(17,29,43)[rank%3];ranks=[r for r in range(world) if (17,29,43)[r%3]==seed];shard=ranks.index(rank)
    torch.set_num_threads(4);torch.cuda.set_device(local)
    manifest=json.loads(args.manifest.read_text());item=next(r for r in manifest['runs'] if r['seed']==seed)
    checkpoint=Path(item['checkpoint']);primary=json.loads(Path(item['primary_rank0']).read_text().splitlines()[0])
    if W.file_sha(checkpoint)!=item['checkpoint_sha256'] or primary['checkpoint_sha256']!=item['checkpoint_sha256']:raise ValueError('checkpoint identity differs')
    if not all(c['bit_exact'] for c in primary['native_fla_mixer_parity']):raise ValueError('primary native parity not qualified')
    tokenizer,binary_ids,mask_id=W.load_tokenizer(args.base)
    model,identity,config=W.build_model(args.model_root,args.base,binary_ids,mask_id,0)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    provenance=payload['provenance'];contract=provenance['contract']
    if payload['step']!=500 or provenance['seed']!=seed or provenance['mode']!='train':raise ValueError('wrong terminal checkpoint')
    sources=sorted(W.file_sha(p) for p in [Path(W.__file__),Path(tasks.__file__),*sorted((args.model_root/'longrwkv').glob('*.py'))])
    if sources!=contract['source_content_sha256'] or W.canonical_sha(contract)!=provenance['contract_sha256']:raise ValueError('training contract/source changed')
    if contract['base_sha256']!=W.file_sha(args.base/'model.safetensors'):raise ValueError('base changed')
    if W.file_sha(Path(__file__).with_name('evaluate.py'))!=primary['sources']['evaluate.py']:raise ValueError('predictor changed since primary test')
    model.load_state_dict(payload['model'],strict=True);del payload
    model=model.to('cuda').eval();predict=Predictor(model,tokenizer,binary_ids,mask_id,'cuda')
    args.out.mkdir(parents=True,exist_ok=True);out=args.out/f'rank{rank}.jsonl'
    if out.exists():raise ValueError('output exists')
    started=time.perf_counter()
    with out.open('w') as stream:
        def emit(r):stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
        emit(dict(kind='provenance',seed=seed,rank=rank,world=world,manifest_sha256=W.file_sha(args.manifest),checkpoint_sha256=item['checkpoint_sha256'],audit_source_sha256=W.file_sha(Path(__file__)),primary_contract=CONTRACT))
        for index in range(shard,32,len(ranks)):
            example=tasks.make_example('test',CONTRACT['data_seed'],index);cache={}
            for method in ('information_set','random_halves'):
                emit(dict(kind='exact',seed=seed,index=index,family=example['family'],instance_id=example['instance_id'],**endpoint_metrics(example,method,predict,cache)))
            print(json.dumps(dict(rank=rank,seed=seed,index=index,calls=predict.calls)),flush=True)
        emit(dict(kind='complete',seed=seed,rank=rank,actual_audit_nfe=predict.calls,elapsed_seconds=time.perf_counter()-started))
if __name__=='__main__':main()
