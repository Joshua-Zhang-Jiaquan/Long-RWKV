"""Development-only exact endpoints and actual request costs for practical comparators."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from lrwkv_evidence.long_context_eval.exact import audit,partitions
from lrwkv_evidence.long_context_eval.numerical import run as numerical_screen
from lrwkv_evidence.long_context_eval.runtime import metadata as runtime_metadata
from .exact import causal_audit
from lrwkv_evidence.long_context_baselines import models as M
from lrwkv_evidence.long_context_mvp import tasks as T
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--kind',choices=('attention','causal_rwkv'),required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cost-only',action='store_true');args=ap.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8 or os.environ.get('TRITON_F32_DEFAULT')!='ieee':raise ValueError('requires eight IEEE inference ranks')
    torch.set_num_threads(4);torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);dist.init_process_group('nccl')
    args.out.mkdir(parents=True,exist_ok=True);dest=args.out/f'rank{rank}.jsonl'
    if dest.exists():raise ValueError('refuse reused audit output')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if payload['step'] not in ((20,600) if args.cost_only else (600,)) or payload['contract']['recipe']['terminal_step']!=600:raise ValueError('wrong cost or development checkpoint')
    checkpoint_step=payload['step']
    root=Path(__file__).resolve().parents[2]
    if payload['contract']['recipe']['comparator']!=args.kind:raise ValueError('wrong comparator checkpoint')
    if W.canonical_sha(payload['contract'])!=payload['contract_sha256']:raise ValueError('contract checksum mismatch')
    for rel,digest in payload['contract']['source_hashes'].items():
        if W.file_sha(root/rel)!=digest:raise ValueError('training source mismatch: '+rel)
    if W.file_sha(args.base/'model.safetensors')!=payload['contract']['base_sha256']:raise ValueError('base checksum mismatch')
    tok,model=M.build(args.kind,args.base);model.load_state_dict(payload['model'],strict=True)
    identity=dict(kind=args.kind,parameters=sum(p.numel() for p in model.parameters()))
    contract_sha=payload['contract_sha256'];del payload
    model=model.to(device).eval();packed=model
    def predict(ex,serial,visible):
        canvas=T.inference_canvas(ex,serial,tok,visible,8-len(visible))
        b=C.pack([canvas],device)
        return packed(b)[0].double().log_softmax(-1)
    def causal_path(serial,y):
        ids=torch.tensor([serial['prefix_ids']],dtype=torch.long,device=device)
        logits,state=model.next_cached(ids);values=[logits[0].double().log_softmax(-1)]
        for i in range(7):
            token=torch.tensor([[tok.binary_ids[(y>>i)&1]]],dtype=torch.long,device=device)
            logits,state=model.next_cached(token,state);values.append(logits[0].double().log_softmax(-1))
        return torch.stack(values)
    # Same non-test input on every process, before endpoint collection.
    check=T.problem('dev',20270923,999,'dependent');serial=T.serialize(check,tok)
    repeats=torch.stack([predict(check,serial,{}) if args.kind=='attention' else causal_path(serial,check['support'][0]) for _ in range(3)])
    difference=float((repeats-repeats[0]).abs().max());gathered=[torch.empty_like(repeats[0]) for _ in range(8)]
    dist.all_gather(gathered,repeats[0]);cross=max(float((x-gathered[0]).abs().max()) for x in gathered)
    if difference>1e-5 or cross>1e-4:raise ValueError('numerical qualification failed')
    provenance=dict(kind='provenance',comparator=args.kind,rank=rank,checkpoint_sha256=W.file_sha(args.checkpoint),contract_sha256=contract_sha,
                    checkpoint_step=checkpoint_step,cost_only=args.cost_only,timing_repeats=10 if args.cost_only else 3,
                    model_identity=identity,runtime=runtime_metadata(torch,device),task_version=T.VERSION,split='dev',data_seed=20270923,
                    precision='FP32/IEEE; torch TF32 disabled',within_repeat_max_logprob_diff=difference,cross_rank_max_logprob_diff=cross,
                    scope='Development gate and cost screen; no final confirmation or model comparison claim',
                    timing='Pretokenized batch1; attention uses uncached full canvases, causal RWKV uses one prefix prefill plus seven cached token steps; one warmup; measured repeat count in timing_repeats')
    def write(row):
        with dest.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
    write(provenance)
    family=T.FAMILIES[rank%2];index=rank//2
    ex=T.problem('dev',20270923,index,family)
    def condition(length,position):
        condition_started=time.perf_counter()
        serial=T.serialize(ex,tok,length,position)
        torch.cuda.synchronize();audit_started=time.perf_counter()
        exact=None if args.cost_only else (audit(ex,lambda visible:predict(ex,serial,visible).cpu().tolist()) if args.kind=='attention' else causal_audit(ex,lambda y:causal_path(serial,y).cpu().tolist()))
        torch.cuda.synchronize();audit_seconds=time.perf_counter()-audit_started
        def request(groups,seed):
            generator=torch.Generator(device=device).manual_seed(seed);visible={}
            if args.kind=='causal_rwkv':
                ids=torch.tensor([serial['prefix_ids']],dtype=torch.long,device=device)
                logits,state=model.next_cached(ids);answer=0
                for i in range(8):
                    bit=int(torch.multinomial(logits.double().softmax(-1),1,generator=generator).item())
                    answer|=bit<<i
                    if i<7:
                        token=torch.tensor([[tok.binary_ids[bit]]],dtype=torch.long,device=device)
                        logits,state=model.next_cached(token,state)
                return answer
            for group in groups:
                lp=predict(ex,serial,visible)
                draws=torch.multinomial(lp.exp(),1,generator=generator).squeeze(-1).tolist()
                visible.update({i:draws[i] for i in group})
            return sum(visible[i]<<i for i in range(8))
        costs=[]
        for method,groups in (partitions(ex) if args.kind=='attention' else {'cached_causal':[[i] for i in range(8)]}).items():
            request(groups,7100);torch.cuda.synchronize();times=[];peak=0;outputs=[]
            for rep in range(10 if args.cost_only else 3):
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                y=request(groups,7101+rep);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                peak=max(peak,torch.cuda.max_memory_allocated());outputs.append(y)
            costs.append(dict(method=method,calls=len(groups),seconds=times,peak_allocated_bytes=peak,sample_outputs=outputs,
                              scope='Timing draws are not an accuracy estimator'))
        row=dict(kind='cost_condition' if args.cost_only else 'condition',family=family,index=index,context_tokens=serial['context_tokens'],requested_context_tokens=length,
                 position=position,evidence_span=serial['evidence_span'],records=serial['records'],filler_tokens=serial['filler_tokens'],exact=exact,costs=costs,
                 audit_seconds=audit_seconds,condition_wall_seconds=time.perf_counter()-condition_started)
        write(row);return row
    short=condition(None,'middle')
    if args.cost_only:
        def cost_common_long(spec):
            common_serial=T.serialize(check,tok,spec['context_tokens'],spec['position'])
            if args.kind=='causal_rwkv':return causal_path(common_serial,check['support'][0])
            visible={} if spec['history']=='all_masked' else {i:check['bits'][i] for i in range(4)}
            return predict(check,common_serial,visible)
        write(dict(kind='numerical_qualification',checks=numerical_screen(cost_common_long,args.kind)))
        for length in (1024,4096,16384):
            for position in ('far','middle','near'):condition(length,position)
        write(dict(kind='complete',cost_only=True,scope='Request cost only; task competence and long-context accuracy not measured'))
        dist.barrier();dist.destroy_process_group();return
    info=next(r for r in short['exact']['results'] if r['method']==('information_set' if args.kind=='attention' else 'cached_causal'))
    local_pass=info['joint_kl_nats']<.1 and info['exact_valid_mass']>.9
    gate=torch.tensor(int(local_pass),device=device);dist.all_reduce(gate,op=dist.ReduceOp.MIN)
    write(dict(kind='gate',local_pass=local_pass,all_eight_conditions_pass=bool(gate),criterion='Each of four development conditions per family: primary policy KL<0.1nat and exact valid mass>0.9'))
    if bool(gate):
        def common_long(spec):
            common_serial=T.serialize(check,tok,spec['context_tokens'],spec['position'])
            if args.kind=='causal_rwkv':return causal_path(common_serial,check['support'][0])
            visible={} if spec['history']=='all_masked' else {i:check['bits'][i] for i in range(4)}
            return predict(check,common_serial,visible)
        write(dict(kind='numerical_qualification',checks=numerical_screen(common_long,args.kind)))
        for length in (1024,4096,16384):
            for position in ('far','middle','near'):condition(length,position)
    write(dict(kind='complete',long_context_gate_passed=bool(gate)))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
