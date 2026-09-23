"""Development-only exact endpoints and uncached request costs for trained RWKV."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from .exact import audit,partitions
from .numerical import run as numerical_screen
from .runtime import metadata as runtime_metadata
from lrwkv_evidence.long_context_mvp import tasks as T
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--model-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--diagnostic-step200',action='store_true')
    ap.add_argument('--cost-only',action='store_true');args=ap.parse_args()
    if args.cost_only and args.diagnostic_step200:raise ValueError('separate cost and milestone studies')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8 or os.environ.get('TRITON_F32_DEFAULT')!='ieee':raise ValueError('requires eight IEEE inference ranks')
    torch.set_num_threads(4);torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);dist.init_process_group('nccl')
    args.out.mkdir(parents=True,exist_ok=True);dest=args.out/f'rank{rank}.jsonl'
    if dest.exists():raise ValueError('refuse reused audit output')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    expected_step=200 if args.diagnostic_step200 else 600
    if payload['step']!=expected_step or payload['contract']['recipe']['terminal_step']!=600:raise ValueError('wrong fixed development milestone')
    root=Path(__file__).resolve().parents[2]
    if W.canonical_sha(payload['contract'])!=payload['contract_sha256']:raise ValueError('contract checksum mismatch')
    for rel,digest in payload['contract']['source_hashes'].items():
        if W.file_sha(root/rel)!=digest:raise ValueError('training source mismatch: '+rel)
    if W.file_sha(args.base/'model.safetensors')!=payload['contract']['base_sha256']:raise ValueError('base checksum mismatch')
    tok,bits,mask=W.load_tokenizer(args.base);model,identity,_=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head);model.load_state_dict(payload['model'],strict=True)
    contract_sha=payload['contract_sha256'];initial_checkpoint_sha=payload['contract'].get('initial_checkpoint_sha256');del payload
    model=model.to(device).eval();packed=C.SerialDenoiser(model).eval()
    def predict(ex,serial,visible):
        canvas=T.inference_canvas(ex,serial,tok,visible,8-len(visible))
        b=C.pack([canvas],device)
        return packed(b)[0].double().log_softmax(-1)
    # Same non-test input on every process, before endpoint collection.
    check=T.problem('dev',20270923,999,'dependent');serial=T.serialize(check,tok)
    repeats=torch.stack([predict(check,serial,{}) for _ in range(3)])
    difference=float((repeats-repeats[0]).abs().max());gathered=[torch.empty_like(repeats[0]) for _ in range(8)]
    dist.all_gather(gathered,repeats[0]);cross=max(float((x-gathered[0]).abs().max()) for x in gathered)
    if difference>1e-5 or cross>1e-4:raise ValueError('numerical qualification failed')
    provenance=dict(kind='provenance',rank=rank,checkpoint_sha256=W.file_sha(args.checkpoint),contract_sha256=contract_sha,
                    checkpoint_step=expected_step,diagnostic_step200=args.diagnostic_step200,
                    initial_checkpoint_sha256=initial_checkpoint_sha,
                    cost_only=args.cost_only,timing_repeats=10 if args.cost_only else 3,
                    model_identity=identity,runtime=runtime_metadata(torch,device),task_version=T.VERSION,split='dev',data_seed=20270923,
                    precision='FP32/IEEE; torch TF32 disabled',within_repeat_max_logprob_diff=difference,cross_rank_max_logprob_diff=cross,
                    scope='Development gate and cost screen; no final confirmation or model comparison claim',
                    timing='Pretokenized batch1 uncached full-canvas requests; one warmup; measured repeat count in timing_repeats; no optimized AR speed claim')
    def write(row):
        with dest.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
    write(provenance)
    family=T.FAMILIES[rank%2];index=rank//2
    ex=T.problem('dev',20270923,index,family)
    def condition(length,position):
        condition_started=time.perf_counter()
        serial=T.serialize(ex,tok,length,position)
        torch.cuda.synchronize();audit_started=time.perf_counter()
        exact=None if args.cost_only else audit(ex,lambda visible:predict(ex,serial,visible).cpu().tolist())
        torch.cuda.synchronize();audit_seconds=time.perf_counter()-audit_started
        def request(groups,seed):
            generator=torch.Generator(device=device).manual_seed(seed);visible={}
            for group in groups:
                lp=predict(ex,serial,visible)
                draws=torch.multinomial(lp.exp(),1,generator=generator).squeeze(-1).tolist()
                visible.update({i:draws[i] for i in group})
            return sum(visible[i]<<i for i in range(8))
        costs=[]
        for method,groups in partitions(ex).items():
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
            visible={} if spec['history']=='all_masked' else {i:check['bits'][i] for i in range(4)}
            return predict(check,common_serial,visible)
        write(dict(kind='numerical_qualification',checks=numerical_screen(cost_common_long)))
        for length in (1024,4096,16384):
            for position in ('far','middle','near'):condition(length,position)
        write(dict(kind='complete',cost_only=True,scope='Request cost only; task competence and long-context accuracy not measured'))
        dist.barrier();dist.destroy_process_group();return
    info=next(r for r in short['exact']['results'] if r['method']=='information_set')
    local_pass=info['joint_kl_nats']<.1 and info['exact_valid_mass']>.9
    gate=torch.tensor(int(local_pass),device=device);dist.all_reduce(gate,op=dist.ReduceOp.MIN)
    write(dict(kind='gate',local_pass=local_pass,all_eight_conditions_pass=bool(gate),criterion='Each of four development conditions per family: information-set KL<0.1nat and exact valid mass>0.9'))
    if bool(gate) and not args.diagnostic_step200:
        def common_long(spec):
            common_serial=T.serialize(check,tok,spec['context_tokens'],spec['position'])
            visible={} if spec['history']=='all_masked' else {i:check['bits'][i] for i in range(4)}
            return predict(check,common_serial,visible)
        write(dict(kind='numerical_qualification',checks=numerical_screen(common_long)))
        for length in (1024,4096,16384):
            for position in ('far','middle','near'):condition(length,position)
    write(dict(kind='complete',long_context_gate_passed=bool(gate)))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
