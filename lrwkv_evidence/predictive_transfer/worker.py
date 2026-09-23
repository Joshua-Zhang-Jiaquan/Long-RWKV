"""Fresh public-base training with fixed endpoints and discarded qualification."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from . import training as R
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M
BASE_SHA='e162387e439dfa3387a0ca7da61638749d00c9862b8cc0192ae5d366c8c1a524'
QUAL_STEPS=(1,301,701,702,2501,2502,2503,2504)


@torch.inference_mode()
def qualify(packed,tok,device,seed):
    packed.eval()
    rows=R.records(701,0,tok,seed,True)[:2]
    b=R.batch(rows,device); normal=packed(b)
    singles=torch.cat([packed(R.batch([r],device)) for r in rows])
    poisoned=dict(b);poisoned['input_ids']=b['input_ids'].clone()
    poisoned['input_ids'][0,:int(b['doc_starts'][0,1])]=tok.binary_ids[1]
    changed=packed(poisoned)
    check=dict(serial_difference=float((normal-singles).abs().max()),
               other_document_difference=float((normal[1:]-changed[1:]).abs().max()),
               own_document_difference=float((normal[0]-changed[0]).abs().max()))
    if check['serial_difference']>1e-5 or check['other_document_difference']>1e-5 or check['own_document_difference']<1e-6:
        raise ValueError(str(check))
    packed.train();return check


def main():
    ap=argparse.ArgumentParser()
    for name in ('base','model-root','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--qualification',action='store_true')
    ap.add_argument('--qualification-out',type=Path)
    args=ap.parse_args()
    if args.seed not in ((202709230,) if args.qualification else R.SEEDS):raise ValueError('unregistered seed')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8 or os.environ.get('TRITON_F32_DEFAULT')!='ieee':raise ValueError('eight IEEE ranks required')
    torch.set_num_threads(4);torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl');device=torch.device('cuda',local)
    root=Path(__file__).resolve().parents[2]
    sources=json.loads((root/'transfer_sources.json').read_text())
    for rel,digest in sources.items():
        if W.file_sha(root/rel)!=digest:raise ValueError('source mismatch '+rel)
    source_sha=W.canonical_sha(sources)
    if W.file_sha(args.base/'model.safetensors')!=BASE_SHA:raise ValueError('wrong public base')
    if not args.qualification:
        q=json.loads((args.qualification_out/'completion.json').read_text())
        if not q.get('execution_complete') or not q.get('qualification') or q['source_sha256']!=source_sha:
            raise ValueError('requires matching source-only qualification')
        protocol=json.loads(Path(os.environ['TRANSFER_PROTOCOL']).read_text())
        if protocol['status']!='frozen' or protocol['training']!=R.RECIPE or protocol['training_source_sha256']!=source_sha:
            raise ValueError('unfrozen or changed protocol')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'train.jsonl').exists() or (args.out/'completion.json').exists():raise ValueError('reused output')
    tok,bits,mask=W.load_tokenizer(args.base)
    model,identity,_=W.build_model(args.model_root,args.base,bits,mask,args.seed)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head)
    model=model.to(device);packed=C.SerialDenoiser(model)
    check=qualify(packed,tok,device,args.seed)
    W.atomic_json(args.out/f'qualification_rank{rank}.json',check)
    ddp=torch.nn.parallel.DistributedDataParallel(packed,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    contract=dict(recipe=R.RECIPE,seed=args.seed,source_sha256=source_sha,base_sha256=BASE_SHA,
                  qualification=args.qualification,initialization_seed=args.seed)
    if not args.qualification:contract['protocol_sha256']=W.file_sha(Path(os.environ['TRANSFER_PROTOCOL']))
    if rank==0:W.atomic_json(args.out/'provenance.json',dict(contract=contract,identity=identity))
    dist.barrier();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    updates=QUAL_STEPS if args.qualification else range(1,R.RECIPE['steps']+1)
    for update,step in enumerate(updates,1):
        began=time.perf_counter()
        if step==2501:
            optimizer=torch.optim.AdamW(ddp.parameters(),lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
        rows=R.records(step,rank,tok,args.seed,args.qualification)
        optimizer.zero_grad(set_to_none=True);loss_sum=0.;entropy_sum=0.;tokens=0
        for j,row in enumerate(rows):
            b=R.batch([row],device)
            with ddp.no_sync() if j<len(rows)-1 else nullcontext():
                value=R.loss(ddp(b),b)/len(rows)
                if not bool(torch.isfinite(value)):raise ValueError('nonfinite loss')
                value.backward()
            loss_sum+=float(value.detach());entropy_sum+=float(M.oracle_entropy(b))/len(rows)
            tokens+=b['input_ids'].numel()
        norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.,error_if_nonfinite=True)
        if update==1:
            missing=[name for name,p in model.named_parameters() if p.grad is None or p.dtype!=torch.float32 or p.grad.dtype!=torch.float32]
            if missing:raise ValueError('missing/wrong gradient '+str(missing[:5]))
        lr=3e-5*min(step/50,1.) if step<=2500 else 1e-5*min((step-2500)/20,1.)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        values=torch.tensor([loss_sum,entropy_sum,tokens],dtype=torch.float64,device=device);dist.all_reduce(values)
        torch.cuda.synchronize()
        profile=torch.tensor([time.perf_counter()-began,torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],dtype=torch.float64,device=device)
        dist.all_reduce(profile,op=dist.ReduceOp.MAX)
        if rank==0:
            row=dict(update=update,step=step,loss=float(values[0]/8),oracle_entropy=float(values[1]/8),excess_ce=float((values[0]-values[1])/8),
                     global_input_tokens=int(values[2]),lr=lr,grad_norm_rank0=float(norm),elapsed_seconds=time.perf_counter()-started,
                     max_rank_step_seconds=float(profile[0]),max_rank_peak_allocated_bytes=int(profile[1]),max_rank_peak_reserved_bytes=int(profile[2]))
            with (args.out/'train.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
        if step in (700,1500,2500,3100) or update==len(updates):
            dist.barrier()
            if rank==0:
                tmp=args.out/'resume.pt.tmp'
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                                step=step,update=update,contract=contract),tmp)
                os.replace(tmp,args.out/'resume.pt')
            dist.barrier()
    if rank==0:
        W.atomic_json(args.out/'completion.json',dict(execution_complete=True,qualification=args.qualification,step=step,updates=update,
                      seed=args.seed,source_sha256=source_sha,checkpoint_sha256=W.file_sha(args.out/'resume.pt'),elapsed_seconds=time.perf_counter()-started))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
