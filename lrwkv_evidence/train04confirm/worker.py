"""Fresh-seed full-lineage replication; never reads confirmation conditions."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C,worker as D
from lrwkv_evidence.train04binding import worker as B
from . import contract as K,recipe as R


def main():
    parser=argparse.ArgumentParser()
    for name in ('manifest','base','model-root','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[17,53,71,89],required=True)
    parser.add_argument('--phase',choices=['parent','independent','complementary'],required=True)
    parser.add_argument('--parent',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--qualification',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    manifest,manifest_sha=K.validate(args.manifest,root,args.model_root,args.qualification)
    if args.qualification:
        if args.seed!=17 or args.phase!='parent' or args.parent or args.resume:raise ValueError('qualification is a discarded fresh 20-update run')
    elif args.seed not in R.SEEDS:raise ValueError('confirmation requires a fresh data seed')
    if args.phase=='parent' and args.parent:raise ValueError('parent phase starts from released weights')
    if args.phase!='parent' and not args.parent:raise ValueError('branch requires its seed-specific parent')
    if os.environ.get('TRITON_F32_DEFAULT') not in (None,'tf32'):raise ValueError('training precision differs from development')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('requires eight ranks')
    torch.set_num_threads(4);torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl');device=torch.device('cuda',local)
    if W.file_sha(args.base/'model.safetensors')!=manifest['base_sha256']:raise ValueError('base checkpoint mismatch')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'completion.json').exists():raise ValueError('completed output exists')
    if (args.out/'train.jsonl').exists():raise ValueError('use a new output directory when resuming')
    tok,bits,mask=W.load_tokenizer(args.base);model,identity,_=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head);model=model.to(device);packed=C.SerialDenoiser(model)
    checks=dict(contiguous=D.qualify(model,packed,tok,device),public_labels=B.qualify(model,packed,tok,device))
    ddp=torch.nn.parallel.DistributedDataParallel(packed,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    stop=20 if args.qualification else (1500 if args.phase=='parent' else 2500)
    parent_sha=None;start=0
    if args.parent:
        parent_sha=W.file_sha(args.parent)
        payload=torch.load(args.parent,map_location='cpu',weights_only=False);pc=payload['contract']
        if payload['step']!=1500 or pc.get('version')!=6 or pc.get('phase')!='parent' or pc.get('seed')!=args.seed or pc.get('manifest_sha256')!=manifest_sha or pc.get('qualification'):
            raise ValueError('wrong seed, recipe, or parent checkpoint')
        if W.canonical_sha(pc)!=payload['contract_sha256']:raise ValueError('parent contract corrupted')
        model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);start=1500;del payload
    contract=dict(version=6,phase=args.phase,seed=args.seed,initial_seed=0,manifest_sha256=manifest_sha,
                  qualification=args.qualification,stop_step=stop,parent_checkpoint_sha256=parent_sha,
                  base_sha256=manifest['base_sha256'],objective='rao_blackwell',recipe=K.RECIPE)
    digest=W.canonical_sha(contract)
    if args.resume:
        payload=torch.load(args.resume,map_location='cpu',weights_only=False)
        if payload['contract_sha256']!=digest or payload['contract']!=contract:raise ValueError('resume contract mismatch')
        model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);start=payload['step'];del payload
    if not 0<=start<stop or (args.phase!='parent' and start<1500):raise ValueError('invalid resume step')
    if rank==0:W.atomic_json(args.out/'provenance.json',dict(contract=contract,contract_sha256=digest,identity=identity,qualification_checks=checks,start_step=start,stop_step=stop))
    dist.barrier();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    mode='independent' if args.phase=='parent' else args.phase
    for step in range(start+1,stop+1):
        batch=R.batch(step,rank,tok,args.seed,mode,device)
        optimizer.zero_grad(set_to_none=True);value=R.loss(step,ddp(batch),batch)
        if not bool(torch.isfinite(value)):raise ValueError('nonfinite training loss')
        value.backward();norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.,error_if_nonfinite=True)
        if step in (start+1,301,701,1501):
            missing=[name for name,p in model.named_parameters() if p.grad is None or p.dtype!=torch.float32 or p.grad.dtype!=torch.float32]
            if missing:raise ValueError('missing or incorrect gradients: '+str(missing[:5]))
        rate=3e-5*min(step/50,1.)
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.step()
        values=torch.tensor([float(value),batch['input_ids'].numel()],device=device,dtype=torch.float64);dist.all_reduce(values)
        if rank==0:
            record=dict(step=step,phase=R.phase(step),seed=args.seed,mean_loss=float(values[0]/8),global_input_tokens=int(values[1]),lr=rate,grad_norm_rank0=float(norm),elapsed_seconds=time.perf_counter()-started)
            with (args.out/'train.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps(record),flush=True)
        if step in (300,700,1500,2500) or step==stop:
            dist.barrier()
            if rank==0:
                payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),step=step,contract=contract,contract_sha256=digest)
                temporary=args.out/'resume.pt.tmp';torch.save(payload,temporary);os.replace(temporary,args.out/'resume.pt');del payload
                W.atomic_json(args.out/'checkpoint_status.json',dict(step=step,contract_sha256=digest))
            dist.barrier()
    if rank==0:
        W.atomic_json(args.out/'completion.json',dict(execution_complete=True,qualification=args.qualification,step=stop,seed=args.seed,phase=args.phase,checkpoint_sha256=W.file_sha(args.out/'resume.pt'),peak_memory_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
