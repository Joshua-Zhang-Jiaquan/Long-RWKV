"""Fixed-N8 continuation with matched independent/complementary policy histories."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.train04binding import core as B,worker as Q
from . import core as P


def main():
    ap=argparse.ArgumentParser()
    for name in ('base','model-root','parent','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--mode',choices=['independent','complementary'],required=True)
    ap.add_argument('--updates',type=int,choices=[20,400,1000],required=True)
    ap.add_argument('--resume',type=Path);args=ap.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('requires8ranks')
    torch.set_num_threads(4);torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);dist.init_process_group('nccl');device=torch.device('cuda',local)
    tok,bits,mask=W.load_tokenizer(args.base);model,identity,_=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head);model=model.to(device);packed=C.SerialDenoiser(model)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'completion.json').exists():raise ValueError('completed output exists')
    checks=Q.qualify(model,packed,tok,device)
    parent=torch.load(args.parent,map_location='cpu',weights_only=False);pc=parent['contract']
    if parent['step']!=1500 or pc['version']!=4 or pc['mask_law']!='mixture' or pc['objective']!='rao_blackwell':raise ValueError('wrong shared parent')
    pi={k:v for k,v in pc.items() if k!='sources'};pi['source_hashes']=sorted(pc['sources'].values())
    if W.canonical_sha(pi)!=parent['contract_sha256']:raise ValueError('parent contract mismatch')
    root=Path(__file__).resolve().parents[2];sources={}
    for path,digest in pc['sources'].items():
        original=Path(path)
        if '/model_source/' in path:current=args.model_root/'longrwkv'/original.name
        else:current=root/'lrwkv_evidence'/original.parent.name/original.name
        if W.file_sha(current)!=digest:raise ValueError('parent source mismatch:'+str(current))
        sources[str(current)]=digest
    for path in (Path(__file__),Path(P.__file__),Path(Q.__file__)):
        sources[str(path)]=W.file_sha(path)
    base_sha=W.file_sha(args.base/'model.safetensors')
    if base_sha!=pc['base_sha256']:raise ValueError('parent base mismatch')
    contract=dict(version=5,serialization=pc['serialization'],seed=17,initial_seed=0,objective='rao_blackwell',mask_law='information_set_only',history_coupling=args.mode,parent_checkpoint_sha256=W.file_sha(args.parent),global_batch=32,per_rank_batch=4,conditions_per_update=16,target_dimension=8,terminal_additional_updates=1000,lr=3e-5,weight_decay=.01,optimizer='AdamW beta(.9,.95) eps1e-8; clip1',dtype=pc['dtype'],head=pc['head'],execution=pc['execution'],base_sha256=base_sha,sources=sources)
    ci={k:v for k,v in contract.items() if k!='sources'};ci['source_hashes']=sorted(sources.values());contract_sha=W.canonical_sha(ci)
    ddp=torch.nn.parallel.DistributedDataParallel(packed,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    payload=parent
    if args.resume:
        del payload,parent;payload=torch.load(args.resume,map_location='cpu',weights_only=False)
        if payload['contract_sha256']!=contract_sha:raise ValueError('resume contract mismatch')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);start=payload['step']-1500;del payload
    if not args.resume:del parent
    if not 0<=start<args.updates:raise ValueError('wrong continuation range')
    if rank==0:W.atomic_json(args.out/'provenance.json',dict(contract=contract,contract_sha256=contract_sha,identity=identity,packed_qualification=checks,start_step=1500+start,stop_step=1500+args.updates))
    dist.barrier();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for update in range(start+1,args.updates+1):
        canvases=P.training_canvases(update,rank,tok,args.mode);batch=B.pack(canvases,device)
        optimizer.zero_grad(set_to_none=True);value=M.loss(ddp(batch),batch,'rao_blackwell')
        if not bool(torch.isfinite(value)):raise ValueError('nonfinite loss')
        value.backward();norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.,error_if_nonfinite=True)
        if update==start+1:
            missing=[name for name,p in model.named_parameters() if p.grad is None or p.dtype!=torch.float32 or p.grad.dtype!=torch.float32]
            if missing:raise ValueError('missing or incorrect gradients:'+str(missing[:5]))
        for group in optimizer.param_groups:group['lr']=3e-5
        optimizer.step()
        stats=torch.tensor([float(value),float(M.oracle_entropy(batch)),batch['input_ids'].numel()],device=device,dtype=torch.float64);dist.all_reduce(stats)
        if rank==0:
            record=dict(step=1500+update,additional_update=update,n=8,mean_loss=float(stats[0]/8),oracle_entropy=float(stats[1]/8),global_input_tokens=int(stats[2]),grad_norm_rank0=float(norm),elapsed_seconds=time.perf_counter()-started)
            with (args.out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if update%100==0 or update==args.updates:
            metrics=Q.development(packed,tok,rank,device,8)
            if rank==0:
                with (args.out/'dev.jsonl').open('a') as f:f.write(json.dumps(dict(step=1500+update,metrics=[metrics]))+'\n')
    dist.barrier()
    if rank==0:
        payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),step=1500+args.updates,contract_sha256=contract_sha,contract=contract)
        temporary=args.out/'resume.pt.tmp';torch.save(payload,temporary);os.replace(temporary,args.out/'resume.pt')
        W.atomic_json(args.out/'completion.json',dict(execution_complete=True,objective='rao_blackwell',step=1500+args.updates,checkpoint_sha256=W.file_sha(args.out/'resume.pt'),peak_memory_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
