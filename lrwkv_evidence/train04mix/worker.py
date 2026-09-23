"""Matched hard/Rao-Blackwell development curriculum; no locked test access."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from lrwkv_evidence.train04 import worker as W,tasks
from lrwkv_evidence.train04dev import core as C, worker as P
from lrwkv_evidence.train04 import evaluate as E
from . import core as M


@torch.inference_mode()
def qualify(model,packed,tok,device):
    model.eval();checks=[]
    for n in (2,4,8):
        canvases=[C.make_canvas(C.example('train',17,i,n),tok,17,stage=4) for i in range(4)]
        batch=C.pack(canvases,device);together=packed(batch)
        separate=torch.cat([packed(C.pack([c],device)) for c in canvases])
        difference=float((together-separate).abs().max())
        poisoned=dict(batch);poisoned['input_ids']=batch['input_ids'].clone()
        poisoned['input_ids'][0,:int(batch['doc_starts'][0,1])]=tok.binary_ids[1]
        changed=packed(poisoned)
        leak=float((changed[1:]-together[1:]).abs().max());own=float((changed[0]-together[0]).abs().max())
        row=dict(n=n,packed_serial_max_logit_difference=difference,other_document_max_difference=leak,poisoned_document_difference=own)
        checks.append(row)
        if difference>1e-3 or leak>1e-5 or own<1e-6:raise ValueError(f'packed qualification failed: {row}')
    model.train();return checks


@torch.inference_mode()
def development(packed,tok,rank,device,n):
    packed.eval();stats=torch.zeros(7,dtype=torch.float64,device=device)
    canvases=[]
    for index in range(rank,64,8):
        ex=C.example('dev',51917,index,n)
        for stage in range(1,9):canvases.append(C.make_canvas(ex,tok,90210,stage=stage))
    for start in range(0,len(canvases),4):
        batch=C.pack(canvases[start:start+4],device);lp=packed(batch).double().log_softmax(-1);p=batch['oracle'].double();mask=batch['masked']
        entropy=torch.where((p==0)|(p==1),0.,-p*torch.log(p.clamp_min(1e-300))-(1-p)*torch.log((1-p).clamp_min(1e-300)))
        ce=-(p*lp[...,1]+(1-p)*lp[...,0]);kl=ce-entropy
        deterministic=mask&(p!=.5);fair=mask&(p==.5);correct=(lp[...,1]>=lp[...,0])==(p==1)
        stats+=torch.stack([(kl*mask).sum(),mask.sum(),(correct&deterministic).sum(),deterministic.sum(),
                            ((lp[...,1].exp()-.5).abs()*fair).sum(),fair.sum(),((kl*mask).sum(-1)*8/(batch['stage']*n)).sum()])
    dist.all_reduce(stats);v=stats.cpu().tolist();packed.train()
    return dict(n=n,conditions=64,marginal_kl_per_masked_bit=v[0]/v[1],deterministic_accuracy=v[2]/v[3],fair_probability_absolute_error=v[4]/v[5],elbo_excess_per_token=v[6]/512)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--objective',choices=['hard','rao_blackwell'],required=True)
    ap.add_argument('--stop-step',type=int,choices=[700,1500],default=700);ap.add_argument('--resume',type=Path);ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--model-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('requires8ranks')
    torch.set_num_threads(4);torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);dist.init_process_group('nccl');device=torch.device('cuda',local)
    tok,bits,mask=W.load_tokenizer(args.base);model,identity,config=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head)
    model=model.to(device);packed=C.SerialDenoiser(model)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'completion.json').exists():raise ValueError('completed output exists')
    checks=qualify(model,packed,tok,device)
    source_files=[Path(__file__),Path(C.__file__),Path(M.__file__),Path(E.__file__),Path(P.__file__),Path(W.__file__),Path(tasks.__file__),*sorted((args.model_root/'longrwkv').glob('*.py'))]
    parent=torch.load(args.parent,map_location='cpu',weights_only=False)
    parent_contract=parent['contract']
    parent_identity={k:v for k,v in parent_contract.items() if k!='sources'}
    parent_identity['source_hashes']=sorted(parent_contract['sources'].values())
    expected_parent_sources=[Path(P.__file__),Path(C.__file__),Path(W.__file__),Path(tasks.__file__),*sorted((args.model_root/'longrwkv').glob('*.py'))]
    if parent['step']!=300 or parent_contract['version']!=2 or parent_contract['objective']!=args.objective:raise ValueError('wrong phase300 parent')
    if W.canonical_sha(parent_identity)!=parent['contract_sha256']:raise ValueError('parent contract corrupted')
    if sorted(parent_contract['sources'].values())!=sorted(W.file_sha(p) for p in expected_parent_sources):raise ValueError('parent implementation differs')
    if parent_contract['base_sha256']!=W.file_sha(args.base/'model.safetensors'):raise ValueError('parent base differs')
    parent_sha=W.file_sha(args.parent)
    contract=dict(version=3,seed=17,initial_seed=0,objective=args.objective,mask_law='half standard corruption, half uniform information-set round; score reveal group only',parent_checkpoint_sha256=parent_sha,global_batch=32,per_rank_batch=4,
                  dtype='float32 throughout; TF32 disabled in torch; deterministic algorithms requested',head='FP32 binary log-odds of original head rows',execution='separate document forwards',
                  curriculum={'1-300':2,'301-700':4,'701-1500':8},lr=3e-5,warmup_updates=50,weight_decay=.01,
                  optimizer='AdamW beta(.9,.95) eps1e-8; clip1',base_sha256=W.file_sha(args.base/'model.safetensors'),
                  sources={str(p):W.file_sha(p) for p in source_files})
    # Paths change across immutable stages; compare source contents when resuming.
    identity_contract={k:v for k,v in contract.items() if k!='sources'};identity_contract['source_hashes']=sorted(contract['sources'].values())
    contract_sha=W.canonical_sha(identity_contract)
    ddp=torch.nn.parallel.DistributedDataParallel(packed,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    payload=parent
    if args.resume:
        del payload,parent
        payload=torch.load(args.resume,map_location='cpu',weights_only=False)
        if payload['contract_sha256']!=contract_sha:raise ValueError('resume contract differs')
    model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);start_step=payload['step'];del payload
    if not args.resume:del parent
    if args.stop_step<=start_step:raise ValueError('stop must follow resume')
    if rank==0:W.atomic_json(args.out/'provenance.json',dict(contract=contract,contract_sha256=contract_sha,identity=identity,packed_qualification=checks,start_step=start_step,stop_step=args.stop_step))
    dist.barrier();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(start_step+1,args.stop_step+1):
        n=C.curriculum_n(step)
        examples=[C.example('train',17,(step-1)*32+rank*4+j,n) for j in range(4)]
        canvases=[M.canvas(e,tok,17) for e in examples];batch=M.pack(canvases,device)
        optimizer.zero_grad(set_to_none=True);value=M.loss(ddp(batch),batch,args.objective)
        if not bool(torch.isfinite(value)):raise ValueError('nonfinite loss')
        value.backward();norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.,error_if_nonfinite=True)
        if step==start_step+1:
            missing=[name for name,p in model.named_parameters() if p.grad is None or p.dtype!=torch.float32 or p.grad.dtype!=torch.float32]
            if missing:raise ValueError(f'missing/incorrect gradient {missing[:5]}')
        rate=3e-5*min(step/50,1.)
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.step()
        values=torch.tensor([float(value),int(batch['masked'].sum()),int((batch['masked'].sum(-1)==0).sum()),float(M.oracle_entropy(batch)),int(batch['policy_branch'].sum())],device=device,dtype=torch.float64);dist.all_reduce(values)
        if rank==0:
            record=dict(step=step,n=n,mean_loss=float(values[0]/8),masked_targets=int(values[1]),empty_masks=int(values[2]),lr=rate,grad_norm_rank0=float(norm),oracle_entropy=float(values[3]/8),policy_examples=int(values[4]),elapsed_seconds=time.perf_counter()-started)
            with (args.out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if step%100==0 or step==args.stop_step:
            metrics=[development(packed,tok,rank,device,d) for d in sorted({n,8})]
            if rank==0:
                with (args.out/'dev.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,metrics=metrics))+'\n')
                print(json.dumps(dict(step=step,development=metrics)),flush=True)
    dist.barrier()
    if rank==0:
        payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),step=args.stop_step,contract_sha256=contract_sha,contract=contract)
        temporary=args.out/'resume.pt.tmp';torch.save(payload,temporary);os.replace(temporary,args.out/'resume.pt')
        W.atomic_json(args.out/'completion.json',dict(execution_complete=True,objective=args.objective,step=args.stop_step,checkpoint_sha256=W.file_sha(args.out/'resume.pt'),peak_memory_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
