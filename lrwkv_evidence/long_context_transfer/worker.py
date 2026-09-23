"""Controlled warm start from the validated seed71 posterior conditioner."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from lrwkv_evidence.long_context_mvp import training as R,tasks as T
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C


@torch.inference_mode()
def qualify(packed,tok,device):
    packed.eval();checks=[]
    for step in (1,201):
        records=R.records(step,0,tok);b=R.batch(records,device);normal=packed(b)
        singles=torch.cat([packed(R.batch([r],device)) for r in records])
        poisoned=dict(b);poisoned['input_ids']=b['input_ids'].clone()
        poisoned['input_ids'][0,:int(b['doc_starts'][0,1])]=tok.binary_ids[1]
        changed=packed(poisoned)
        row=dict(context='evidence_only' if step==1 else 1024,
                 serial_difference=float((normal-singles).abs().max()),
                 other_document_difference=float((normal[1:]-changed[1:]).abs().max()),
                 own_document_difference=float((normal[0]-changed[0]).abs().max()))
        if row['serial_difference']>1e-5 or row['other_document_difference']>1e-5 or row['own_document_difference']<1e-6:
            raise ValueError(f'isolation qualification failed: {row}')
        checks.append(row)
    packed.train();return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--model-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--qualification',action='store_true');ap.add_argument('--qualification-out',type=Path)
    ap.add_argument('--initial-checkpoint',type=Path,required=True);ap.add_argument('--initial-sha256',required=True)
    args=ap.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('requires eight ranks')
    torch.set_num_threads(4);torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl');device=torch.device('cuda',local)
    root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__),Path(R.__file__),Path(T.__file__),Path(W.__file__),Path(C.__file__),
           Path(T.oracle_marginals.__code__.co_filename),*sorted((args.model_root/'longrwkv').glob('*.py'))]
    if W.file_sha(args.initial_checkpoint)!=args.initial_sha256:raise ValueError('initial checkpoint checksum mismatch')
    recipe=dict(R.RECIPE,initialization='Auxiliary posterior seed71 independent fixed2500; fresh adaptation optimizer')
    contract=dict(recipe=recipe,initial_checkpoint_sha256=args.initial_sha256,base_sha256=W.file_sha(args.base/'model.safetensors'),
                  source_hashes={str(p.relative_to(root)):W.file_sha(p) for p in paths})
    digest=W.canonical_sha(contract)
    if not args.qualification:
        if not args.qualification_out:raise ValueError('passed qualification required')
        q=json.loads((args.qualification_out/'completion.json').read_text())
        if not q.get('qualification') or not q.get('execution_complete') or q.get('contract_sha256')!=digest or q.get('step')!=20:
            raise ValueError('qualification does not match source/recipe')
        for i in range(8):
            evidence=json.loads((args.qualification_out/f'qualification_rank{i}.json').read_text())
            if evidence['contract_sha256']!=digest:raise ValueError('rank qualification mismatch')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'train.jsonl').exists() or (args.out/'completion.json').exists():raise ValueError('refuse reused output')
    tok,bits,mask=W.load_tokenizer(args.base);model,identity,_=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head)
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if initial['step']!=2500 or initial['contract']['seed']!=71 or initial['contract']['phase']!='independent':
        raise ValueError('wrong auxiliary conditioner')
    model.load_state_dict(initial['model'],strict=True);del initial
    identity['initial_checkpoint_sha256']=args.initial_sha256
    model=model.to(device);packed=C.SerialDenoiser(model)
    checks=qualify(packed,tok,device)
    W.atomic_json(args.out/f'qualification_rank{rank}.json',dict(contract_sha256=digest,checks=checks))
    ddp=torch.nn.parallel.DistributedDataParallel(packed,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    stop=20 if args.qualification else 600
    if rank==0:W.atomic_json(args.out/'provenance.json',dict(contract=contract,contract_sha256=digest,identity=identity,qualification=args.qualification,stop_step=stop))
    dist.barrier();started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(1,stop+1):
        # Discarded qualification tests both short and 1K training graphs.
        data_step=step if not args.qualification or step<=10 else 200+step-10
        b=R.batch(R.records(data_step,rank,tok),device);optimizer.zero_grad(set_to_none=True)
        value=R.loss(ddp(b),b)
        if not bool(torch.isfinite(value)):raise ValueError('nonfinite loss')
        value.backward();norm=torch.nn.utils.clip_grad_norm_(ddp.parameters(),1.,error_if_nonfinite=True)
        if step in (1,11,201):
            missing=[name for name,p in model.named_parameters() if p.grad is None or p.dtype!=torch.float32 or p.grad.dtype!=torch.float32]
            if missing:raise ValueError('missing or wrong dtype gradients: '+str(missing[:5]))
        lr=3e-5*min(step/50,1.)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        values=torch.tensor([value.detach().double(),b['input_ids'].numel()],dtype=torch.float64,device=device);dist.all_reduce(values)
        if rank==0:
            row=dict(step=step,data_step=data_step,loss=float(values[0]/8),global_input_tokens=int(values[1]),grad_norm_rank0=float(norm),lr=lr,elapsed_seconds=time.perf_counter()-started)
            with (args.out/'train.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
        if step==stop or (not args.qualification and step==200):
            dist.barrier()
            if rank==0:
                tmp=args.out/'resume.pt.tmp';torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),step=step,contract=contract,contract_sha256=digest),tmp);os.replace(tmp,args.out/'resume.pt')
            dist.barrier()
    if rank==0:W.atomic_json(args.out/'completion.json',dict(execution_complete=True,qualification=args.qualification,step=stop,contract_sha256=digest,checkpoint_sha256=W.file_sha(args.out/'resume.pt'),peak_memory_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
