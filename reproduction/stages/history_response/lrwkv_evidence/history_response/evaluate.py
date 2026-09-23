"""Exhaustive history-by-constraint evidence responses on fixed checkpoints."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from lrwkv_evidence.distance_intervention import tasks as T
from . import design as H

from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.long_context_eval.numerical import run as numerical_screen
from lrwkv_evidence.long_context_eval.runtime import metadata


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','base','model-root','out'): ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args(); root=Path(__file__).resolve().parents[2]
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8 or os.environ.get('TRITON_F32_DEFAULT')!='ieee': raise ValueError('eight IEEE ranks required')
    torch.set_num_threads(4); torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl')
    for rel,digest in json.loads((root/'history_response_sources.json').read_text()).items():
        if W.file_sha(root/rel)!=digest: raise ValueError('frozen evaluation source mismatch: '+rel)
    manifest=json.loads((root/'history_response_manifest.json').read_text())
    checkpoint_sha=W.file_sha(args.checkpoint)
    if checkpoint_sha!=os.environ['DISTANCE_CHECKPOINT_SHA256']: raise ValueError('checkpoint changed')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    contract=payload['contract']
    if W.canonical_sha(contract)!=payload['contract_sha256']: raise ValueError('bad checkpoint contract')
    role=os.environ['DISTANCE_ROLE']
    if checkpoint_sha!=manifest['checkpoints'][role]['sha256']: raise ValueError('wrong frozen selector')
    if role=='original':
        if checkpoint_sha!=manifest['initial_checkpoint_sha256'] or payload['step']!=2500: raise ValueError('wrong original checkpoint')
    else:
        execution=json.loads(Path(os.environ['DISTANCE_EXECUTION']).read_text())
        if payload['step']!=execution['terminal_steps'] or contract['source_sha256']!=execution['source_sha256']:
            raise ValueError('wrong terminal/source selector')
        if role!=f"{contract['arm']}_seed{contract['seed']}": raise ValueError('wrong arm/seed')
        if contract['initial_checkpoint_sha256']!=manifest['initial_checkpoint_sha256']: raise ValueError('wrong selected lineage')
    tok,bits,mask=W.load_tokenizer(args.base)
    model,identity,_=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head)
    model.load_state_dict(payload['model'],strict=True); del payload
    model=model.to('cuda').eval(); serial=C.SerialDenoiser(model)
    def predict(ex,encoded,visible):
        prefix=encoded['ids']
        canvas=dict(ids=prefix+[bits[visible[i]] if i in visible else mask for i in range(8)],
                    prefix=len(prefix),stage=8-len(visible),masked=[i not in visible for i in range(8)],gold=[0]*8,oracle=[.5]*8)
        return serial(B.pack([B.decorate(ex,canvas,tok)],'cuda'))[0].double().log_softmax(-1)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/f'rank{rank}.jsonl').open('x') as stream:
        def emit(row): stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
        emit(dict(kind='provenance',rank=rank,world=8,role=role,checkpoint_sha256=checkpoint_sha,
                  manifest_sha256=W.file_sha(root/'history_response_manifest.json'),seed=T.SEED,followup='history_response_v1',
                  runtime=metadata(torch,torch.device('cuda',local)),identity=identity,
                  selection='post-study diagnostic; reused32problems and seven fixed checkpoints; selected seed71 lineage'))
        check=T.example(0)
        def screen(spec):
            visible={} if spec['history']=='all_masked' else {i:0 for i in check['information_set']}
            return predict(check,T.serialize(check,tok,spec['context_tokens'],spec['position']),visible)
        emit(dict(kind='numerical_screen',rows=numerical_screen(screen)))
        count=0
        for index in range(rank,32,8):
            ex=T.example(index)
            variants=[H.flip(ex,row) for row in range(4)]
            histories=H.histories(ex)
            for position in ('far','middle','near'):
                started=time.perf_counter()
                encoded=T.serialize(ex,tok,16384,position)
                changed=[T.serialize(cf,tok,16384,position) for cf in variants]
                token_indices=[T.validate_layout(encoded,cf) for cf in changed]
                initial=predict(ex,encoded,{}).cpu().tolist()
                original=[];flipped=[[] for _ in range(4)]
                for visible in histories:
                    original.append(predict(ex,encoded,visible).cpu().tolist())
                    for row in range(4):
                        flipped[row].append(predict(variants[row],changed[row],visible).cpu().tolist())
                emit(dict(kind='condition',index=index,position=position,
                          serial={k:v for k,v in encoded.items() if k!='ids'},
                          changed_native_token_indices=token_indices,
                          flipped_token_sha256=[e['token_sha256'] for e in changed],
                          initial=initial,original=original,flipped=flipped,
                          audit_seconds=time.perf_counter()-started))
                count+=1
        emit(dict(kind='complete',conditions=count,all_declared_cells_retained=True))
    dist.destroy_process_group()


if __name__=='__main__': main()
