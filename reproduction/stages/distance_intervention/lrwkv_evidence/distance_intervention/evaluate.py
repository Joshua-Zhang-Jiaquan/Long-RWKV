"""All fixed endpoints and counterfactuals; no competence-based omissions."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from . import tasks as T
from .diagnostics import stage_errors, evidence_response
from lrwkv_evidence.posterior_context_probe.exact import audit
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
    for rel,digest in json.loads((root/'distance_eval_sources.json').read_text()).items():
        if W.file_sha(root/rel)!=digest: raise ValueError('frozen evaluation source mismatch: '+rel)
    manifest=json.loads((root/'distance_eval_manifest.json').read_text())
    checkpoint_sha=W.file_sha(args.checkpoint)
    if checkpoint_sha!=os.environ['DISTANCE_CHECKPOINT_SHA256']: raise ValueError('checkpoint changed')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    contract=payload['contract']
    if W.canonical_sha(contract)!=payload['contract_sha256']: raise ValueError('bad checkpoint contract')
    role=os.environ['DISTANCE_ROLE']
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
                  manifest_sha256=W.file_sha(root/'distance_eval_manifest.json'),seed=T.SEED,
                  runtime=metadata(torch,torch.device('cuda',local)),identity=identity,
                  selection='three adaptation seeds of selected seed71 lineage; fresh conditions within previously observed test catalog'))
        check=T.example(0)
        def screen(spec):
            visible={} if spec['history']=='all_masked' else {i:0 for i in check['information_set']}
            return predict(check,T.serialize(check,tok,spec['context_tokens'],spec['position']),visible)
        emit(dict(kind='numerical_screen',rows=numerical_screen(screen)))
        # The untrained longer-length probe also receives repeat/cross-rank checks.
        for position in ('far','near'):
            encoded=T.serialize(check,tok,32768,position)
            a=predict(check,encoded,{}); b=predict(check,encoded,{})
            values=[torch.empty_like(a) for _ in range(8)]; dist.all_gather(values,a)
            within=float((a-b).abs().max()); cross=max(float((v-values[0]).abs().max()) for v in values)
            if within>1e-5 or cross>1e-4: raise ValueError('32K numerical instability')
            emit(dict(kind='numerical_32k',position=position,within=within,cross=cross))
        def condition(index,length,position,primary):
            ex=T.example(index); encoded=T.serialize(ex,tok,length,position); started=time.perf_counter(); cache={}
            def cached(visible):
                key=tuple(sorted(visible.items()))
                if key not in cache: cache[key]=predict(ex,encoded,visible).cpu().tolist()
                return cache[key]
            exact=audit(ex,cached); stages=stage_errors(ex,cached)
            info=next(r for r in exact['results'] if r['method']=='information_set')
            if abs(stages['total_estimation_nats']-info['estimation_nats'])>1e-8: raise ValueError('stage decomposition mismatch')
            cf,meta=T.counterfactual(ex); flipped=T.serialize(cf,tok,length,position)
            token_index=T.validate_layout(encoded,flipped)
            original_lp=cached(meta['visible']); flipped_lp=predict(cf,flipped,meta['visible']).cpu().tolist()
            response=evidence_response(original_lp,flipped_lp,meta)
            emit(dict(kind='condition',index=index,primary=primary,
                      serial={k:v for k,v in encoded.items() if k!='ids'}, exact=exact, stages=stages,
                      counterfactual=dict(meta=meta,changed_native_token_index=token_index,
                                          flipped_token_sha256=flipped['token_sha256'],response=response),
                      audit_seconds=time.perf_counter()-started))
            if primary and index<8:
                # No audit cache: repeated serial requests, synchronized end-to-end.
                for method,groups in T.partitions(ex).items():
                    times=[]; peaks=[]
                    for repeat in range(6):
                        generator=torch.Generator(device='cuda'); generator.manual_seed(20271023+index*100+repeat)
                        visible={}; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                        for group in groups:
                            lp=predict(ex,encoded,visible)
                            draws=torch.multinomial(lp.exp(),1,generator=generator).flatten().cpu().tolist()
                            visible.update({i:draws[i] for i in group})
                        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                        if repeat>0: times.append(elapsed); peaks.append(torch.cuda.max_memory_allocated())
                    emit(dict(kind='timing',index=index,position=position,method=method,warmups=1,
                              seconds=times,peak_allocated_bytes=peaks,scope='16K single-request FP32/IEEE; five repeats; no audit cache'))
        for index in range(rank,32,8):
            for position in ('far','middle','near'): condition(index,16384,position,True)
        condition(rank,None,'evidence_only',False)
        for position in ('far','middle','near'): condition(rank,4096,position,False)
        for position in ('far','near'): condition(rank,32768,position,False)
        emit(dict(kind='complete',conditions=18,all_declared_cells_retained=True))
    dist.destroy_process_group()


if __name__=='__main__': main()
