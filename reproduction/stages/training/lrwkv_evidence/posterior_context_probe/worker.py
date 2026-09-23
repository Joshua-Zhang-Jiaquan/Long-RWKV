"""Selected checkpoint, fresh development conditions, unchanged labelled targets."""
import argparse,json,os,math,time
from pathlib import Path
import torch
import torch.distributed as dist
from . import tasks as T
from .exact import audit
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.train04confirm import contract as K
from lrwkv_evidence.long_context_eval.numerical import run as numerical_screen
from lrwkv_evidence.long_context_eval.runtime import metadata

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','base','model-root','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();root=Path(__file__).resolve().parents[2]
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8 or os.environ.get('TRITON_F32_DEFAULT')!='ieee':raise ValueError('eight IEEE ranks required')
    torch.set_num_threads(4);torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True);dist.init_process_group('nccl')
    manifest,manifest_sha=K.validate(root/'manifest.json',root,args.model_root)
    for rel,digest in json.loads((root/'probe_sources.json').read_text()).items():
        if W.file_sha(root/rel)!=digest:raise ValueError('probe source mismatch')
    checkpoint_sha=W.file_sha(args.checkpoint)
    if checkpoint_sha!='52dbc7f289513e475d10422dbccffa52f74ec55fc887253ea13ae0476dc3e2c9':raise ValueError('wrong selected checkpoint')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);contract=payload['contract']
    if payload['step']!=2500 or W.canonical_sha(contract)!=payload['contract_sha256'] or contract['manifest_sha256']!=manifest_sha:raise ValueError('bad checkpoint lineage')
    if W.file_sha(args.base/'model.safetensors')!=manifest['base_sha256']:raise ValueError('base mismatch')
    tok,bits,mask=W.load_tokenizer(args.base);model,identity,_=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head);model.load_state_dict(payload['model'],strict=True);del payload
    model=model.to('cuda').eval();serial=C.SerialDenoiser(model)
    def predict(ex,encoded,visible):
        prefix=encoded['ids'];n=8
        canvas=dict(ids=prefix+[bits[visible[i]] if i in visible else mask for i in range(n)],prefix=len(prefix),stage=8-len(visible),masked=[i not in visible for i in range(n)],gold=[0]*n,oracle=[.5]*n)
        return serial(B.pack([B.decorate(ex,canvas,tok)],'cuda'))[0].double().log_softmax(-1)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/f'rank{rank}.jsonl').open('x') as stream:
        def emit(row):stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        emit(dict(kind='provenance',rank=rank,world=8,checkpoint_sha256=checkpoint_sha,manifest_sha256=manifest_sha,seed=T.SEED,split='dev',runtime=metadata(torch,torch.device('cuda',local)),selection='seed71 independent selected after frozen confirmation for learned short competence; development only',serialization='original public output labels beside values; original prompt token sequence intact at evidence-only',scope='No final confirmation or broad quality claim'))
        ex=T.example(rank)
        def condition(length,position):
            encoded=T.serialize(ex,tok,length,position);start=time.perf_counter()
            result=audit(ex,lambda visible:predict(ex,encoded,visible).cpu().tolist())
            emit(dict(kind='condition',index=rank,family=ex['family'],serial={k:v for k,v in encoded.items() if k!='ids'},exact=result,audit_seconds=time.perf_counter()-start))
            return result
        short=condition(None,'evidence_only');info=next(r for r in short['results'] if r['method']=='information_set')
        passed=info['exact_valid_mass']>=.9 and info['joint_kl_nats']<=.2
        status=torch.tensor(int(passed),device='cuda');dist.all_reduce(status,op=dist.ReduceOp.MIN)
        emit(dict(kind='gate',local_passed=passed,all_passed=bool(status),criteria='Each condition information-set valid mass >=0.9 and KL<=0.2'))
        if bool(status):
            check=T.example(999)
            def screen(spec):
                visible={} if spec['history']=='all_masked' else {i:0 for i in check['information_set']}
                return predict(check,T.serialize(check,tok,spec['context_tokens'],spec['position']),visible)
            emit(dict(kind='numerical_screen',rows=numerical_screen(screen)))
            for length in (1024,4096,16384):
                for position in ('far','middle','near'):condition(length,position)
        emit(dict(kind='complete',long_sweep_complete=bool(status)))
    dist.destroy_process_group()

if __name__=='__main__':main()
