"""Separate source calibration and sealed-target evaluation invocations."""
import argparse,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from . import tasks as T
from . import training as R
from . import prediction as P
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.long_context_eval.runtime import metadata


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','base','model-root','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--split',choices=('calibration','heldout'),required=True)
    args=ap.parse_args();root=Path(__file__).resolve().parents[2]
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8 or os.environ.get('TRITON_F32_DEFAULT')!='ieee':raise ValueError('eight IEEE ranks')
    torch.set_num_threads(4);torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl')
    for rel,digest in json.loads((root/'transfer_eval_sources.json').read_text()).items():
        if W.file_sha(root/rel)!=digest:raise ValueError('evaluation source mismatch '+rel)
    protocol_path=Path(os.environ['TRANSFER_PROTOCOL']);protocol=json.loads(protocol_path.read_text())
    if protocol['status']!='frozen':raise ValueError('unfrozen protocol')
    checkpoint_sha=W.file_sha(args.checkpoint)
    if checkpoint_sha!=os.environ['TRANSFER_CHECKPOINT_SHA256']:raise ValueError('checkpoint mismatch')
    payload=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    contract=payload['contract'];seed=contract['seed']
    if payload['step']!=3100 or contract['qualification'] or seed not in R.SEEDS:raise ValueError('not a fixed terminal lineage')
    if contract['protocol_sha256']!=W.file_sha(protocol_path):raise ValueError('training protocol changed')
    if args.split=='heldout':
        path=Path(os.environ['TRANSFER_PREDICTIONS'])
        if W.file_sha(path)!=os.environ['TRANSFER_PREDICTIONS_SHA256']:raise ValueError('unsealed predictions')
        pred=json.loads(path.read_text())
        if pred['checkpoint_sha256']!=checkpoint_sha or pred['protocol_sha256']!=W.file_sha(protocol_path):raise ValueError('wrong prediction selector')
        if [p['cell'] for p in pred['predictions']]!=P.panel('heldout'):raise ValueError('incomplete predictions')
    tok,bits,mask=W.load_tokenizer(args.base)
    model,identity,_=W.build_model(args.model_root,args.base,bits,mask,seed)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head)
    model.load_state_dict(payload['model'],strict=True);del payload
    serial=C.SerialDenoiser(model.to('cuda').eval())
    def predict(ex,policy,round_index,position,bits=None):
        canvas=R.canvas(ex,tok,policy,round_index,16384,position,bits)
        return serial(R.batch([canvas],'cuda'))[0].double().log_softmax(-1)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/f'rank{rank}.jsonl').open('x') as stream:
        def emit(row):stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        emit(dict(kind='provenance',rank=rank,world=8,seed=seed,split=args.split,checkpoint_sha256=checkpoint_sha,
                  protocol_sha256=W.file_sha(protocol_path),runtime=metadata(torch,torch.device('cuda',local)),identity=identity))
        # Numerical screens use a source qualification context even for held-out execution.
        ex=T.make_example('qualification',202709230,999)
        for position in P.POSITIONS:
            for rd in (0,1):
                a=predict(ex,0,rd,position);b=predict(ex,0,rd,position)
                all_values=[torch.empty_like(a) for _ in range(8)];dist.all_gather(all_values,a)
                within=float((a-b).abs().max());cross=max(float((v-all_values[0]).abs().max()) for v in all_values)
                if not torch.isfinite(a).all() or within>1e-5 or cross>1e-4:raise ValueError('numerical instability')
                emit(dict(kind='numerical_screen',position=position,round=rd,within=within,cross=cross))
        count=0
        for cell in P.panel(args.split)[rank::8]:
            ex=P.example(cell);ys=T.support(ex);began=time.perf_counter()
            initial=predict(ex,0,0,cell['position']).cpu().tolist()
            conditional=[[predict(ex,p,1,cell['position'],y).cpu().tolist() for y in ys] for p in (0,1)]
            emit(dict(kind='condition',cell=cell,instance_id=ex['instance_id'],initial=initial,conditional=conditional,
                      metrics=P.metrics(ex,initial,conditional),seconds=time.perf_counter()-began));count+=1
        emit(dict(kind='complete',conditions=count))
    dist.destroy_process_group()

if __name__=='__main__':main()
