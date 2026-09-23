"""Exact terminal-checkpoint evaluation on the frozen confirmation panel."""
import argparse
import json
import os
from pathlib import Path
import torch
from lrwkv_evidence.train04 import tasks,worker as W
from lrwkv_evidence.train04.evaluate import fixed_groups
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04dev.evaluate import endpoint_metrics
from lrwkv_evidence.train04binding import core as B
from . import contract as K

PANEL_SHA='146737f87dba9abe14a4d6cab77cb191cc2507fb132a36276405975d221f6766'
PARTITIONS_SHA='4843768ddcf506d901dcf862ca5c3fd4ea770dececa78ae40efdd983cf64ab93'
METHODS=('one','information_set','random_halves','sequential')


def load_panel(root,manifest):
    paths=[root/'theory_mvp/train04confirm'/name for name in ('PANEL.json','PARTITIONS.json')]
    for path,digest,key in zip(paths,(PANEL_SHA,PARTITIONS_SHA),('panel_sha256','partitions_sha256')):
        if K.sha(path)!=digest or manifest.get(key)!=digest:raise ValueError('confirmation inputs are not frozen')
    panel,partitions=[json.loads(path.read_text()) for path in paths]
    if partitions['panel_sha256']!=PANEL_SHA:raise ValueError('partition panel mismatch')
    groups={r['index']:r for r in partitions['records']}
    if len(groups)!=54 or len(panel['records'])!=54:raise ValueError('incomplete confirmation inputs')
    result=[]
    for record in panel['records']:
        ex=tasks.make_example(panel['split'],panel['seed'],record['index'])
        if ex['instance_id']!=record['instance_id'] or record['instance_id']!=groups[record['index']]['instance_id']:
            raise ValueError('condition identity mismatch')
        import hashlib
        if hashlib.sha256(ex['prompt'].encode()).hexdigest()!=record['prompt_sha256']:raise ValueError('public prompt changed')
        if any(fixed_groups(ex,method)!=groups[record['index']]['groups'][method] for method in METHODS):raise ValueError('reveal groups changed')
        result.append((record,ex))
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('manifest','checkpoint','base','model-root','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    manifest,manifest_sha=K.validate(args.manifest,root,args.model_root)
    if os.environ.get('TRITON_F32_DEFAULT')!='ieee':raise ValueError('frozen inference requires IEEE')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);contract=payload['contract']
    if contract.get('version')!=6 or contract.get('qualification') or contract.get('manifest_sha256')!=manifest_sha or contract.get('phase') not in ('independent','complementary') or contract.get('seed') not in (53,71,89) or payload['step']!=2500:
        raise ValueError('not a frozen terminal confirmation checkpoint')
    if W.canonical_sha(contract)!=payload['contract_sha256']:raise ValueError('checkpoint contract changed')
    if W.file_sha(args.base/'model.safetensors')!=manifest['base_sha256'] or contract['base_sha256']!=manifest['base_sha256']:raise ValueError('base checkpoint mismatch')
    panel=load_panel(root,manifest)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8:raise ValueError('requires eight evaluation ranks')
    torch.set_num_threads(4);torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
    tok,bits,mask=W.load_tokenizer(args.base);model,identity,_=W.build_model(args.model_root,args.base,bits,mask,0)
    model.backbone.lm_head=C.StableBinaryHead(model.backbone.lm_head)
    model.load_state_dict(payload['model'],strict=True);del payload
    model=model.to('cuda').eval();serial=C.SerialDenoiser(model);calls=0
    @torch.inference_mode()
    def predict(ex,visible,stage):
        nonlocal calls
        prefix=tok.encode(ex['prompt']);n=ex['n']
        canvas=dict(ids=prefix+[bits[visible[i]] if i in visible else mask for i in range(n)],prefix=len(prefix),stage=stage,
                    masked=[i not in visible for i in range(n)],gold=[0]*n,oracle=[.5]*n)
        logits=serial(B.pack([B.decorate(ex,canvas,tok)],'cuda'))[0];calls+=1
        return logits.double().log_softmax(-1).cpu().tolist()
    args.out.mkdir(parents=True,exist_ok=True);dest=args.out/f'rank{rank}.jsonl'
    # Exclusive creation prevents replacing previously observed confirmation output.
    with dest.open('x') as stream:
        def emit(row):stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        emit(dict(kind='provenance',rank=rank,world=world,checkpoint_sha256=W.file_sha(args.checkpoint),manifest_sha256=manifest_sha,
                  panel_sha256=PANEL_SHA,partitions_sha256=PARTITIONS_SHA,seed=contract['seed'],phase=contract['phase'],step=2500,
                  triton_f32_default='ieee',torch_tf32=False,scope='Fresh condition pairs within the fixed held-out structure catalog'))
        for position in range(rank,len(panel),world):
            record,ex=panel[position];cache={}
            for method in METHODS:
                emit(dict(kind='endpoint',n=8,index=record['index'],instance_id=record['instance_id'],family=record['family'],structure=record['structure'],
                          **endpoint_metrics(ex,method,predict,cache)))
        emit(dict(kind='complete',actual_audit_calls=calls))


if __name__=='__main__':main()
