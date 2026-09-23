"""Immutable staged 8-H100 jobs; max32 concurrent queued+running, bounded2h per job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E
from submit_mvp_lookup import live_usage
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','train','eval','exact','probe'],required=True)
    ap.add_argument('--seed',type=int,choices=[17,29,43],default=17)
    ap.add_argument('--smoke',type=Path); ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--submit',action='store_true'); args=ap.parse_args()
    if args.mode=='train' and not args.smoke: ap.error('--smoke required')
    if args.mode in ('eval','probe') and not args.checkpoint: ap.error('--checkpoint required')
    files={str(p.relative_to(ROOT)):p for p in [ROOT/'lrwkv_evidence/__init__.py',*sorted((ROOT/'lrwkv_evidence/train04').glob('*.py')),ROOT/'qz/launch_train04.sh',ROOT/'theory_mvp/train04/PROTOCOL.json']}
    files.update({'model_source/longrwkv/'+p.name:p for p in sorted((ROOT.parent/'rwkv04b/longrwkv').glob('*.py'))})
    if args.mode=='exact': files['exact_manifest.json']=ROOT/'results/train04/exact_manifest.json'
    hashes={rel:hashlib.sha256(p.read_bytes()).hexdigest() for rel,p in files.items()}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_train04_stages'/digest
    out=Path(E.OUTPUTS)/f'train04_{digest}_{args.mode}_s{args.seed}'
    exports={'TRAIN04_ROOT':str(stage),'TRAIN04_OUT':str(out),'TRAIN04_MODE':args.mode,'TRAIN04_SEED':str(args.seed)}
    if args.smoke: exports['TRAIN04_SMOKE']=str(args.smoke.resolve())
    if args.checkpoint: exports['TRAIN04_CHECKPOINT']=str(args.checkpoint.resolve())
    body=E.make_body(f'lrwkv-train04-{digest[:10]}-{args.mode}-s{args.seed}',E.wrapped(exports,str(stage/'qz/launch_train04.sh')),
        'User-authorized controlled0.4B tiedA1 adaptation; 20step qualification/500step training/locked test; 32H100 cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='7200000')
    if args.mode=='exact': body['description']='Post-primary exact probability audit of both two-call policies for locked training seeds17/29/43; no training or tuning;8H100 under32cap'
    if args.mode=='probe': body['description']='Frozen seed43 forward-repeat and precision diagnostic on8H100; no training; within32GPUcap'
    record=dict(mode=args.mode,seed=args.seed,stage=str(stage),out=str(out),source_hashes=hashes,body=body)
    destination=ROOT/'results/train04'/f'plan_{digest}_{args.mode}_s{args.seed}.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs())
        if record['capacity']['live_reserved_gpus']+8>32: raise SystemExit('32 concurrent GPU cap exceeded')
        if C.already_submitted(body['name']): raise SystemExit('Exact job already submitted; inspect ledger')
        if args.mode=='train':
            receipt=json.loads(args.smoke.read_text())
            if not receipt.get('qualified') or receipt.get('steps')!=20: raise SystemExit('smoke not qualified')
        if args.checkpoint and not args.checkpoint.is_file(): raise SystemExit('checkpoint missing')
        for rel,p in files.items():
            if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[rel]: raise SystemExit('source changed during staging')
            dest=stage/rel; dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=hashes[rel]: raise SystemExit('stage conflict')
            if not dest.exists(): shutil.copyfile(p,dest)
        record['submission']=C.submit_one(body,dry_run=False)
    destination.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('source_hashes','body')},indent=2))
if __name__=='__main__': main()
