"""Bounded matched development jobs, eight H100s each, no test-driven selection."""
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
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--objective',choices=['hard','rao_blackwell'],required=True)
 ap.add_argument('--stop-step',type=int,choices=[300,700,1500],default=300);ap.add_argument('--resume',type=Path);ap.add_argument('--submit',action='store_true');a=ap.parse_args()
 files={str(p.relative_to(ROOT)):p for p in [ROOT/'lrwkv_evidence/__init__.py',ROOT/'lrwkv_evidence/train04/worker.py',ROOT/'lrwkv_evidence/train04/tasks.py',*sorted((ROOT/'lrwkv_evidence/train04dev').glob('*.py')),ROOT/'qz/launch_train04dev.sh',ROOT/'theory_mvp/train04dev/PROTOCOL.json']}
 files.update({'model_source/longrwkv/'+p.name:p for p in sorted((ROOT.parent/'rwkv04b/longrwkv').glob('*.py'))})
 hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
 stage=Path(E.G)/'long_rwkv_train04dev_stages'/digest;out=Path(E.OUTPUTS)/f'train04dev_{digest}_{a.objective}_s{a.stop_step}'
 env={'DEV_ROOT':str(stage),'DEV_OUT':str(out),'DEV_OBJECTIVE':a.objective,'DEV_STOP':str(a.stop_step)}
 if a.resume:
  if not a.resume.is_file():raise ValueError('resume missing')
  env['DEV_RESUME']=str(a.resume.resolve())
 body=E.make_body(f'lrwkv-dev04-{digest[:8]}-{a.objective[:4]}-{a.stop_step}',E.wrapped(env,str(stage/'qz/launch_train04dev.sh')),'Matched hard/Rao-Blackwell FP32 packed curriculum development; same finalN8task; no locked test;8H100 under32cap',E.SPEC_8GPU)
 body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='7200000')
 record=dict(stage=str(stage),out=str(out),objective=a.objective,stop_step=a.stop_step,sources=hashes,body=body)
 if a.submit:
  census=live_usage(C.list_all_jobs());record['capacity']=census
  if census['live_reserved_gpus']+8>32:raise SystemExit('32GPU cap')
  if C.already_submitted(body['name']):raise SystemExit('already submitted')
  for rel,p in files.items():
   if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[rel]:raise ValueError('source changed')
   dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=hashes[rel]:raise ValueError('immutable stage conflict')
   if not dest.exists():shutil.copyfile(p,dest)
  record['submission']=C.submit_one(body,dry_run=False)
 dest=ROOT/'results/train04dev'/f'plan_{digest}_{a.objective}_{a.stop_step}.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))
if __name__=='__main__':
    from submission_lock import campaign_submission_lock
    with campaign_submission_lock(ROOT):main()
