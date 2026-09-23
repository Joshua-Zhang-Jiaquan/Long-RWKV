"""Stage the public-label encoding diagnostic from matched phase300 parents."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import campaign as C
import emit_jobs as E
from submit_mvp_lookup import live_usage
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--objective',choices=['hard','rao_blackwell'],required=True)
    ap.add_argument('--stop-step',type=int,choices=[320,700,1500],default=700);ap.add_argument('--mask-law',choices=['ordinary','mixture'],required=True);ap.add_argument('--resume',type=Path);ap.add_argument('--submit',action='store_true');a=ap.parse_args()
    parent_plan=ROOT/'results/train04dev'/f'plan_fa94c6878fef276e_{a.objective}_300.json';parent=json.loads(parent_plan.read_text())
    done=json.loads((Path(parent['out'])/'completion.json').read_text());checkpoint=Path(parent['out'])/'resume.pt'
    if not done['execution_complete'] or done['step']!=300 or sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('parent checkpoint incomplete/changed')
    files={rel:Path(parent['stage'])/rel for rel in parent['sources']}
    for rel,p in files.items():
        if sha(p)!=parent['sources'][rel]:raise ValueError('parent source differs')
    for p in sorted((ROOT/'lrwkv_evidence/train04binding').glob('*.py')):files[str(p.relative_to(ROOT))]=p
    for rel in ('lrwkv_evidence/train04mix/core.py','lrwkv_evidence/train04/evaluate.py','qz/launch_train04binding.sh','theory_mvp/train04binding/PROTOCOL.json'):files[rel]=ROOT/rel
    hashes={k:sha(p) for k,p in files.items()};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_train04binding_stages'/digest;out=Path(E.OUTPUTS)/f'train04binding_{digest}_{a.mask_law}_{a.objective}_s{a.stop_step}'
    env=dict(BIND_ROOT=str(stage),BIND_OUT=str(out),BIND_OBJECTIVE=a.objective,BIND_PARENT=str(checkpoint),BIND_STOP=str(a.stop_step),BIND_MASK_LAW=a.mask_law)
    if a.resume:
        if not a.resume.is_file():raise ValueError('resume missing')
        env['BIND_RESUME']=str(a.resume.resolve())
    body=E.make_body(f'lrwkv-bind04-{digest[:8]}-{a.mask_law[:3]}-{a.objective[:4]}-{a.stop_step}',E.wrapped(env,str(stage/'qz/launch_train04binding.sh')),'Public-label scaffold diagnostic;matched phase300 parent and optimizer;8H100 under32cap;320 is discarded qualification',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='7200000')
    record=dict(mask_law=a.mask_law,stage=str(stage),out=str(out),objective=a.objective,stop_step=a.stop_step,parent_plan=str(parent_plan),parent_checkpoint_sha256=done['checkpoint_sha256'],sources=hashes,body=body)
    if a.submit:
        record['capacity']=live_usage(C.list_all_jobs())
        if record['capacity']['live_reserved_gpus']+8>32:raise SystemExit('32GPU cap')
        if C.already_submitted(body['name']):raise SystemExit('already submitted')
        for rel,p in files.items():
            if sha(p)!=hashes[rel]:raise ValueError('source changed')
            dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and sha(dest)!=hashes[rel]:raise ValueError('immutable source conflict')
            if not dest.exists():shutil.copyfile(p,dest)
        record['submission']=C.submit_one(body,dry_run=False)
    dest=ROOT/'results/train04binding'/f'plan_{digest}_{a.mask_law}_{a.objective}_{a.stop_step}.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))
if __name__=='__main__':
    from submission_lock import campaign_submission_lock
    with campaign_submission_lock(ROOT):main()
