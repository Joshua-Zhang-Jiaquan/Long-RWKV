"""Immutable paired-history stages with shared 32-H100 capacity enforcement."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import campaign as C
import emit_jobs as E
from submit_mvp_lookup import live_usage
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['independent','complementary'],required=True);ap.add_argument('--updates',type=int,choices=[20,400,1000],required=True);ap.add_argument('--resume',type=Path);ap.add_argument('--submit',action='store_true');a=ap.parse_args()
    parent_path=ROOT/'results/train04binding/plan_86b2795e56d63506_mixture_rao_blackwell_1500.json';parent=json.loads(parent_path.read_text());checkpoint=Path(parent['out'])/'resume.pt';done=json.loads((Path(parent['out'])/'completion.json').read_text())
    if not done['execution_complete'] or done['step']!=1500 or sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('parent incomplete or changed')
    files={rel:Path(parent['stage'])/rel for rel in parent['sources']}
    for rel,p in files.items():
        if sha(p)!=parent['sources'][rel]:raise ValueError('parent source changed')
    for p in sorted((ROOT/'lrwkv_evidence/train04paired').glob('*.py')):files[str(p.relative_to(ROOT))]=p
    for rel in ('qz/launch_train04paired.sh','theory_mvp/train04paired/PLAN.md'):files[rel]=ROOT/rel
    hashes={rel:sha(p) for rel,p in files.items()};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_train04paired_stages'/digest;out=Path(E.OUTPUTS)/f'train04paired_{digest}_{a.mode}_u{a.updates}'
    env=dict(PAIR_ROOT=str(stage),PAIR_OUT=str(out),PAIR_PARENT=str(checkpoint),PAIR_MODE=a.mode,PAIR_UPDATES=str(a.updates))
    if a.resume:
        if not a.resume.is_file():raise ValueError('resume missing')
        env['PAIR_RESUME']=str(a.resume.resolve())
    body=E.make_body(f'lrwkv-pair04-{digest[:8]}-{a.mode[:4]}-{a.updates}',E.wrapped(env,str(stage/'qz/launch_train04paired.sh')),'Matched N8 policy histories;20update probes discarded;8H100 under32cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='7200000')
    record=dict(history_coupling=a.mode,mask_law='information_set_only/'+a.mode,objective='rao_blackwell',stage=str(stage),out=str(out),stop_step=1500+a.updates,parent_plan=str(parent_path),parent_checkpoint_sha256=done['checkpoint_sha256'],sources=hashes,body=body)
    if a.submit:
        record['capacity']=live_usage(C.list_all_jobs())
        if record['capacity']['live_reserved_gpus']+8>32:raise SystemExit('32GPU cap')
        if C.already_submitted(body['name']):raise SystemExit('already submitted')
        for rel,p in files.items():
            if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
            dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and sha(dest)!=hashes[rel]:raise ValueError('immutable stage conflict')
            if not dest.exists():shutil.copyfile(p,dest)
        record['submission']=C.submit_one(body,dry_run=False)
    dest=ROOT/'results/train04paired'/f'plan_{digest}_{a.mode}_{a.updates}.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))

if __name__=='__main__':
    from submission_lock import campaign_submission_lock
    with campaign_submission_lock(ROOT):main()
