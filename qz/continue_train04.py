"""Continue the already-authorized bounded study; stop on failures, never auto-retry jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from submit_mvp_lookup import live_usage
ROOT=Path(__file__).resolve().parents[1]

def read(path): return json.loads(path.read_text())

def newest_plan(mode,seed):
    plans=[p for p in (ROOT/'results/train04').glob(f'plan_*_{mode}_s{seed}.json') if read(p).get('submission',{}).get('job_id')]
    return max(plans,key=lambda p:p.stat().st_mtime) if plans else None

def status(plan):
    return C.job_status(read(plan)['submission']['job_id'])[0]

def submit(mode,seed,extra):
    cmd=[sys.executable,str(ROOT/'qz/submit_train04.py'),'--mode',mode,'--seed',str(seed),'--submit',*extra]
    subprocess.run(cmd,cwd=ROOT,check=True)
    return newest_plan(mode,seed)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke-plan',type=Path,required=True);ap.add_argument('--max-hours',type=float,default=4);args=ap.parse_args()
    start=time.monotonic(); history=ROOT/'results/train04/controller.jsonl'
    def note(**row):
        row['utc']=C.timestamp()
        with history.open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    smoke=read(args.smoke_plan); receipt=Path(smoke['out'])/'completion.json'
    while not receipt.exists():
        state=status(args.smoke_plan);note(phase='await_smoke',status=state)
        if state not in C.LIVE_STATUSES: raise SystemExit('smoke terminated without qualification; inspect logs')
        if time.monotonic()-start>args.max_hours*3600: raise SystemExit('controller time bound reached')
        time.sleep(30)
    if read(receipt).get('qualified') is not True: raise SystemExit('smoke failed qualification')
    train={};evaluation={};done=set()
    while len(done)<3:
        for seed in (17,29,43):
            if seed not in train:
                plan=newest_plan('train',seed)
                if plan:train[seed]=plan
                elif live_usage(C.list_all_jobs())['live_reserved_gpus']+8<=32:
                    train[seed]=submit('train',seed,['--smoke',str(receipt)]); note(phase='submitted_train',seed=seed)
            if seed in train:
                result=read(train[seed]);completion=Path(result['out'])/'completion.json'
                if not completion.exists():
                    st=status(train[seed]);note(phase='training',seed=seed,status=st)
                    if st not in C.LIVE_STATUSES: raise SystemExit(f'trainingseed{seed} terminated without completion')
                    continue
                if read(completion).get('qualified') is not True:raise SystemExit('training not qualified')
                if seed not in evaluation:
                    plan=newest_plan('eval',seed)
                    if plan:evaluation[seed]=plan
                    elif live_usage(C.list_all_jobs())['live_reserved_gpus']+8<=32:
                        evaluation[seed]=submit('eval',seed,['--checkpoint',str(Path(result['out'])/'final.pt')]);note(phase='submitted_eval',seed=seed)
            if seed in evaluation and seed not in done:
                plan=read(evaluation[seed]);st=status(evaluation[seed]);note(phase='evaluation',seed=seed,status=st)
                if st not in C.LIVE_STATUSES:
                    if st!='job_succeeded':raise SystemExit(f'evaluationseed{seed} failed: {st}')
                    from lrwkv_evidence.train04.report import aggregate
                    aggregate(Path(plan['out']));done.add(seed);note(phase='evaluation_audited',seed=seed)
        if time.monotonic()-start>args.max_hours*3600:raise SystemExit('controller time bound reached; existing jobs remain bounded')
        if len(done)<3:time.sleep(30)
    directories=[read(evaluation[s])['out'] for s in (17,29,43)]
    subprocess.run([sys.executable,'-m','lrwkv_evidence.train04.report',*directories,'--out',str(ROOT/'results/train04/final')],cwd=ROOT,check=True)
    note(phase='complete',seeds=sorted(done))
if __name__=='__main__':
    sys.path.insert(0,str(ROOT));main()
