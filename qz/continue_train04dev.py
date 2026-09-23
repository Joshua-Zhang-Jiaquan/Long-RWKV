"""Bounded watcher: audit fixed development checkpoints once complete; never choose a model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from submit_mvp_lookup import live_usage
ROOT=Path(__file__).resolve().parents[1]
PLANS=[ROOT/'results/train04dev'/f'plan_fa94c6878fef276e_{arm}_300.json' for arm in ('hard','rao_blackwell')]
STATE=ROOT/'results/train04dev/audit_watcher.json'

def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)


def main():
    global PLANS,STATE
    parser=argparse.ArgumentParser();parser.add_argument('--plans',type=Path,nargs='+');parser.add_argument('--state',type=Path)
    args=parser.parse_args()
    if args.plans:
        if args.state is None:raise ValueError('custom plans require a distinct state file')
        PLANS=[p.resolve() for p in args.plans];STATE=args.state.resolve();STATE.parent.mkdir(parents=True,exist_ok=True)
    started=time.time();state=dict(started_unix=started,status='waiting',arms={})
    for _ in range(180):
        all_terminal=True
        for path in PLANS:
            plan=json.loads(path.read_text());arm=Path(plan['stage']).name+'/'+plan['objective']+('/'+plan['mask_law'] if 'mask_law' in plan else '')
            # A submitted audit receipt is authoritative for idempotent submission.
            audits=[]
            for receipt in (ROOT/'results/train04dev').glob('plan_eval_*.json'):
                r=json.loads(receipt.read_text())
                if r['training_plan']==str(path.resolve()) and not r.get('numerical_qualification',False) and r.get('triton_precision','default')=='default' and r.get('submission',{}).get('job_id'):audits.append(r)
            if audits:
                state['arms'][arm]=dict(status='audit_submitted',job_id=audits[0]['submission']['job_id'],out=audits[0]['out']);continue
            done=Path(plan['out'])/'completion.json'
            if not done.exists():
                status,_=C.job_status(plan['submission']['job_id'])
                state['arms'][arm]=dict(status='waiting_training',scheduler_status=status)
                if status in ('job_failed','job_stopped','job_cancelled','job_canceled','job_succeeded'):
                    state['arms'][arm]['status']='terminal_without_completion'
                else:all_terminal=False
                continue
            all_terminal=False
            usage=live_usage(C.list_all_jobs())
            if usage['live_reserved_gpus']+8>32:
                state['arms'][arm]=dict(status='waiting_capacity',reserved=usage['live_reserved_gpus']);continue
            run=subprocess.run([sys.executable,str(ROOT/'qz/submit_train04dev_eval.py'),'--training-plan',str(path),'--submit'],cwd=ROOT,text=True,capture_output=True)
            state['arms'][arm]=dict(status='submission_returned',returncode=run.returncode,tail=(run.stdout+run.stderr)[-1200:])
            if run.returncode:
                state['status']='submission_error';save(state);raise SystemExit(run.returncode)
        state['updated_unix']=time.time();save(state)
        if all_terminal:
            state['status']='finished';save(state);return
        time.sleep(20)
    state['status']='watch_timeout_no_resubmission';save(state)

if __name__=='__main__':main()
