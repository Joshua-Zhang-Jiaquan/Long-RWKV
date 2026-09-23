"""Wait for released capacity, then run and collect the fixed cost supplement."""
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from confirm_capacity import live_usage
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'results/distance_intervention_cost'


def run(script,*args):
    r=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,text=True,capture_output=True)
    print(r.stdout+r.stderr,flush=True)
    if r.returncode:raise RuntimeError('cost action failed: '+script)


def main():
    FOLDER.mkdir(exist_ok=True)
    while True:
        jobs=C.list_all_jobs();census=live_usage(jobs);planpath=FOLDER/'plan_attention.json'
        plan=json.loads(planpath.read_text()) if planpath.exists() else {}
        if not plan.get('submission'):
            if census['live_reserved_gpus']+8<=48:
                run('qz/submit_distance_cost.py','--submit');plan=json.loads(planpath.read_text())
            else:
                (FOLDER/'watcher.json').write_text(json.dumps(dict(status='waiting_for_released_capacity',capacity=census,at=C.timestamp()),indent=2)+'\n')
                time.sleep(30);continue
        job=next((j for j in jobs if j['job_id']==plan['submission']['job_id']),None)
        # A just-submitted handle first appears on the next read; never resubmit.
        if job and job['status'] in C.TERMINAL_STATUS and job['status']!='job_succeeded':raise ValueError('cost job failed: '+job['status'])
        complete=True
        for rank in range(8):
            f=Path(plan['out'])/f'rank{rank}.jsonl'
            try:last=json.loads(f.read_text().splitlines()[-1])
            except (FileNotFoundError,IndexError,json.JSONDecodeError):complete=False;break
            if last.get('kind')!='complete':complete=False;break
        if complete and not (FOLDER/'exact_attention.json').exists():
            run('theory_mvp/distance_intervention/collect_cost.py','--plan',planpath,'--out',FOLDER/'exact_attention.json')
        ready=all((ROOT/'results/distance_intervention'/f'exact_{a}_seed{s}.json').exists() for s in (20271011,20271012,20271013) for a in ('near','balanced'))
        if complete and ready:
            run('theory_mvp/distance_intervention/cost_certificate.py')
            (FOLDER/'watcher.json').write_text(json.dumps(dict(status='complete',at=C.timestamp()),indent=2)+'\n');return
        (FOLDER/'watcher.json').write_text(json.dumps(dict(status='cost_complete_waiting_for_terminal_quality' if complete else 'cost_running',job_id=plan['submission']['job_id'],scheduler_status=job['status'] if job else 'awaiting_listing',at=C.timestamp()),indent=2)+'\n')
        time.sleep(30)


if __name__=='__main__':main()
