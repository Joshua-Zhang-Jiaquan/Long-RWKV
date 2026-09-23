"""Submit/collect independent cost screens within the existing shared GPU cap."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from confirm_capacity import live_usage,CAP
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-plan',type=Path,action='append',required=True)
    ap.add_argument('--hours',type=float,default=6);args=ap.parse_args()
    folder=ROOT/'results/long_context_mvp';state_path=folder/'cost_watcher.json'
    paths=[p.resolve() for p in args.training_plan]
    if len(set(paths))!=len(paths):raise ValueError('duplicate training plans')
    state=dict(status='running_cost_screens',started_unix=time.time(),items={})
    deadline=time.monotonic()+args.hours*3600
    def save():
        state['updated_unix']=time.time();temp=state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    try:
        while time.monotonic()<deadline:
            for training_path in paths:
                training=json.loads(training_path.read_text());kind=training.get('kind','rwkv')
                plans=[]
                for p in folder.glob(f'plan_cost_{kind}_*.json'):
                    r=json.loads(p.read_text())
                    if Path(r['training_plan']).resolve()==training_path and r.get('submission'):plans.append((p,r))
                if len(plans)>1:raise ValueError('duplicate submitted cost studies')
                if not plans:
                    capacity=live_usage(C.list_all_jobs())
                    if capacity['live_reserved_gpus']+8>CAP:continue
                    result=subprocess.run([sys.executable,str(ROOT/'qz/submit_long_context_cost.py'),
                                           '--training-plan',str(training_path),'--submit'],cwd=ROOT,text=True,capture_output=True)
                    if result.returncode and f'{CAP} GPU cap' not in result.stdout+result.stderr:
                        raise RuntimeError((result.stdout+result.stderr)[-2000:])
                    print(result.stdout,flush=True);continue
                path,plan=plans[0];key=str(training_path)
                if state['items'].get(key,{}).get('status')=='collected':continue
                complete=True
                for rank in range(8):
                    f=Path(plan['out'])/f'rank{rank}.jsonl';lines=f.read_text().splitlines() if f.exists() else []
                    try:complete=complete and bool(lines) and json.loads(lines[-1]).get('kind')=='complete'
                    except json.JSONDecodeError:complete=False
                if complete:
                    dest=folder/f'cost_{kind}_{Path(plan["stage"]).name}.json'
                    result=subprocess.run([sys.executable,'-m','lrwkv_evidence.long_context_eval.cost_collect',
                                           '--plan',str(path),'--out',str(dest)],cwd=ROOT,text=True,capture_output=True)
                    if result.returncode:raise RuntimeError((result.stdout+result.stderr)[-2000:])
                    state['items'][key]=dict(status='collected',report=str(dest));print(result.stdout,flush=True)
                else:
                    status,_=C.job_status(plan['submission']['job_id'])
                    state['items'][key]=dict(status=status,job_id=plan['submission']['job_id'])
                    if status in {'job_failed','job_stopped','job_cancelled','job_canceled','job_succeeded'}:
                        raise RuntimeError(kind+': '+status+' without complete cost shards')
            if len(state['items'])==len(paths) and all(v['status']=='collected' for v in state['items'].values()):
                state['status']='all_cost_screens_collected';save();return
            save();time.sleep(30)
        state['status']='watch_timeout_no_resubmission';save()
    except Exception as error:
        state.update(status='needs_attention',error=str(error));save();raise


if __name__=='__main__':main()
