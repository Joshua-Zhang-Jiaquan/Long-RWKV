"""Collect an already submitted milestone audit; never submit or resubmit jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    if not plan.get('diagnostic_step200') or not plan.get('submission'):raise ValueError('requires submitted diagnostic')
    destination=ROOT/'results/long_context_mvp'/f'exact_diagnostic200_{Path(plan["stage"]).name}.json'
    state_path=destination.parent/'diagnostic_watcher.json'
    state=dict(status='monitoring_submitted_diagnostic',job_id=plan['submission']['job_id'],started_unix=time.time())
    def save():
        state['updated_unix']=time.time();temp=state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    try:
        deadline=time.monotonic()+7200
        while time.monotonic()<deadline:
            complete=True
            for rank in range(8):
                p=Path(plan['out'])/f'rank{rank}.jsonl'
                lines=p.read_text().splitlines() if p.exists() else []
                try:complete=complete and bool(lines) and json.loads(lines[-1]).get('kind')=='complete'
                except json.JSONDecodeError:complete=False
            if complete:
                state['status']='validating';save()
                result=subprocess.run([sys.executable,'-m','lrwkv_evidence.long_context_eval.collect',
                                       '--plan',str(args.plan),'--out',str(destination)],cwd=ROOT,text=True,capture_output=True)
                if result.returncode:raise RuntimeError((result.stdout+result.stderr)[-2000:])
                state.update(status='diagnostic_collected',result=str(destination));save()
                print(result.stdout,flush=True);return
            status,_=C.job_status(state['job_id']);state['scheduler_status']=status
            if status in {'job_failed','job_stopped','job_cancelled','job_canceled','job_succeeded'}:
                raise RuntimeError(status+' without all eight completed shards')
            save();time.sleep(30)
        state['status']='watch_timeout_no_resubmission';save()
    except Exception as error:
        state.update(status='needs_attention',error=str(error));save();raise


if __name__=='__main__':main()
