"""Publish the short gate promptly so comparators can overlap the long sweep."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-plan',type=Path,required=True);args=ap.parse_args()
    folder=ROOT/'results/long_context_mvp';state_path=folder/'transfer_short_gate_watcher.json'
    state=dict(status='waiting_for_evaluation',training_plan=str(args.training_plan.resolve()),started_unix=time.time())
    def save():
        state['updated_unix']=time.time();tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    try:
        deadline=time.monotonic()+6*3600
        while time.monotonic()<deadline:
            matches=[]
            for p in folder.glob('plan_eval_*.json'):
                r=json.loads(p.read_text())
                if Path(r['training_plan']).resolve()==args.training_plan.resolve() and r.get('submission'):matches.append((p,r))
            if len(matches)>1:raise ValueError('ambiguous submitted evaluation')
            if matches:
                path,plan=matches[0];ready=True
                for rank in range(8):
                    f=Path(plan['out'])/f'rank{rank}.jsonl';lines=f.read_bytes().splitlines() if f.exists() else []
                    try:ready=ready and len(lines)>=3 and json.loads(lines[2]).get('kind')=='gate'
                    except json.JSONDecodeError:ready=False
                if ready:
                    out=folder/f'gate_development_{Path(plan["stage"]).name}.json'
                    result=subprocess.run([sys.executable,'-m','lrwkv_evidence.long_context_eval.short_gate',
                                           '--plan',str(path),'--out',str(out)],cwd=ROOT,text=True,capture_output=True)
                    if result.returncode:raise RuntimeError((result.stdout+result.stderr)[-2000:])
                    state.update(status='validated_short_gate_complete',report=str(out));save();print(result.stdout,flush=True);return
            save();time.sleep(30)
        state['status']='watch_timeout';save()
    except Exception as error:
        state.update(status='needs_attention',error=str(error));save();raise


if __name__=='__main__':main()
