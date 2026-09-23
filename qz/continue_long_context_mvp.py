"""Wait for authorized capacity, qualify once, then launch the fixed development run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from confirm_capacity import live_usage,CAP
ROOT=Path(__file__).resolve().parents[1]
TERMINAL={'job_failed','job_stopped','job_cancelled','job_canceled','job_succeeded'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qualification-plan',type=Path,required=True)
    ap.add_argument('--hours',type=float,default=6)
    ap.add_argument('--submit-script',choices=('submit_long_context_mvp.py','submit_long_context_transfer.py'),default='submit_long_context_mvp.py')
    ap.add_argument('--state-name',default='watcher.json')
    ap.add_argument('--require-failed-report-for',type=Path);args=ap.parse_args()
    initial=json.loads(args.qualification_plan.read_text());digest=Path(initial['stage']).name
    if Path(args.state_name).name!=args.state_name:raise ValueError('state name must be a basename')
    state_path=ROOT/'results/long_context_mvp'/args.state_name;deadline=time.monotonic()+args.hours*3600
    state=dict(status='waiting_for_capacity',qualification_plan=str(args.qualification_plan),started_unix=time.time())
    def save():
        state['updated_unix']=time.time();tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    try:
        while time.monotonic()<deadline:
            for phase in ('qualification','development600'):
                if phase=='development600' and args.require_failed_report_for:
                    required=json.loads(args.require_failed_report_for.read_text())
                    completion=Path(required['out'])/'completion.json'
                    if not completion.exists():break
                    checkpoint=json.loads(completion.read_text())['checkpoint_sha256']
                    reports=[json.loads(p.read_text()) for p in (ROOT/'results/long_context_mvp').glob('exact_development_*.json')]
                    reports=[r for r in reports if r['checkpoint_sha256']==checkpoint and r.get('comparator','rwkv')=='rwkv' and not r.get('diagnostic_step200')]
                    if not reports:break
                    if len(reports)!=1 or not reports[0]['execution_complete']:raise ValueError('ambiguous/incomplete prerequisite report')
                    if reports[0]['competence_gate_passed']:
                        state['status']='followup_not_needed_original_gate_passed';save();return
                    state['prerequisite_failed_checkpoint_sha256']=checkpoint
                path=ROOT/'results/long_context_mvp'/f'plan_{digest}_{phase}.json'
                record=json.loads(path.read_text()) if path.exists() else None
                if record and record['sources']!=initial['sources']:raise ValueError('source set changed')
                if not record or not record.get('submission'):
                    capacity=live_usage(C.list_all_jobs());state['capacity']=capacity
                    if capacity['live_reserved_gpus']+8>CAP:break
                    command=[sys.executable,str(ROOT/'qz'/args.submit_script),'--expected-source-digest',digest,'--submit']
                    command+=['--qualification'] if phase=='qualification' else ['--qualification-plan',str(args.qualification_plan)]
                    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
                    if result.returncode:
                        if f'{CAP} GPU cap' in result.stdout+result.stderr:break
                        raise RuntimeError((result.stdout+result.stderr)[-1500:])
                    print(result.stdout,flush=True);record=json.loads(path.read_text())
                    if record['sources']!=initial['sources']:raise ValueError('submitted source set changed')
                job=record['submission']['job_id'];status,_=C.job_status(job)
                state[phase]=dict(job_id=job,scheduler_status=status)
                done=Path(record['out'])/'completion.json'
                if done.exists():
                    if phase=='development600':state['status']='development_training_complete_evaluation_pending';save();return
                    continue
                if status in TERMINAL:raise RuntimeError(f'{phase}: {status} without complete evidence')
                break
            state['status']='monitoring';save();time.sleep(30)
        state['status']='watch_timeout_no_resubmission';save()
    except Exception as error:
        state.update(status='needs_attention',error=str(error));save();raise


if __name__=='__main__':main()
