"""Bounded orchestration of the frozen three-seed experiment; no outcome selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from confirm_capacity import live_usage, CAP

ROOT=Path(__file__).resolve().parents[1]
TERMINAL={'job_failed','job_stopped','job_cancelled','job_canceled','job_succeeded'}


def load(path):return json.loads(path.read_text())


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def endpoint_complete(out):
    for rank in range(8):
        path=out/f'rank{rank}.jsonl'
        if not path.exists():return False
        lines=path.read_text().splitlines()
        if not lines:return False
        try:last=json.loads(lines[-1])
        except json.JSONDecodeError:return False
        if last.get('kind')!='complete':return False
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True);parser.add_argument('--hours',type=float,default=6)
    args=parser.parse_args();manifest_path=args.manifest.resolve();manifest=load(manifest_path)
    if manifest.get('status')!='frozen' or manifest['training']['data_seeds']!=[53,71,89]:raise ValueError('requires frozen three-seed manifest')
    digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest();short=digest[:16]
    folder=ROOT/'results/train04confirm';args.state.parent.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',started_unix=time.time(),manifest_sha256=digest,items={})
    deadline=time.monotonic()+args.hours*3600

    def plan_path(seed,phase):return folder/f'plan_{short}_confirm_{phase}_seed{seed}.json'

    def status_or_ready(path,completed):
        if not path.exists():return False
        plan=load(path)
        if plan.get('manifest_sha256')!=digest:raise ValueError('receipt manifest mismatch')
        job=plan.get('submission',{}).get('job_id')
        if not job:return False
        if completed(Path(plan['out'])):
            state['items'][path.name]=dict(job_id=job,evidence_status='complete')
            return True
        status,_=C.job_status(job);state['items'][path.name]=dict(job_id=job,scheduler_status=status)
        if status in TERMINAL:raise RuntimeError(f'{job}: {status} without complete evidence')
        return False

    def submit(command):
        capacity=live_usage(C.list_all_jobs())
        if capacity['live_reserved_gpus']+8>CAP:return False
        # Submission scripts take the shared lock and recheck capacity/receipts.
        result=subprocess.run([sys.executable,*command],cwd=ROOT,capture_output=True,text=True)
        if result.returncode:
            if f'{CAP} GPU cap' in result.stdout+result.stderr:return False
            raise RuntimeError((result.stdout+result.stderr)[-2000:])
        print(result.stdout,flush=True);return True

    try:
        while time.monotonic()<deadline:
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=digest:raise ValueError('frozen manifest changed')
            for seed in (53,71,89):
                parent=plan_path(seed,'parent')
                ready=status_or_ready(parent,lambda out:(out/'completion.json').exists())
                if not ready:continue
                for phase in ('independent','complementary'):
                    path=plan_path(seed,phase)
                    if not path.exists() or not load(path).get('submission',{}).get('job_id'):
                        submit([str(ROOT/'qz/submit_train04confirm.py'),'--manifest',str(manifest_path),'--seed',str(seed),'--phase',phase,'--parent-plan',str(parent),'--submit'])
                    if not status_or_ready(path,lambda out:(out/'completion.json').exists()):continue
                    audit=folder/f'plan_eval_{short}_{phase}_seed{seed}.json'
                    if not audit.exists() or not load(audit).get('submission',{}).get('job_id'):
                        submit([str(ROOT/'qz/submit_train04confirm_eval.py'),'--training-plan',str(path),'--submit'])
                    if not status_or_ready(audit,endpoint_complete):continue
                    record=load(audit);dest=folder/f'exact_{short}_{phase}_seed{seed}.json'
                    if not dest.exists():
                        result=subprocess.run([sys.executable,'-m','theory_mvp.train04confirm.collect','--root',record['out'],'--stage',record['stage'],'--out',str(dest)],cwd=ROOT,capture_output=True,text=True)
                        if result.returncode:raise RuntimeError('collection failed: '+(result.stdout+result.stderr)[-2000:])
                    state['items'][f'{phase}/seed{seed}']=dict(status='collected',result=str(dest))
            state['updated_unix']=time.time()
            if all(state['items'].get(f'{phase}/seed{seed}',{}).get('status')=='collected' for seed in (53,71,89) for phase in ('independent','complementary')):
                state['status']='complete_six_checkpoint_audits';save(args.state,state);return
            save(args.state,state);time.sleep(30)
        state['status']='watch_timeout_no_resubmission';save(args.state,state)
    except Exception as error:
        state.update(status='needs_attention',error=str(error),updated_unix=time.time());save(args.state,state);raise


if __name__=='__main__':main()
