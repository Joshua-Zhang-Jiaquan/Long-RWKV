"""Collect completed fixed phase700 audits, then continue both declared N8 arms.

No outcome-based selection: both treatments retain their original optimizer and
protocol. Submission commands enforce the shared lock and 32-H100 cap.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from submit_mvp_lookup import live_usage

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'results/train04mix/n8_continuation_watcher.json'
ARMS=('hard','rao_blackwell')

def main():
    state={'status':'waiting','arms':{},'started_unix':time.time()}
    for _ in range(180):
        for arm in ARMS:
            # Existing receipt, not a previous watcher's state, prevents duplication.
            receipts=[]
            for path in (ROOT/'results/train04mix').glob(f'plan_*_{arm}_1500.json'):
                receipt=json.loads(path.read_text())
                if receipt.get('submission',{}).get('job_id'):receipts.append(receipt)
            if receipts:
                state['arms'][arm]={'status':'submitted','job_id':receipts[0]['submission']['job_id']};continue
            training=ROOT/f'results/train04mix/plan_25ce57244ae9a384_{arm}_700.json'
            plan=json.loads(training.read_text())
            audits=[]
            for path in (ROOT/'results/train04dev').glob('plan_eval_*.json'):
                receipt=json.loads(path.read_text())
                if receipt['training_plan']==str(training) and receipt.get('submission',{}).get('job_id'):audits.append(receipt)
            if len(audits)!=1:raise ValueError(f'expected exactly one phase700 audit for {arm}')
            audit=audits[0];status,_=C.job_status(audit['submission']['job_id'])
            if status!='job_succeeded':
                state['arms'][arm]={'status':'waiting_audit','scheduler_status':status}
                if status in ('job_failed','job_stopped','job_cancelled','job_canceled'):raise RuntimeError(f'audit terminal without success: {arm} {status}')
                continue
            output=ROOT/f'results/train04mix/phase700/exact_{arm}.json';output.parent.mkdir(parents=True,exist_ok=True)
            run=subprocess.run([sys.executable,'-m','lrwkv_evidence.train04dev.collect_evaluation','--root',audit['out'],'--out',str(output)],cwd=ROOT,capture_output=True,text=True)
            if run.returncode:raise RuntimeError(run.stderr[-2000:])
            capacity=live_usage(C.list_all_jobs())
            if capacity['live_reserved_gpus']+8>32:
                state['arms'][arm]={'status':'waiting_capacity','reserved':capacity['live_reserved_gpus']};continue
            run=subprocess.run([sys.executable,str(ROOT/'qz/submit_train04mix.py'),'--objective',arm,'--stop-step','1500','--resume',str(Path(plan['out'])/'resume.pt'),'--submit'],cwd=ROOT,capture_output=True,text=True)
            state['arms'][arm]={'status':'submission_returned','returncode':run.returncode,'tail':(run.stdout+run.stderr)[-1400:]}
            if run.returncode:raise RuntimeError(state['arms'][arm]['tail'])
        state['updated_unix']=time.time()
        if all(state['arms'].get(a,{}).get('status')=='submitted' for a in ARMS):state['status']='finished'
        temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)
        if state['status']=='finished':return
        time.sleep(20)
    state['status']='timeout_no_resubmission';STATE.write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':main()
