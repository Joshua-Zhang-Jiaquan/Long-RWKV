"""Submit once per frozen selector, validate terminal outputs, then analyze."""
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/header_distance'


def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,check=True)


def main():
    roles=[f'{a}_seed{s}' for s in (20271011,20271012,20271013) for a in ('near','balanced')]
    for role in roles:
        path=FOLDER/f'plan_{role}.json'
        if path.exists() and json.loads(path.read_text()).get('submission'):
            raise ValueError('controller is single-use; inspect existing submissions before any continuation')
    for role in roles:
        run('qz/submit_header_distance.py','--role',role,'--submit')
    while True:
        jobs={j['job_id']:j for j in C.list_all_jobs()}
        finished=[];status=[]
        for role in roles:
            plan=json.loads((FOLDER/f'plan_{role}.json').read_text())
            job=jobs[plan['submission']['job_id']]
            status.append(dict(role=role,job_id=job['job_id'],status=job['status']))
            if job['status'] in C.LIVE_STATUSES:
                continue
            complete=True
            for rank in range(8):
                path=Path(plan['out'])/f'rank{rank}.jsonl'
                if not path.exists():complete=False;break
                lines=path.read_text().splitlines()
                if not lines or json.loads(lines[-1]).get('kind')!='complete':complete=False;break
            if not complete:
                raise RuntimeError(f"{role} terminal without complete output: {job['status']}; no automatic retry")
            dest=FOLDER/f'exact_{role}.json'
            if not dest.exists():
                run('theory_mvp/header_distance/analyze.py','--role',role)
            else:
                result=json.loads(dest.read_text())
                if not result['validated'] or result['manifest_sha256']!=plan['manifest_sha256']:
                    raise ValueError('existing validation mismatch')
            finished.append(role)
        state=dict(validated=finished,jobs=status,updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (FOLDER/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
        if len(finished)==6:
            run('theory_mvp/header_distance/analyze.py')
            return
        time.sleep(30)


if __name__=='__main__':main()
