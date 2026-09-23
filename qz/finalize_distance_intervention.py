"""Collect finished fixed checkpoints and generate the complete analysis."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/distance_intervention'
ROLES=['original']+[f'{arm}_seed{seed}' for seed in (20271011,20271012,20271013) for arm in ('near','balanced')]


def run(script,*args):
    result=subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,text=True,capture_output=True)
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:raise RuntimeError('finalization command failed: '+script)


def complete(plan):
    out=Path(plan['out'])
    for rank in range(8):
        path=out/f'rank{rank}.jsonl'
        if not path.exists():return False
        try:last=json.loads(path.read_text().splitlines()[-1])
        except (IndexError,json.JSONDecodeError):return False
        if last.get('kind')!='complete':return False
    return True


def main():
    while True:
        finished=[]
        for role in ROLES:
            planpath=FOLDER/f'plan_eval_{role}.json'
            if not planpath.exists():continue
            plan=json.loads(planpath.read_text());dest=FOLDER/f'exact_{role}.json'
            if not dest.exists() and complete(plan):run('theory_mvp/distance_intervention/collect.py','--role',role)
            if dest.exists():
                result=json.loads(dest.read_text())
                if result.get('validated') and result['checkpoint_sha256']==plan['checkpoint_sha256'] and result['manifest_sha256']==plan['manifest_sha256']:
                    finished.append(role)
                else:raise ValueError('existing collected result does not match selector')
        state=dict(validated_roles=finished,all_seven_validated=len(finished)==7,updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        tmp=FOLDER/'finalization.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(FOLDER/'finalization.json')
        if len(finished)==7:
            run('theory_mvp/distance_intervention/report.py')
            run('theory_mvp/distance_intervention/figures.py')
            print('Complete exact panel, paired report and figure written. Paper integration still requires review.',flush=True)
            return
        time.sleep(30)


if __name__=='__main__':main()
