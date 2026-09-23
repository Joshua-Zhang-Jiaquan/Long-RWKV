"""One bounded campaign: train six, source-calibrate, seal, then evaluate six.

No automatic retries. All admissions use the shared campaign lock and 32+16 cap.
"""
import fcntl,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'qz'))
import campaign as C
from confirm_capacity import live_usage
from lrwkv_evidence.predictive_transfer.training import SEEDS
FOLDER=ROOT/'results/predictive_transfer'


def log(**record):
    record['at']=C.timestamp()
    with (FOLDER/'controller.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)


def invoke(args):
    result=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True)
    log(command=args,returncode=result.returncode,output=result.stdout[-2500:],error=result.stderr[-2500:])
    if result.returncode:raise RuntimeError('campaign command failed; manual inspection required')


def complete(plan,split=None):
    state,body=C.job_status(plan['submission']['job_id'])
    if state in C.TERMINAL_STATUS and state!='job_succeeded':raise RuntimeError('failed attempt retained: '+str(plan['submission']))
    if state!='job_succeeded':return False
    out=Path(plan['out'])
    if split is None:
        done=json.loads((out/'completion.json').read_text())
        if not done.get('execution_complete'):raise ValueError('successful scheduler without completion')
    else:
        for rank in range(8):
            rows=(out/f'rank{rank}.jsonl').read_text().splitlines()
            if not rows or json.loads(rows[-1]).get('kind')!='complete':raise ValueError('incomplete successful evaluation')
    return True


def capacity():
    census=live_usage(C.list_all_jobs())
    return census['live_reserved_gpus']+8<=48 and any(census['project_reserved_gpus'][p]+8<=cap for p,cap in census['project_caps'].items())


def plan(path,args):
    if path.exists():
        value=json.loads(path.read_text())
        if value.get('submission'):
            if value['submission']['state']!='submitted':raise RuntimeError('submission needs manual inspection')
            return value
    if not capacity():return None
    invoke(args+['--submit'])
    value=json.loads(path.read_text())
    if value.get('submission',{}).get('state')!='submitted':raise RuntimeError('submission was not accepted')
    return value


def main():
    # The operator freezes the protocol only after inspecting qualification.
    if not (ROOT/'revision/transfer/FROZEN.json').exists():raise ValueError('protocol not frozen')
    while True:
        trained={};calibrated={};heldout={}
        for seed in SEEDS:
            training=plan(FOLDER/f'plan_seed{seed}.json',['qz/submit_predictive_transfer.py','--seed',str(seed)])
            if training is None:continue
            trained[seed]=complete(training)
            if not trained[seed]:continue
            cal=plan(FOLDER/f'plan_calibration_seed{seed}.json',['qz/submit_transfer_eval.py','--seed',str(seed),'--split','calibration'])
            if cal is not None:calibrated[seed]=complete(cal,'calibration')
        if len(calibrated)==6 and all(calibrated.values()):
            if not (FOLDER/'PREDICTION_SEAL.json').exists():invoke(['-m','revision.transfer.analyze','seal'])
            for seed in SEEDS:
                target=plan(FOLDER/f'plan_heldout_seed{seed}.json',['qz/submit_transfer_eval.py','--seed',str(seed),'--split','heldout'])
                if target is not None:heldout[seed]=complete(target,'heldout')
            if len(heldout)==6 and all(heldout.values()):
                invoke(['-m','revision.transfer.analyze','report']);log(status='complete');return
        log(status='running',trained=trained,calibrated=calibrated,heldout=heldout)
        time.sleep(60)

if __name__=='__main__':
    lock=(FOLDER/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as exc:
        log(status='stopped_for_manual_inspection',error=repr(exc),traceback=traceback.format_exc())
        raise
