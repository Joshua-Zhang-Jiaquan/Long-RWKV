"""Complete one declared adaptation and its exact development audit, without retries."""
import json,subprocess,sys,time
from pathlib import Path
import campaign as C
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from theory_mvp.posterior_context_probe.validate_training import validate
from theory_mvp.posterior_context_probe.collect import collect
FOLDER=ROOT/'results/long_context_mvp';DIGEST='dacbbe1355018e7f'

def run(args):
    p=subprocess.run([sys.executable,*args],cwd=ROOT,text=True,capture_output=True)
    print(p.stdout,flush=True)
    if p.returncode:raise RuntimeError((p.stdout+p.stderr)[-2000:])

def main():
    state=dict(status='waiting_for_qualification',started_unix=time.time());deadline=time.monotonic()+4*3600
    def save():
        state['updated_unix']=time.time();p=FOLDER/'posterior_adaptation_watcher.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
    validated=set()
    try:
        while time.monotonic()<deadline:
            for phase in ('qualification','development200'):
                path=FOLDER/f'plan_{DIGEST}_{phase}.json'
                if not path.exists():
                    run(['qz/submit_posterior_context_adapt.py','--qualification-plan',str(FOLDER/f'plan_{DIGEST}_qualification.json'),'--expected-source-digest',DIGEST,'--submit'])
                plan=json.loads(path.read_text());done=Path(plan['out'])/'completion.json';job=plan['submission']['job_id']
                if not done.exists():
                    status,_=C.job_status(job);state.update(status='waiting_for_'+phase,job_id=job,scheduler_status=status)
                    if status in ('job_failed','job_stopped','job_succeeded','job_canceled','job_cancelled'):raise RuntimeError('terminal job without completion')
                    break
                if phase not in validated:
                    result=validate(path);(FOLDER/f'posterior_adaptation_{phase}_validation.json').write_text(json.dumps(result,indent=2)+'\n');validated.add(phase)
            else:
                train=FOLDER/f'plan_{DIGEST}_development200.json'
                matches=[p for p in FOLDER.glob('plan_adapt_eval_*.json') if Path(json.loads(p.read_text())['training_plan']).resolve()==train.resolve() and json.loads(p.read_text()).get('submission')]
                if not matches:
                    run(['qz/submit_posterior_context_adapt_eval.py','--training-plan',str(train),'--submit']);state['status']='evaluation_submitted'
                elif len(matches)!=1:raise ValueError('multiple evaluations')
                else:
                    plan=json.loads(matches[0].read_text());out=Path(plan['out']);complete=True
                    for rank in range(8):
                        f=out/f'rank{rank}.jsonl'
                        try:complete=complete and f.exists() and json.loads(f.read_text().splitlines()[-1])['kind']=='complete'
                        except (IndexError,json.JSONDecodeError):complete=False
                    if complete:
                        dest=FOLDER/f'posterior_adaptation_{Path(plan["stage"]).name}.json'
                        run(['-m','theory_mvp.posterior_context_probe.collect','--plan',str(matches[0]),'--out',str(dest)])
                        result=json.loads(dest.read_text());state.update(status='completed',report=str(dest),short_gate_passed=result['short_gate_passed'],long_sweep_complete=result['long_sweep_complete']);save();return
                    status,_=C.job_status(plan['submission']['job_id']);state.update(status='evaluation_running',job_id=plan['submission']['job_id'],scheduler_status=status)
                    if status in ('job_failed','job_stopped','job_succeeded','job_canceled','job_cancelled'):raise RuntimeError('terminal evaluation without complete raw outputs')
            save();time.sleep(30)
        state['status']='observation_timeout_no_resubmission';save()
    except Exception as e:state.update(status='needs_attention',error=str(e));save();raise
if __name__=='__main__':main()
