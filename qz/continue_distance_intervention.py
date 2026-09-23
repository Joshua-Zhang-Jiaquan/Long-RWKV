"""Advance the fixed study using artifacts plus live scheduler state; no retries."""
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from confirm_capacity import live_usage
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha

FOLDER=ROOT/'results/distance_intervention'
ROLES=[f'{arm}_seed{seed}' for seed in (20271011,20271012,20271013) for arm in ('near','balanced')]


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def terminal_training(plan):
    out=Path(plan['out']);done=out/'completion.json'
    if not done.exists():return None
    result=json.loads(done.read_text())
    if not result.get('execution_complete') or result['step']!=plan['steps'] or result['source_sha256']!=plan['source_sha256']:
        raise ValueError('training completion contract mismatch')
    rows=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows]!=list(range(1,plan['steps']+1)):raise ValueError('noncontiguous training')
    for r in rows:
        if not all(math.isfinite(r[k]) for k in ('loss','oracle_entropy','excess_ce','grad_norm_rank0','max_rank_step_seconds')):
            raise ValueError('nonfinite training')
        if r['global_input_tokens']<32*16370:raise ValueError('not all16K training')
    for rank in range(8):
        q=json.loads((out/f'qualification_rank{rank}.json').read_text())
        if q['source_sha256']!=plan['source_sha256'] or [r['position'] for r in q['checks']]!=['far','middle','near']:
            raise ValueError('missing isolation checks')
        if any(r['serial_difference']>1e-5 or r['other_document_difference']>1e-5 or r['own_document_difference']<1e-6 for r in q['checks']):
            raise ValueError('failed isolation')
    if sha(out/'resume.pt')!=result['checkpoint_sha256']:raise ValueError('checkpoint checksum mismatch')
    return dict(completion=result,steps=len(rows),input_tokens=sum(r['global_input_tokens'] for r in rows),
                training_gpu_hours=8*rows[-1]['elapsed_seconds']/3600,
                max_peak_allocated_bytes=max(r['max_rank_peak_allocated_bytes'] for r in rows),
                last_three_update_seconds=[r['max_rank_step_seconds'] for r in rows[-3:]])


def invoke(script,*args):
    result=subprocess.run([sys.executable,str(ROOT/'qz'/script),*map(str,args)],cwd=ROOT,text=True,capture_output=True)
    with (FOLDER/'controller_actions.log').open('a') as f:f.write(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(script+' failed: '+(result.stdout+result.stderr)[-3000:])


def main():
    last_message=None
    while True:
        jobs=C.list_all_jobs();byid={j['job_id']:j for j in jobs};census=live_usage(jobs)
        plans={p.stem.removeprefix('plan_'):json.loads(p.read_text()) for p in FOLDER.glob('plan_*.json')}
        states={}
        for key,plan in plans.items():
            if not plan.get('submission'):continue
            job_id=plan['submission']['job_id']
            job=byid.get(job_id)
            if job is None:raise ValueError('submitted handle missing from authoritative listing: '+job_id)
            status=job['status'];states[key]=dict(job_id=job_id,status=status)
            if status in C.TERMINAL_STATUS and status!='job_succeeded':
                raise ValueError('study job failed; inspect before any repair: '+key+' '+status)
        q=plans['qualification'];validated_path=FOLDER/'qualification_validation.json'
        if not validated_path.exists():
            validated=terminal_training(q)
            if validated:atomic(validated_path,validated)
        if validated_path.exists():
            execution_path=FOLDER/'FROZEN_EXECUTION.json'
            if not execution_path.exists():
                validation=json.loads(validated_path.read_text())
                seconds=max(validation['last_three_update_seconds'])
                steps=min(600,math.floor(36000/seconds/12)*12)
                if steps<24:raise ValueError('one-day runtime gate failed: fewer than24 updates')
                execution=dict(terminal_steps=steps,source_sha256=q['source_sha256'],
                               qualification_plan=str(FOLDER/'plan_qualification.json'),
                               qualification_validation_sha256=sha(validated_path),profile_max_step_seconds=seconds,
                               predicted_steady_training_hours=steps*seconds/3600,
                               selectors=ROLES,selection_basis='runtime only; fixed before main training/evaluation')
                with execution_path.open('x') as f:json.dump(execution,f,indent=2);f.write('\n')
            for role in ROLES:
                if role not in plans:
                    if census['live_reserved_gpus']+8>48:break
                    arm,seed=role.split('_seed')
                    invoke('submit_distance_intervention.py','--arm',arm,'--seed',seed,
                           '--qualification-plan',FOLDER/'plan_qualification.json','--submit')
                    jobs=C.list_all_jobs();census=live_usage(jobs)
        # Admit evaluations as capacity opens, without delaying available training slots.
        plans={p.stem.removeprefix('plan_'):json.loads(p.read_text()) for p in FOLDER.glob('plan_*.json')}
        for role in ROLES:
            if role not in plans:continue
            path=FOLDER/f'validation_{role}.json'
            if not path.exists():
                result=terminal_training(plans[role])
                if result:atomic(path,result)
            if path.exists() and 'eval_'+role not in plans:
                jobs=C.list_all_jobs();census=live_usage(jobs)
                if census['live_reserved_gpus']+8<=48:invoke('submit_distance_evaluation.py','--role',role,'--submit')
        eval_complete=[]
        for role in ['original']+ROLES:
            planpath=FOLDER/f'plan_eval_{role}.json'
            if not planpath.exists():continue
            plan=json.loads(planpath.read_text());out=Path(plan['out'])
            complete=True
            for rank in range(8):
                f=out/f'rank{rank}.jsonl'
                if not f.exists():complete=False;break
                lines=f.read_text().splitlines()
                try:last=json.loads(lines[-1])
                except (IndexError,json.JSONDecodeError):complete=False;break
                if last.get('kind')!='complete':complete=False;break
            if complete:eval_complete.append(role)
        state=dict(updated_at=C.timestamp(),status='all_evaluations_written' if len(eval_complete)==7 else 'running',
                   capacity=census,jobs=states,evaluation_outputs_complete=eval_complete)
        atomic(FOLDER/'watcher.json',state)
        message=json.dumps(dict(status=state['status'],reserved=census['live_reserved_gpus'],jobs=states,completed=eval_complete),sort_keys=True)
        if message!=last_message:print(message,flush=True);last_message=message
        if len(eval_complete)==7:return
        time.sleep(30)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic(FOLDER/'watcher_error.json',dict(at=C.timestamp(),error=repr(error)))
        raise
