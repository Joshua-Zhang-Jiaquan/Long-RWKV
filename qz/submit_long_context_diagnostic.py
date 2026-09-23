"""Audit the preserved short-training milestone; never select it as final model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage, route_body
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--capture',type=Path,default=ROOT/'results/long_context_mvp/milestone_capture.json')
    ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    capture=json.loads(args.capture.read_text())
    if capture['status']!='captured' or capture['step']!=200:
        raise ValueError('requires preserved development step200')
    training_path=Path(capture['training_plan']);training=json.loads(training_path.read_text())
    if training['phase']!='development600':raise ValueError('wrong training phase')
    checkpoint=Path(capture['checkpoint'])
    import torch
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if payload['step']!=200 or payload['contract']['recipe']['terminal_step']!=600 or payload['contract_sha256']!=capture['contract_sha256']:
        raise ValueError('captured checkpoint mismatch')
    del payload
    checkpoint_sha=sha(checkpoint)
    sources={rel:Path(training['stage'])/rel for rel in training['sources']}
    if any(sha(p)!=training['sources'][rel] for rel,p in sources.items()):raise ValueError('changed training source')
    for p in (ROOT/'lrwkv_evidence/long_context_eval').glob('*.py'):sources[str(p.relative_to(ROOT))]=p
    sources['qz/launch_long_context_eval.sh']=ROOT/'qz/launch_long_context_eval.sh'
    hashes={rel:sha(p) for rel,p in sources.items()}
    digest=hashlib.sha256(json.dumps(dict(sources=hashes,checkpoint=checkpoint_sha,diagnostic_step200=True),sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_long_context_eval_stages'/digest
    out=Path(E.OUTPUTS)/f'long_context_diagnostic200_{digest}'
    env=dict(LCM_ROOT=str(stage),LCM_EVAL_OUT=str(out),LCM_CHECKPOINT=str(checkpoint),LCM_DIAGNOSTIC_STEP200='1',
             TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    body=E.make_body(f'lrwkv-lcm-diag200-{digest[:8]}',E.wrapped(env,str(stage/'qz/launch_long_context_eval.sh')),
                     'Step200 short-task diagnostic; terminal600 unchanged;8H100 under shared48cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='3600000')
    record=dict(stage=str(stage),out=str(out),training_plan=str(training_path.resolve()),checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_sha,contract_sha256=capture['contract_sha256'],diagnostic_step200=True,sources=hashes,body=body)
    dest=ROOT/'results/long_context_mvp'/f'plan_diagnostic200_{digest}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('already submitted')
    if args.submit:
        # Refuse multiple diagnostic jobs even if analysis sources change.
        for p in dest.parent.glob('plan_diagnostic200_*.json'):
            old=json.loads(p.read_text())
            if old.get('submission') and old['checkpoint_sha256']==checkpoint_sha:
                raise ValueError('diagnostic already submitted for this checkpoint')
        record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'])
        if C.already_submitted(body['name']):raise ValueError('duplicate scheduler job')
        for rel,p in sources.items():
            if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
            target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=hashes[rel]:raise ValueError('stage conflict')
            if not target.exists():shutil.copyfile(p,target)
        (stage/'evaluation_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
        record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))


if __name__=='__main__':
    with campaign_submission_lock(ROOT):main()
