"""Cost-only full requests; no answer-quality result is produced by this job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage,route_body,PRIMARY
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-plan',type=Path,required=True)
    ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    training=json.loads(args.training_plan.read_text());kind=training.get('kind','rwkv')
    if kind not in ('rwkv','attention','causal_rwkv'):raise ValueError('unknown model')
    done=json.loads((Path(training['out'])/'completion.json').read_text())
    allowed=(600,) if kind=='rwkv' else (20,600)
    if not done['execution_complete'] or done['step'] not in allowed:raise ValueError('requires complete qualified/trained checkpoint')
    checkpoint=Path(training['out'])/'resume.pt'
    if sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
    sources={rel:Path(training['stage'])/rel for rel in training['sources']}
    if any(sha(p)!=training['sources'][rel] for rel,p in sources.items()):raise ValueError('changed training source')
    for p in (ROOT/'lrwkv_evidence/long_context_eval').glob('*.py'):sources[str(p.relative_to(ROOT))]=p
    if kind!='rwkv':
        for p in (ROOT/'lrwkv_evidence/long_context_baseline_eval').glob('*.py'):sources[str(p.relative_to(ROOT))]=p
    launcher='qz/launch_long_context_eval.sh' if kind=='rwkv' else 'qz/launch_long_context_baseline_eval.sh'
    sources[launcher]=ROOT/launcher
    hashes={rel:sha(p) for rel,p in sources.items()}
    digest=hashlib.sha256(json.dumps(dict(sources=hashes,checkpoint=done['checkpoint_sha256'],cost_only=True),sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_long_context_cost_stages'/digest
    out=Path(E.OUTPUTS)/f'long_context_cost_{kind}_{digest}'
    env=dict(LCM_ROOT=str(stage),LCM_EVAL_OUT=str(out),LCM_CHECKPOINT=str(checkpoint),LCM_COST_ONLY='1',
             TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    if kind!='rwkv':env.update(LCM_KIND=kind,LCM_BASE=str(Path(E.G)/'models'/('pythia-410m' if kind=='attention' else 'rwkv7-0.4B')))
    body=E.make_body(f'lrwkv-lccost-{kind}-{digest[:8]}',E.wrapped(env,str(stage/launcher)),
                     'Full-request cost only; no quality claim; short/1K/4K/16K,10timing repeats;8H100 under48cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(cost_only=True,comparator=kind,stage=str(stage),out=str(out),training_plan=str(args.training_plan.resolve()),
                checkpoint_sha256=done['checkpoint_sha256'],checkpoint_step=done['step'],contract_sha256=done['contract_sha256'],sources=hashes,body=body)
    dest=ROOT/'results/long_context_mvp'/f'plan_cost_{kind}_{digest}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('already submitted')
    if args.submit:
        for p in dest.parent.glob(f'plan_cost_{kind}_*.json'):
            old=json.loads(p.read_text())
            if old.get('submission') and old['checkpoint_sha256']==done['checkpoint_sha256']:raise ValueError('already submitted cost study for this checkpoint')
        record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
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
