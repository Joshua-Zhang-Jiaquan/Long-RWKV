"""Submit exact development gate and conditional long-context expansion."""
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
    training=json.loads(args.training_plan.read_text());out_train=Path(training['out'])
    done=json.loads((out_train/'completion.json').read_text())
    if training['phase']!='development200' or done.get('qualification') or not done.get('execution_complete') or done.get('step')!=200:
        raise ValueError('requires completed development600')
    checkpoint=out_train/'resume.pt'
    if sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
    sources={rel:Path(training['stage'])/rel for rel in training['sources']}
    if any(sha(p)!=training['sources'][rel] for rel,p in sources.items()):raise ValueError('changed training source')
    sources['lrwkv_evidence/posterior_context_adapt/evaluate.py']=ROOT/'lrwkv_evidence/posterior_context_adapt/evaluate.py'
    sources['qz/launch_posterior_context_adapt_eval.sh']=ROOT/'qz/launch_posterior_context_adapt_eval.sh'
    hashes={rel:sha(p) for rel,p in sources.items()}
    digest=hashlib.sha256(json.dumps(dict(sources=hashes,checkpoint=done['checkpoint_sha256']),sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_posterior_context_adapt_eval_stages'/digest;out=Path(E.OUTPUTS)/f'posterior_context_adapt_eval_{digest}'
    env=dict(PROBE_ROOT=str(stage),PROBE_OUT=str(out),PROBE_CHECKPOINT=str(checkpoint),PROBE_SHA256=done['checkpoint_sha256'],TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    body=E.make_body(f'lrwkv-pca-eval-{digest[:8]}',E.wrapped(env,str(stage/'qz/launch_posterior_context_adapt_eval.sh')),'Fixed200 label-preserving adaptation; exact dev endpoint probe;8H100',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(stage=str(stage),out=str(out),training_plan=str(args.training_plan.resolve()),checkpoint_sha256=done['checkpoint_sha256'],sources=hashes,body=body)
    dest=ROOT/'results/long_context_mvp'/f'plan_adapt_eval_{digest}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('already submitted')
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
        if C.already_submitted(body['name']):raise ValueError('duplicate scheduler job')
        for rel,p in sources.items():
            if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
            target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=hashes[rel]:raise ValueError('stage conflict')
            if not target.exists():shutil.copyfile(p,target)
        (stage/'adapt_eval_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
        record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))


if __name__=='__main__':
    with campaign_submission_lock(ROOT):main()
