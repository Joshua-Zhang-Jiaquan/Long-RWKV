"""Submit the frozen exhaustive-history diagnostic with the established capacity gate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage, route_body, PRIMARY
from distance_admission import run_when_capacity
from distance_shared import stage_execution
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',required=True);ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    folder=ROOT/'results/header_distance';prior=ROOT/'results/distance_intervention';manifestpath=ROOT/'theory_mvp/header_distance/FROZEN.json'
    manifest=json.loads(manifestpath.read_text())
    roles=manifest['roles']
    if args.role not in roles:raise ValueError('undeclared checkpoint selector')
    if sha(Path(manifest['checkpoints'][args.role]['path']))!=manifest['checkpoints'][args.role]['sha256']:raise ValueError('frozen checkpoint changed')
    for item in manifest['prior_outputs'].values():
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('archived result changed')
    q=json.loads((prior/'plan_qualification.json').read_text())
    trainpath=ROOT/'results/train04confirm/plan_8cd809d08a96ea13_confirm_independent_seed71.json' if args.role=='original' else prior/f'plan_{args.role}.json'
    train=json.loads(trainpath.read_text());done=json.loads((Path(train['out'])/'completion.json').read_text())
    checkpoint=Path(train['out'])/'resume.pt'
    if not done.get('execution_complete') or done.get('qualification') or sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('checkpoint incomplete or changed')
    if args.role=='original':
        if done['checkpoint_sha256']!=manifest['initial_checkpoint_sha256']:raise ValueError('wrong original')
    else:
        execution=json.loads((prior/'FROZEN_EXECUTION.json').read_text())
        if done['step']!=execution['terminal_steps'] or done['source_sha256']!=execution['source_sha256']:raise ValueError('not fixed qualified terminal model')
    sources={rel:Path(q['stage'])/rel for rel in q['sources']}
    if any(sha(p)!=q['sources'][rel] for rel,p in sources.items()):raise ValueError('qualified source changed')
    for rel,digest in manifest['sources'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('frozen evaluation source changed: '+rel)
        sources[rel]=ROOT/rel
    sources['header_distance_manifest.json']=manifestpath
    hashes={rel:sha(p) for rel,p in sources.items()}
    digest=hashlib.sha256(json.dumps(dict(sources=hashes,checkpoint=done['checkpoint_sha256']),sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_header_distance_stages'/digest
    out=Path(E.OUTPUTS)/f'header_distance_{digest}_{args.role}'
    execution_local=prior/'FROZEN_EXECUTION.json'
    execution_shared=stage_execution(execution_local,E.G)
    env=dict(DISTANCE_ROOT=str(stage),DISTANCE_OUT=str(out),DISTANCE_ROLE=args.role,
             DISTANCE_CHECKPOINT=str(checkpoint),DISTANCE_CHECKPOINT_SHA256=done['checkpoint_sha256'],
             DISTANCE_EXECUTION=str(execution_shared),
             TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    cache_seed=None
    cache_receipt=prior/'evaluation_cache_seed.json'
    if cache_receipt.exists():
        cache_seed=json.loads(cache_receipt.read_text())
        if sha(Path(cache_seed['seed_manifest']))!=cache_seed['seed_manifest_sha256']:raise ValueError('cache seed manifest changed')
        original=json.loads((ROOT/Path(cache_seed['source_evaluation_plan'])).read_text())
        if original['manifest_sha256']!=manifest['parent_evaluation_sha256']:raise ValueError('cache seed is from a different evaluation runtime')
        if not Path(cache_seed['cache']).is_dir():raise ValueError('shared compiled cache missing')
        env['TRITON_CACHE_DIR']=cache_seed['cache']
    body=E.make_body(f'lrwkv-header-{digest[:8]}-{args.role}',E.wrapped(env,str(stage/'qz/launch_header_distance.sh')),
                     'Fresh32problems x fixed header/task2x2 plus legacy controls; six fixed adaptations;8H100 under48cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='5400000')
    record=dict(stage=str(stage),out=str(out),sources=hashes,body=body,role=args.role,training_plan=str(trainpath),
                checkpoint=str(checkpoint),checkpoint_sha256=done['checkpoint_sha256'],manifest_sha256=sha(manifestpath))
    if cache_seed:record['compiled_cache_seed']=cache_seed
    dest=folder/f'plan_{args.role}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('duplicate evaluation')
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
        if C.already_submitted(body['name']):raise ValueError('duplicate scheduler name')
        for rel,p in sources.items():
            if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
            target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=hashes[rel]:raise ValueError('immutable stage conflict')
            if not target.exists():shutil.copyfile(p,target)
        (stage/'header_distance_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
        record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))


if __name__=='__main__':
    run_when_capacity(ROOT,main)
