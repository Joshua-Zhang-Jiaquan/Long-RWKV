"""Submit an isolated qualification or a manifest-frozen full-lineage replicate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage, CAP, route_body
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm import contract as K


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',action='store_true')
    parser.add_argument('--manifest',type=Path);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--phase',choices=['parent','independent','complementary'],required=True)
    parser.add_argument('--parent-plan',type=Path);parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    original=json.loads((ROOT/'results/train04paired/plan_0a82c9577c02d4e6_complementary_400.json').read_text())
    sources={rel:Path(original['stage'])/rel for rel in original['sources']}
    for rel,path in sources.items():
        if K.sha(path)!=original['sources'][rel]:raise ValueError('changed original development source')
    for path in (ROOT/'lrwkv_evidence/train04confirm').glob('*.py'):sources[str(path.relative_to(ROOT))]=path
    for rel in ('qz/launch_train04confirm.sh','qz/launch_train04confirm_eval.sh',
                'lrwkv_evidence/train04dev/evaluate.py','lrwkv_evidence/train04dev/history_response.py',
                'lrwkv_evidence/train04dev/collect_evaluation.py',
                'theory_mvp/train04confirm/PANEL.json','theory_mvp/train04confirm/PARTITIONS.json'):
        sources[rel]=ROOT/rel
    hashes={rel:K.sha(path) for rel,path in sources.items()}
    if args.qualification:
        if args.seed!=17 or args.phase!='parent' or args.manifest or args.parent_plan:raise ValueError('qualification is a separate seed17 parent probe')
        provenance=json.loads((Path(original['out'])/'provenance.json').read_text())
        manifest=dict(status='qualification_only',recipe=K.RECIPE,training=K.TRAINING,sources=hashes,base_sha256=provenance['contract']['base_sha256'])
    else:
        if not args.manifest or args.seed not in (53,71,89):raise ValueError('frozen manifest and fresh seed required')
        manifest=json.loads(args.manifest.read_text())
        if manifest.get('status')!='frozen' or manifest.get('recipe')!=K.RECIPE or manifest.get('training')!=K.TRAINING or manifest.get('sources')!=hashes:raise ValueError('frozen recipe differs from staged execution')
    payload=json.dumps(manifest,sort_keys=True,indent=2)+'\n';digest=hashlib.sha256(payload.encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_train04confirm_stages'/digest
    tag=f"{'qual' if args.qualification else 'confirm'}_{args.phase}_seed{args.seed}"
    out=Path(E.OUTPUTS)/f'train04confirm_{digest}_{tag}'
    env=dict(CONFIRM_ROOT=str(stage),CONFIRM_OUT=str(out),CONFIRM_SEED=str(args.seed),CONFIRM_PHASE=args.phase,CONFIRM_QUALIFICATION=str(int(args.qualification)))
    if args.phase!='parent':
        if not args.parent_plan:raise ValueError('branch needs a completed same-seed parent')
        parent=json.loads(args.parent_plan.read_text());folder=Path(parent['out']);done=json.loads((folder/'completion.json').read_text())
        if parent['seed']!=args.seed or parent['phase']!='parent' or parent['manifest_sha256']!=hashlib.sha256(payload.encode()).hexdigest() or done['step']!=1500 or done.get('qualification') or not done['execution_complete'] or K.sha(folder/'resume.pt')!=done['checkpoint_sha256']:raise ValueError('parent lineage mismatch')
        env['CONFIRM_PARENT']=str(folder/'resume.pt')
    elif args.parent_plan:raise ValueError('parent phase has no parent plan')
    body=E.make_body(f'lrwkv-confirm04-{digest[:8]}-{tag}',E.wrapped(env,str(stage/'qz/launch_train04confirm.sh')),'Fresh-seed full-lineage replication or discarded20update qualification;8H100 under48cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(stage=str(stage),out=str(out),seed=args.seed,phase=args.phase,qualification=args.qualification,
                manifest_sha256=hashlib.sha256(payload.encode()).hexdigest(),manifest=manifest,sources=hashes,body=body)
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs())
        record['budget_project_id']=route_body(body,record['capacity'])
        if C.already_submitted(body['name']):raise SystemExit('already submitted')
        for rel,path in sources.items():
            if K.sha(path)!=hashes[rel]:raise ValueError('source changed while staging')
            dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and K.sha(dest)!=hashes[rel]:raise ValueError('immutable stage conflict')
            if not dest.exists():shutil.copyfile(path,dest)
        dest=stage/'manifest.json'
        if dest.exists() and dest.read_text()!=payload:raise ValueError('immutable manifest conflict')
        dest.write_text(payload)
        K.validate(dest,stage,stage/'model_source',args.qualification)
        record['submission']=C.submit_one(body,dry_run=False)
    dest=ROOT/'results/train04confirm'/f'plan_{digest}_{tag}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('refuse to overwrite submission receipt')
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('manifest','sources','body')},indent=2))


if __name__=='__main__':
    from submission_lock import campaign_submission_lock
    with campaign_submission_lock(ROOT):main()
