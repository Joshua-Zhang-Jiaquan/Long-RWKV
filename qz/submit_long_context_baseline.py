"""Stage practical comparator development; shared project caps remain in force."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage,route_body
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('attention','causal_rwkv'),required=True);ap.add_argument('--qualification',action='store_true')
    ap.add_argument('--qualification-plan',type=Path);ap.add_argument('--expected-source-digest')
    ap.add_argument('--ieee',action='store_true');ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    if not args.qualification:
        if not args.qualification_plan:raise ValueError('qualification plan required')
        q=json.loads(args.qualification_plan.read_text())
        if q.get('kind')!=args.kind or q.get('phase')!='qualification':raise ValueError('wrong qualification')
        sources={rel:Path(q['stage'])/rel for rel in q['sources']}
        if any(sha(p)!=q['sources'][rel] for rel,p in sources.items()):raise ValueError('changed qualified source')
        # Qualified attention sources remain immutable when a failed causal
        # implementation or its precision is revised in the workspace.
        if args.ieee and not q.get('ieee',False):raise ValueError('qualification precision mismatch')
        args.ieee=q.get('ieee',False)
    else:
        original=json.loads((ROOT/'results/train04confirm/plan_8cd809d08a96ea13_confirm_parent_seed53.json').read_text())
        sources={rel:Path(original['stage'])/rel for rel in original['sources']}
        for rel,path in sources.items():
            if sha(path)!=original['sources'][rel]:raise ValueError('changed inherited implementation')
        for p in (ROOT/'lrwkv_evidence/long_context_mvp').glob('*.py'):sources[str(p.relative_to(ROOT))]=p
        for p in (ROOT/'lrwkv_evidence/long_context_baselines').glob('*.py'):sources[str(p.relative_to(ROOT))]=p
        sources['qz/launch_long_context_baseline.sh']=ROOT/'qz/launch_long_context_baseline.sh'
    hashes={rel:sha(p) for rel,p in sources.items()}
    identity=dict(kind=args.kind,sources=hashes)
    if args.ieee:identity['triton_f32_default']='ieee'
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    if args.expected_source_digest and args.expected_source_digest!=digest:raise ValueError('prepared source digest changed')
    prepared=ROOT/'results/long_context_mvp'/f'active_baseline_{args.kind}_digest.txt'
    if args.submit and prepared.exists() and prepared.read_text().strip()!=digest:raise ValueError('active prepared source digest changed')
    stage=Path(E.G)/'long_rwkv_long_context_baseline_stages'/digest
    phase='qualification' if args.qualification else 'development600'
    out=Path(E.OUTPUTS)/f'long_context_baseline_{args.kind}_{digest}_{phase}'
    env=dict(LCM_ROOT=str(stage),LCM_OUT=str(out),LCM_QUALIFICATION=str(int(args.qualification)),LCM_KIND=args.kind,LCM_BASE=str(Path(E.G)/'models'/('pythia-410m' if args.kind=='attention' else 'rwkv7-0.4B')))
    if args.ieee:env.update(TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    if not args.qualification:
        if not args.qualification_plan:raise ValueError('qualification plan required')
        q=json.loads(args.qualification_plan.read_text());done=json.loads((Path(q['out'])/'completion.json').read_text())
        if q.get('kind')!=args.kind or q['sources']!=hashes or not done.get('qualification') or not done.get('execution_complete') or done.get('step')!=20:raise ValueError('qualification mismatch')
        env['LCM_QUALIFICATION_OUT']=q['out']
    body=E.make_body(f'lrwkv-lcb-{args.kind}-{digest[:8]}-{phase}',E.wrapped(env,str(stage/'qz/launch_long_context_baseline.sh')),'Practical comparator development;8H100;shared32 plus extra16 budget',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(kind=args.kind,stage=str(stage),out=str(out),phase=phase,ieee=args.ieee,sources=hashes,body=body)
    dest=ROOT/'results/long_context_mvp'/f'plan_baseline_{args.kind}_{digest}_{phase}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('refuse duplicate submission')
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'])
        if C.already_submitted(body['name']):raise ValueError('already submitted')
        for rel,p in sources.items():
            target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if sha(p)!=hashes[rel]:raise ValueError('source changed while staging')
            if target.exists() and sha(target)!=hashes[rel]:raise ValueError('immutable stage conflict')
            if not target.exists():shutil.copyfile(p,target)
        (stage/'source_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
        record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))


if __name__=='__main__':
    with campaign_submission_lock(ROOT):main()
