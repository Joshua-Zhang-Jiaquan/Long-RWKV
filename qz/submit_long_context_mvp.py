"""Stage the new development runner; respect shared32 plus added16 GPU budgets."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--qualification',action='store_true')
    ap.add_argument('--qualification-plan',type=Path);ap.add_argument('--expected-source-digest');ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    original=json.loads((ROOT/'results/train04confirm/plan_8cd809d08a96ea13_confirm_parent_seed53.json').read_text())
    sources={rel:Path(original['stage'])/rel for rel in original['sources']}
    for rel,path in sources.items():
        if sha(path)!=original['sources'][rel]:raise ValueError('changed inherited implementation')
    for p in (ROOT/'lrwkv_evidence/long_context_mvp').glob('*.py'):sources[str(p.relative_to(ROOT))]=p
    sources['qz/launch_long_context_mvp.sh']=ROOT/'qz/launch_long_context_mvp.sh'
    hashes={rel:sha(p) for rel,p in sources.items()};digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    if args.expected_source_digest and args.expected_source_digest!=digest:raise ValueError('prepared source digest changed')
    prepared=ROOT/'results/long_context_mvp/active_source_digest.txt'
    if args.submit and prepared.exists() and prepared.read_text().strip()!=digest:raise ValueError('active prepared source digest changed')
    stage=Path(E.G)/'long_rwkv_long_context_mvp_stages'/digest
    phase='qualification' if args.qualification else 'development600'
    out=Path(E.OUTPUTS)/f'long_context_mvp_{digest}_{phase}'
    env=dict(LCM_ROOT=str(stage),LCM_OUT=str(out),LCM_QUALIFICATION=str(int(args.qualification)))
    if not args.qualification:
        if not args.qualification_plan:raise ValueError('qualification plan required')
        q=json.loads(args.qualification_plan.read_text());done=json.loads((Path(q['out'])/'completion.json').read_text())
        if q['sources']!=hashes or not done.get('qualification') or not done.get('execution_complete') or done.get('step')!=20:raise ValueError('qualification mismatch')
        env['LCM_QUALIFICATION_OUT']=q['out']
    body=E.make_body(f'lrwkv-lcm-{digest[:8]}-{phase}',E.wrapped(env,str(stage/'qz/launch_long_context_mvp.sh')),'New long-context task development;8H100;shared32 plus extra16 budget',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(stage=str(stage),out=str(out),phase=phase,sources=hashes,body=body)
    dest=ROOT/'results/long_context_mvp'/f'plan_{digest}_{phase}.json';dest.parent.mkdir(parents=True,exist_ok=True)
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
