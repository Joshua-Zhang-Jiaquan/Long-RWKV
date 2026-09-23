"""Same600-update task adaptation, initialized by the learned posterior model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage,route_body,PRIMARY,PROJECT_CAPS,CAP
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qualification',action='store_true')
    ap.add_argument('--qualification-plan',type=Path);ap.add_argument('--expected-source-digest')
    ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    folder=ROOT/'results/long_context_mvp'
    original=json.loads((folder/'plan_posterior_probe_195495c5ae42fe95.json').read_text())
    init_plan=ROOT/'results/train04confirm/plan_8cd809d08a96ea13_confirm_independent_seed71.json'
    initial=json.loads(init_plan.read_text());initial_out=Path(initial['out'])
    done=json.loads((initial_out/'completion.json').read_text());checkpoint=initial_out/'resume.pt'
    if done['step']!=2500 or done['seed']!=71 or done['phase']!='independent' or not done['execution_complete']:
        raise ValueError('wrong initial model')
    if done['checkpoint_sha256']!='52dbc7f289513e475d10422dbccffa52f74ec55fc887253ea13ae0476dc3e2c9' or sha(checkpoint)!=done['checkpoint_sha256']:
        raise ValueError('auxiliary model checksum mismatch')
    initialization=dict(training_plan=str(init_plan),checkpoint=str(checkpoint),checkpoint_sha256=done['checkpoint_sha256'],
                        selection='Development initialization choice after seed71 learned the prior posterior; not a new-task final result',
                        accounting='Additional2500 auxiliary updates from the released initialization; excluded from neither training history nor limitations')
    if args.qualification:
        sources={rel:Path(original['stage'])/rel for rel in original['sources']}
        if any(sha(p)!=original['sources'][rel] for rel,p in sources.items()):raise ValueError('changed original task recipe')
        for p in (ROOT/'lrwkv_evidence/posterior_context_adapt').glob('*.py'):sources[str(p.relative_to(ROOT))]=p
        sources['qz/launch_posterior_context_adapt.sh']=ROOT/'qz/launch_posterior_context_adapt.sh'
    else:
        if not args.qualification_plan:raise ValueError('qualification plan required')
        q=json.loads(args.qualification_plan.read_text())
        if q.get('initialization')!=initialization:raise ValueError('qualification initialization mismatch')
        sources={rel:Path(q['stage'])/rel for rel in q['sources']}
        if any(sha(p)!=q['sources'][rel] for rel,p in sources.items()):raise ValueError('changed qualified source')
    hashes={rel:sha(p) for rel,p in sources.items()}
    digest=hashlib.sha256(json.dumps(dict(sources=hashes,initial_checkpoint=done['checkpoint_sha256']),sort_keys=True).encode()).hexdigest()[:16]
    if args.expected_source_digest and args.expected_source_digest!=digest:raise ValueError('prepared source digest changed')
    stage=Path(E.G)/'long_rwkv_posterior_context_adapt_stages'/digest
    phase='qualification' if args.qualification else 'development200'
    out=Path(E.OUTPUTS)/f'posterior_context_adapt_{digest}_{phase}'
    env=dict(LCM_ROOT=str(stage),LCM_OUT=str(out),LCM_QUALIFICATION=str(int(args.qualification)),
             TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'),LCT_INITIAL_CHECKPOINT=str(checkpoint),LCT_INITIAL_SHA256=done['checkpoint_sha256'])
    if not args.qualification:
        qualified=json.loads((Path(q['out'])/'completion.json').read_text())
        if not qualified.get('qualification') or not qualified.get('execution_complete') or qualified.get('step')!=20:
            raise ValueError('qualification incomplete')
        env['LCM_QUALIFICATION_OUT']=q['out']
    body=E.make_body(f'lrwkv-pca-{digest[:8]}-{phase}',E.wrapped(env,str(stage/'qz/launch_posterior_context_adapt.sh')),
                     'Controlled auxiliary-conditioner warm start; label-preserving200 context adaptation;8H100 under48cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(stage=str(stage),out=str(out),phase=phase,sources=hashes,body=body,initialization=initialization)
    dest=folder/f'plan_{digest}_{phase}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('refuse duplicate submission')
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs())
        if record['capacity']['project_reserved_gpus'][PRIMARY]+8>PROJECT_CAPS[PRIMARY]:
            raise SystemExit(f'{CAP} GPU cap: wait for primary placement after observed extra-project queue delay')
        record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
        if C.already_submitted(body['name']):raise ValueError('already submitted')
        for rel,p in sources.items():
            if sha(p)!=hashes[rel]:raise ValueError('source changed while staging')
            target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=hashes[rel]:raise ValueError('immutable stage conflict')
            if not target.exists():shutil.copyfile(p,target)
        (stage/'source_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
        record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))


if __name__=='__main__':
    with campaign_submission_lock(ROOT):main()
