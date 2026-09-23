"""Stage a distinct development-only probe, preserving frozen training sources."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage,route_body,PRIMARY
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm import contract as K

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    trainplan=ROOT/'results/train04confirm/plan_8cd809d08a96ea13_confirm_independent_seed71.json'
    training=json.loads(trainplan.read_text());old=Path(training['stage']);train=Path(training['out'])
    manifest,_=K.validate(old/'manifest.json',old,old/'model_source')
    done=json.loads((train/'completion.json').read_text());checkpoint=train/'resume.pt'
    if not done.get('execution_complete') or done['step']!=2500 or K.sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('invalid checkpoint')
    sources={rel:old/rel for rel in manifest['sources']};sources['manifest.json']=old/'manifest.json'
    for name in ('PANEL.json','PARTITIONS.json'):sources['theory_mvp/train04confirm/'+name]=old/'theory_mvp/train04confirm'/name
    for p in (ROOT/'lrwkv_evidence/posterior_context_probe').glob('*.py'):sources[str(p.relative_to(ROOT))]=p
    for name in ('__init__.py','runtime.py','numerical.py'):sources['lrwkv_evidence/long_context_eval/'+name]=ROOT/'lrwkv_evidence/long_context_eval'/name
    sources['qz/launch_posterior_context_probe.sh']=ROOT/'qz/launch_posterior_context_probe.sh'
    hashes={rel:K.sha(p) for rel,p in sources.items()}
    digest=hashlib.sha256(json.dumps(dict(sources=hashes,checkpoint=done['checkpoint_sha256']),sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_posterior_context_probe_stages'/digest;out=Path(E.OUTPUTS)/f'posterior_context_probe_{digest}'
    env=dict(PROBE_ROOT=str(stage),PROBE_OUT=str(out),PROBE_CHECKPOINT=str(checkpoint),TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    body=E.make_body(f'lrwkv-posterior-context-{digest[:8]}',E.wrapped(env,str(stage/'qz/launch_posterior_context_probe.sh')),'Selected competent checkpoint; original labelled serialization; fresh dev;80conditions,3policies;8H100',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(stage=str(stage),out=str(out),training_plan=str(trainplan),checkpoint_sha256=done['checkpoint_sha256'],sources=hashes,body=body,scope='Development only; checkpoint selected after observing short-task competence')
    dest=ROOT/'results/long_context_mvp'/f'plan_posterior_probe_{digest}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('already submitted')
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
        if C.already_submitted(body['name']):raise ValueError('duplicate scheduler job')
        for rel,p in sources.items():
            if K.sha(p)!=hashes[rel]:raise ValueError('changed source')
            target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and K.sha(target)!=hashes[rel]:raise ValueError('stage conflict')
            if not target.exists():shutil.copyfile(p,target)
        (stage/'probe_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
        record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))

if __name__=='__main__':
    with campaign_submission_lock(ROOT):main()
