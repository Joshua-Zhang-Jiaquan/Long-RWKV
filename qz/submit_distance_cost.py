"""Matched fresh-condition attention cost, under the existing campaign cap."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage,route_body,PRIMARY
from distance_admission import run_when_capacity
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--submit',action='store_true');args=ap.parse_args()
    folder=ROOT/'results/distance_intervention_cost';folder.mkdir(exist_ok=True)
    designpath=ROOT/'theory_mvp/distance_intervention/FROZEN_COST_SUPPLEMENT.json'
    design=json.loads(designpath.read_text())
    if sha(ROOT/'theory_mvp/distance_intervention/MATCHED_COST_SUPPLEMENT.md')!=design['design_text_sha256']:raise ValueError('changed cost protocol')
    finalpath=ROOT/'theory_mvp/distance_intervention/FROZEN_EVALUATION.json'
    if sha(finalpath)!=design['final_design_sha256']:raise ValueError('changed evaluation protocol')
    inherited=json.loads((ROOT/'results/posterior_context_cost/plan_attention.json').read_text())
    sources={rel:Path(inherited['stage'])/rel for rel in inherited['sources'] if rel not in ('cost_design.json','final_design.json')}
    for rel,p in sources.items():
        if sha(p)!=inherited['sources'][rel]:raise ValueError('changed qualified source')
    for rel,digest in design['sources'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('changed frozen new worker/task')
        sources[rel]=ROOT/rel
    sources['cost_design.json']=designpath;sources['final_design.json']=finalpath
    trainpath=Path(inherited['training_plan']);training=json.loads(trainpath.read_text());trainout=Path(training['out'])
    checkpoint=trainout/'resume.pt';done=json.loads((trainout/'completion.json').read_text())
    if not done['execution_complete'] or done['step']!=20 or sha(checkpoint)!=design['checkpoint_sha256']:raise ValueError('wrong qualified attention checkpoint')
    hashes={rel:sha(p) for rel,p in sources.items()}
    digest=hashlib.sha256(json.dumps(dict(sources=hashes,checkpoint=done['checkpoint_sha256']),sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_distance_cost_stages'/digest;out=Path(E.OUTPUTS)/f'distance_cost_attention_{digest}'
    env=dict(COST_ROOT=str(stage),COST_OUT=str(out),COST_CHECKPOINT=str(checkpoint),COST_SHA256=done['checkpoint_sha256'],
             COST_KIND='attention',COST_BASE=str(Path(E.G)/'models/pythia-410m'),TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    body=E.make_body(f'lrwkv-distcost-{digest[:8]}',E.wrapped(env,str(stage/'qz/launch_distance_cost.sh')),
                     'Matched fresh16K cost;8problems,3positions,3policies,5timings;no quality claim;8H100 under48cap',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
    record=dict(stage=str(stage),out=str(out),sources=hashes,body=body,training_plan=str(trainpath),
                checkpoint_sha256=done['checkpoint_sha256'],design_sha256=sha(designpath),comparator='attention')
    dest=folder/'plan_attention.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('duplicate cost submission')
    if args.submit:
        record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
        if C.already_submitted(body['name']):raise ValueError('duplicate scheduler job')
        for rel,p in sources.items():
            if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
            target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and sha(target)!=hashes[rel]:raise ValueError('stage conflict')
            if not target.exists():shutil.copyfile(p,target)
        (stage/'cost_sources.json').write_text(json.dumps(hashes,indent=2)+'\n')
        record['submission']=C.submit_one(body,dry_run=False)
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))


if __name__=='__main__':
    run_when_capacity(ROOT,main)
