import argparse,hashlib,json,shutil,sys
from pathlib import Path
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage,route_body,PRIMARY
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('rwkv','attention'),required=True);ap.add_argument('--submit',action='store_true');args=ap.parse_args()
 designpath=ROOT/'theory_mvp/posterior_context_cost/FROZEN_DESIGN.json';design=json.loads(designpath.read_text());designsha=sha(designpath)
 if sha(ROOT/'theory_mvp/posterior_context_cost/DESIGN.md')!=design['design_text_sha256'] or sha(ROOT/'theory_mvp/posterior_context_final/FROZEN_DESIGN.json')!=design['final_design_sha256']:raise ValueError('changed design')
 qadapt=ROOT/'results/long_context_mvp/plan_dacbbe1355018e7f_qualification.json'
 attention=ROOT/'results/long_context_mvp/plan_baseline_attention_c0692a0476e06ef9_qualification.json'
 sources={}
 for planpath in (qadapt,attention):
  plan=json.loads(planpath.read_text())
  for rel,digest in plan['sources'].items():
   p=Path(plan['stage'])/rel
   if sha(p)!=digest:raise ValueError('changed inherited source')
   if rel in sources and sha(sources[rel])!=digest:raise ValueError('conflicting inherited source: '+rel)
   sources[rel]=p
 for rel,digest in design['sources'].items():
  if sha(ROOT/rel)!=digest:raise ValueError('changed frozen worker')
  sources[rel]=ROOT/rel
 for rel,digest in json.loads((ROOT/'theory_mvp/posterior_context_final/FROZEN_DESIGN.json').read_text())['sources'].items():
  if sha(ROOT/rel)!=digest:raise ValueError('changed frozen task')
  sources[rel]=ROOT/rel
 sources['cost_design.json']=designpath;sources['final_design.json']=ROOT/'theory_mvp/posterior_context_final/FROZEN_DESIGN.json'
 trainpath=attention if args.kind=='attention' else ROOT/'results/long_context_mvp/plan_dacbbe1355018e7f_development200.json'
 training=json.loads(trainpath.read_text());outtrain=Path(training['out']);done=json.loads((outtrain/'completion.json').read_text());checkpoint=outtrain/'resume.pt'
 if not done['execution_complete'] or done['step']!=(20 if args.kind=='attention' else 200) or sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('wrong cost checkpoint')
 hashes={rel:sha(p) for rel,p in sources.items()};digest=hashlib.sha256(json.dumps(dict(sources=hashes,checkpoint=done['checkpoint_sha256'],kind=args.kind),sort_keys=True).encode()).hexdigest()[:16]
 stage=Path(E.G)/'long_rwkv_posterior_context_cost_stages'/digest;out=Path(E.OUTPUTS)/f'posterior_context_cost_{args.kind}_{digest}'
 base=Path(E.G)/'models'/('pythia-410m' if args.kind=='attention' else 'rwkv7-0.4B')
 env=dict(COST_ROOT=str(stage),COST_OUT=str(out),COST_CHECKPOINT=str(checkpoint),COST_SHA256=done['checkpoint_sha256'],COST_KIND=args.kind,COST_BASE=str(base),TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
 body=E.make_body(f'lrwkv-pccost-{args.kind}-{digest[:8]}',E.wrapped(env,str(stage/'qz/launch_posterior_context_cost.sh')),'Frozen matched-task16K mean-budget check;48conditions,3policies,10timings;8H100',E.SPEC_8GPU)
 body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
 record=dict(stage=str(stage),out=str(out),training_plan=str(trainpath),checkpoint_sha256=done['checkpoint_sha256'],sources=hashes,body=body,comparator=args.kind,design_sha256=designsha)
 dest=ROOT/'results/posterior_context_cost'/f'plan_{args.kind}.json'
 if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('already submitted')
 if args.submit:
  record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
  if C.already_submitted(body['name']):raise ValueError('duplicate scheduler job')
  for rel,p in sources.items():
   if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
   target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and sha(target)!=hashes[rel]:raise ValueError('stage conflict')
   if not target.exists():shutil.copyfile(p,target)
  (stage/'cost_sources.json').write_text(json.dumps(hashes,indent=2)+'\n');record['submission']=C.submit_one(body,dry_run=False)
 dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))
if __name__=='__main__':
 with campaign_submission_lock(ROOT):main()
