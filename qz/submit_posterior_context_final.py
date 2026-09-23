"""Submit one shard of the fixed fresh-condition panel under campaign caps."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage,route_body,PRIMARY
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--role',choices=('unadapted','adapted200'),required=True);ap.add_argument('--shard',type=int,choices=range(4),required=True);ap.add_argument('--submit',action='store_true');args=ap.parse_args()
 designpath=ROOT/'theory_mvp/posterior_context_final/FROZEN_DESIGN.json';design=json.loads(designpath.read_text());designsha=sha(designpath)
 if sha(ROOT/'theory_mvp/posterior_context_final/DESIGN.md')!=design['design_markdown_sha256']:raise ValueError('changed design text')
 for rel,digest in design['sources'].items():
  if sha(ROOT/rel)!=digest:raise ValueError('changed frozen evaluation source')
 q=json.loads((ROOT/'results/long_context_mvp/plan_dacbbe1355018e7f_qualification.json').read_text())
 trainpath=ROOT/'results/train04confirm/plan_8cd809d08a96ea13_confirm_independent_seed71.json' if args.role=='unadapted' else Path(design['checkpoint_selectors']['adapted200']['training_plan'])
 train=json.loads(trainpath.read_text());done=json.loads((Path(train['out'])/'completion.json').read_text());checkpoint=Path(train['out'])/'resume.pt'
 if not done['execution_complete'] or done.get('qualification') or done['step']!=design['checkpoint_selectors'][args.role]['step'] or sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('wrong fixed checkpoint')
 if args.role=='unadapted' and done['checkpoint_sha256']!=design['checkpoint_selectors']['unadapted']['sha256']:raise ValueError('wrong original model')
 if args.role=='adapted200' and (Path(train['stage']).name!='dacbbe1355018e7f' or train['sources']!=q['sources']):raise ValueError('wrong adaptation recipe')
 sources={rel:Path(q['stage'])/rel for rel in q['sources']}
 if any(sha(p)!=q['sources'][rel] for rel,p in sources.items()):raise ValueError('changed inherited source')
 sources.update({rel:ROOT/rel for rel in design['sources']});sources['final_design.json']=designpath
 hashes={rel:sha(p) for rel,p in sources.items()}
 digest=hashlib.sha256(json.dumps(dict(sources=hashes,checkpoint=done['checkpoint_sha256']),sort_keys=True).encode()).hexdigest()[:16]
 stage=Path(E.G)/'long_rwkv_posterior_context_final_stages'/digest;out=Path(E.OUTPUTS)/f'posterior_context_final_{digest}_{args.role}_shard{args.shard}'
 env=dict(PROBE_ROOT=str(stage),PROBE_OUT=str(out),PROBE_CHECKPOINT=str(checkpoint),PROBE_SHA256=done['checkpoint_sha256'],FINAL_ROLE=args.role,FINAL_SHARD=str(args.shard),TRITON_F32_DEFAULT='ieee',TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
 body=E.make_body(f'lrwkv-pcf-{digest[:8]}-{args.role}-{args.shard}',E.wrapped(env,str(stage/'qz/launch_posterior_context_final.sh')),'Frozen fresh-condition panel;selected lineage;all80conditions per shard;8H100',E.SPEC_8GPU)
 body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0,max_running_time_ms='10800000')
 record=dict(stage=str(stage),out=str(out),training_plan=str(trainpath),checkpoint_sha256=done['checkpoint_sha256'],sources=hashes,body=body,role=args.role,shard=args.shard,design_sha256=designsha)
 dest=ROOT/'results/posterior_context_final'/f'plan_{args.role}_shard{args.shard}.json'
 if dest.exists() and json.loads(dest.read_text()).get('submission'):raise ValueError('already submitted')
 if args.submit:
  record['capacity']=live_usage(C.list_all_jobs());record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
  if C.already_submitted(body['name']):raise ValueError('duplicate scheduler job')
  for rel,p in sources.items():
   if sha(p)!=hashes[rel]:raise ValueError('source changed during staging')
   target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and sha(target)!=hashes[rel]:raise ValueError('stage conflict')
   if not target.exists():shutil.copyfile(p,target)
  (stage/'final_sources.json').write_text(json.dumps(hashes,indent=2)+'\n');record['submission']=C.submit_one(body,dry_run=False)
 dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body')},indent=2))
if __name__=='__main__':
 with campaign_submission_lock(ROOT):main()
