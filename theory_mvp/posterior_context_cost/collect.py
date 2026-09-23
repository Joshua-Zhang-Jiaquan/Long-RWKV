import argparse,json,math,statistics
from pathlib import Path
from lrwkv_evidence.train04.worker import file_sha,load_tokenizer
from lrwkv_evidence.long_context_baselines.models import Tokenizer
from lrwkv_evidence.posterior_context_final import tasks as T
from lrwkv_evidence.long_context_eval.numerical import validate as validate_numerical
from theory_mvp.posterior_context_final.paired import bootstrap_mean

def collect(planpath):
 plan=json.loads(planpath.read_text());stage=Path(plan['stage']);out=Path(plan['out']);kind=plan['comparator']
 for rel,digest in plan['sources'].items():
  if file_sha(stage/rel)!=digest:raise ValueError('changed staged source')
 if file_sha(stage/'cost_design.json')!=plan['design_sha256']:raise ValueError('wrong cost design')
 design=json.loads((stage/'cost_design.json').read_text());training=json.loads(Path(plan['training_plan']).read_text())
 if file_sha(Path(training['out'])/'resume.pt')!=plan['checkpoint_sha256']:raise ValueError('changed checkpoint')
 base=Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/models')
 tok=Tokenizer(base/'pythia-410m') if kind=='attention' else load_tokenizer(base/'rwkv7-0.4B')[0]
 conditions=[];hashes={}
 for rank in range(8):
  f=out/f'rank{rank}.jsonl';hashes[str(f)]=file_sha(f);rows=[json.loads(x) for x in f.read_text().splitlines()]
  if len(rows)!=9 or [r['kind'] for r in rows]!=['provenance','numerical_screen']+['condition']*6+['complete']:raise ValueError('incomplete cost shard')
  p=rows[0]
  if p['rank']!=rank or p['world']!=8 or p['comparator']!=kind or p['checkpoint_sha256']!=plan['checkpoint_sha256'] or p['design_sha256']!=plan['design_sha256'] or p['final_panel_sha256']!=design['final_design_sha256'] or p['budget_seconds']!=1.5 or p['timing_repeats']!=10:raise ValueError('wrong provenance')
  validate_numerical(rows[1]['rows']);expected=[(i,pos) for i in (2*rank+1,2*rank+17) for pos in ('far','middle','near')]
  if [(r['index'],r['serial']['position']) for r in rows[2:-1]]!=expected:raise ValueError('wrong conditions')
  for r in rows[2:-1]:
   ex=T.example(r['index']);serial=T.serialize(ex,tok,16384,r['serial']['position'])
   if r['family']!='paired_parity' or r['instance_id']!=ex['instance_id'] or r['serial']!={k:v for k,v in serial.items() if k!='ids'}:raise ValueError('wrong public input')
   if [c['method'] for c in r['costs']]!=design['methods']:raise ValueError('wrong policies')
   for c in r['costs']:
    if c['calls']!=(1 if c['method']=='one' else 2) or len(c['seconds'])!=10 or len(c['sample_outputs'])!=10 or not all(math.isfinite(t) and t>0 for t in c['seconds']) or any(type(v) is not int or not 0<=v<256 for v in c['sample_outputs']) or c['peak_allocated_bytes']<=0:raise ValueError('invalid timing record')
   conditions.append(r)
 summaries=[]
 for method in design['methods']:
  perbase=[];peaks=[]
  for index in design['indices']:
   selected=[r for r in conditions if r['index']==index]
   if len(selected)!=3:raise ValueError('missing paired positions')
   costs=[next(c for c in r['costs'] if c['method']==method) for r in selected]
   perbase.append(statistics.mean(statistics.mean(c['seconds']) for c in costs));peaks.extend(c['peak_allocated_bytes'] for c in costs)
  summaries.append(dict(method=method,mean_request_seconds=statistics.mean(perbase),base_mean_seconds=perbase,peak_allocated_bytes=max(peaks),**bootstrap_mean(perbase,seed=20271002,repeats=10000)))
 return dict(execution_complete=True,comparator=kind,design_sha256=plan['design_sha256'],final_design_sha256=design['final_design_sha256'],checkpoint_sha256=plan['checkpoint_sha256'],raw_sha256=hashes,conditions=conditions,summary=summaries,scope='Batch1 complete requests; pointwise problem-bootstrap intervals; timing samples are not accuracy estimates')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();r=collect(a.plan);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('conditions','raw_sha256')},indent=2))
if __name__=='__main__':main()
