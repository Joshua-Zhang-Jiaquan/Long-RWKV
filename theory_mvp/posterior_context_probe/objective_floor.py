"""Reconstruct policy entropy floors; strictly an in-sample optimization diagnostic."""
import argparse,json,math
from pathlib import Path
from lrwkv_evidence.posterior_context_adapt import training as R

class Tokens:
 binary_ids=(48,49);mask_id=65535
 def encode(self,text):return [99]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--training-plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 plan=json.loads(args.training_plan.read_text());rows=[json.loads(x) for x in (Path(plan['out'])/'train.jsonl').read_text().splitlines()]
 if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('incomplete log')
 # Token identities and filler lengths do not enter the conditional entropy;
 # mock only serialization in this isolated CPU diagnostic process.
 R.serialize=lambda *args,**kwargs:dict(ids=[99])
 result=[]
 for row in rows:
  floor=0.
  for rank in range(8):
   for c in R.records(row['step'],rank,Tokens()):
    floor+=sum(math.log(2) if p==.5 else 0 for p,m in zip(c['oracle'],c['loss_mask']) if m)*c['loss_scale']/32
  excess=row['loss']-floor
  if excess < -1e-6:raise ValueError('negative excess beyond FP32 tolerance')
  result.append(dict(step=row['step'],loss=row['loss'],oracle_entropy_floor=floor,excess_ce=excess))
 out=dict(rows=result,step_count=len(result),last20_mean_excess=sum(r['excess_ce'] for r in result[-20:])/len(result[-20:]),scope='In-sample batch excess CE under the pinned policy objective. No held-out, long-context or endpoint competence certificate.')
 args.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
