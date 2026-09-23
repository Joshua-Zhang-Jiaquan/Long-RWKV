"""All frozen conditions, paired checkpoints, no endpoint-dependent filtering."""
import json,statistics
from pathlib import Path
from lrwkv_evidence.train04.worker import file_sha
from .paired import analyze,bootstrap_mean
ROOT=Path(__file__).resolve().parents[2]

def main():
 folder=ROOT/'results/posterior_context_final';designpath=ROOT/'theory_mvp/posterior_context_final/FROZEN_DESIGN.json';design=json.loads(designpath.read_text());digest=file_sha(designpath)
 roles={};conditions={};rawhash={};checkpoints={}
 for role in ('unadapted','adapted200'):
  allrows=[]
  for shard in range(4):
   p=folder/f'exact_{role}_shard{shard}.json';report=json.loads(p.read_text())
   if not report['execution_complete'] or report['design_sha256']!=digest or report['role']!=role or report['shard']!=shard:raise ValueError('wrong frozen shard')
   if role in checkpoints and checkpoints[role]!=report['checkpoint_sha256']:raise ValueError('mixed checkpoints')
   checkpoints[role]=report['checkpoint_sha256']
   for path,sha in report['raw_sha256'].items():
    if file_sha(Path(path))!=sha:raise ValueError('changed raw evidence')
    rawhash[path]=sha
   allrows+=report['conditions']
  if len(allrows)!=320:raise ValueError('incomplete320-condition panel')
  # Normalize the alternating global index to one family-local base index.
  normalized=[dict(r,index=r['index']//2,requested_context_tokens=r['serial']['requested_context_tokens'],position=r['serial']['position']) for r in allrows]
  roles[role]=analyze(normalized,range(16),bootstrap_seed=design['bootstrap']['seed'],repeats=design['bootstrap']['repeats'])
  conditions[role]=normalized
 def endpoints(role,family,length,pos,method,metric):
  rows=sorted([r for r in conditions[role] if r['family']==family and r['requested_context_tokens']==length and r['position']==pos],key=lambda r:r['index'])
  if [r['index'] for r in rows]!=list(range(16)):raise ValueError('unpaired indices')
  return [next(x for x in r['exact']['results'] if x['method']==method)[metric] for r in rows]
 summaries=[];adaptation=[]
 for role in roles:
  for family in ('systematic','paired_parity'):
   for length,pos in [(None,'evidence_only')]+[(n,p) for n in (1024,4096,16384) for p in ('far','middle','near')]:
    for method in design['methods']:
     kl=endpoints(role,family,length,pos,method,'joint_kl_nats');valid=endpoints(role,family,length,pos,method,'exact_valid_mass')
     summaries.append(dict(role=role,family=family,context_tokens=length,position=pos,method=method,mean_kl=statistics.mean(kl),mean_valid=statistics.mean(valid),kl_interval=bootstrap_mean(kl,seed=design['bootstrap']['seed'],repeats=design['bootstrap']['repeats'])['pointwise_95_percentile_interval'],valid_interval=bootstrap_mean(valid,seed=design['bootstrap']['seed'],repeats=design['bootstrap']['repeats'])['pointwise_95_percentile_interval']))
 for metric in ('joint_kl_nats','exact_valid_mass'):
  for pos in ('far','middle','near','equal_weight_mean_of_three_positions'):
   positions=('far','middle','near') if pos=='equal_weight_mean_of_three_positions' else (pos,)
   arrays=[]
   for position in positions:
    before=endpoints('unadapted','paired_parity',16384,position,'information_set',metric);after=endpoints('adapted200','paired_parity',16384,position,'information_set',metric)
    arrays.append([(b-a if metric=='joint_kl_nats' else a-b) for b,a in zip(before,after)])
   values=[statistics.mean(xs) for xs in zip(*arrays)]
   adaptation.append(dict(family='paired_parity',context_tokens=16384,position=pos,method='information_set',metric=metric,**bootstrap_mean(values,seed=design['bootstrap']['seed'],repeats=design['bootstrap']['repeats'])))
 result=dict(execution_complete=True,design_sha256=digest,checkpoint_sha256=checkpoints,raw_sha256=rawhash,checkpoint_analyses=roles,adaptation_contrasts=adaptation,summary=summaries,scope=design['selection_disclosure'],conditions_per_checkpoint=320,base_problems_per_family=16)
 result['analysis_sources']={str(p.relative_to(ROOT)):file_sha(p) for p in (Path(__file__),Path(__file__).with_name('paired.py'),Path(__file__).with_name('collect.py'),Path(__file__).with_name('figures.py'))}
 (folder/'final_report.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Frozen fresh-condition results','',result['scope'],'','Positive gain means better information-set sampling. Pointwise95% paired bootstrap intervals over16 base problems/family; no training-seed uncertainty.','','| Checkpoint | Comparator | Metric |16K mean-position gain|95% interval|','|---|---|---|---:|---|']
 for role,analysis in roles.items():
  for r in analysis['primary_estimands']:lines.append(f"|{role}|{r['comparator']}|{r['metric']}|{r['mean']:.7g}|{r['pointwise_95_percentile_interval']}|")
 lines+=['','All cells, paired length interactions and adapted-minus-original contrasts are retained in final_report.json.']
 from .figures import write
 write(result,folder)
 (folder/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
