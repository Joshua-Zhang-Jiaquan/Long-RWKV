"""Descriptive analysis added after observing large unaffected-target changes."""
import itertools
import json
from pathlib import Path
import sys
import math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.distance_intervention.tasks import example


def main():
 folder=ROOT/'results/history_response';result={}
 for seed in (20271011,20271012,20271013):
  role=f'balanced_seed{seed}';plan=json.loads((folder/f'plan_{role}.json').read_text());positions={}
  for path in Path(plan['out']).glob('rank*.jsonl'):
   for row in map(json.loads,path.read_text().splitlines()):
    if row['kind']!='condition':continue
    ex=example(row['index']);hidden=[i for i in range(8) if i not in ex['information_set']]
    stats=positions.setdefault(row['position'],dict(unaffected_predictions=0,correct_to_incorrect=0,interventions=0,interventions_with_collateral_error=0,full_flipped_second_error_sum=0.,worst_probability_change=0.,worst_case=None))
    for h,bits in enumerate(itertools.product((0,1),repeat=4)):
     feasible=[y for y in ex['support'] if all((y>>i)&1==b for i,b in zip(ex['information_set'],bits))]
     if len(feasible)!=1:raise ValueError('history oracle failure')
     gold=[(feasible[0]>>i)&1 for i in range(8)]
     for r,mask in enumerate(ex['matrix_rows']):
      i=next(i for i in hidden if mask>>i&1);a=row['original'][h];b=row['flipped'][r][h];collateral=False
      stats['interventions']+=1
      stats['full_flipped_second_error_sum']-=sum(b[j][1-gold[j] if j==i else gold[j]] for j in hidden)
      for j in hidden:
       if j==i:continue
       stats['unaffected_predictions']+=1
       failure=a[j][gold[j]]>a[j][1-gold[j]] and b[j][gold[j]]<=b[j][1-gold[j]]
       stats['correct_to_incorrect']+=int(failure);collateral|=failure
       shift=abs(math.exp(a[j][1])-math.exp(b[j][1]))
       if shift>stats['worst_probability_change']:
        stats['worst_probability_change']=shift
        stats['worst_case']=dict(index=row['index'],history=h,constraint=r,unaffected_coordinate=j,required_bit=gold[j],original_probability_one=math.exp(a[j][1]),flipped_probability_one=math.exp(b[j][1]))
      stats['interventions_with_collateral_error']+=int(collateral)
  for p,s in positions.items():
   if s['interventions']!=2048 or s['unaffected_predictions']!=6144:raise ValueError('incomplete panel')
   s['mean_full_flipped_second_error']=s.pop('full_flipped_second_error_sum')/s['interventions']
  result[role]=positions
 payload=dict(scope='Post hoc descriptive diagnosis motivated by observed large unaffected-target changes; no new inferential test. Full flipped second error averages all four remaining targets across the four flipped prompts and all16 target histories. The flipped first-round error was not measured, so this is not a full counterfactual endpoint KL.',checkpoints=result)
 (folder/'collateral_diagnostic.json').write_text(json.dumps(payload,indent=2)+'\n')
 for role,positions in result.items():
  for p,s in positions.items():print(role,p,s['correct_to_incorrect'],'/',s['unaffected_predictions'],'collateral target errors; mean full flipped second error',s['mean_full_flipped_second_error'])


if __name__=='__main__':main()
