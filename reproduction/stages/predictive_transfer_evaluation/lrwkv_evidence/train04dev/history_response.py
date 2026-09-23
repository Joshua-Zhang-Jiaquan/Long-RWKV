"""Public-condition-preserving interventions on information-set histories.

Only feasible teacher-forced histories are compared. No solved values replace
model predictions. This reports a diagnostic, not a new independent error bound.
"""
import itertools
import math
import statistics
from lrwkv_evidence.train04.evaluate import fixed_groups


def history_response(example,cache):
    free,dependent=fixed_groups(example,'information_set');n=example['n']
    stage=math.ceil(8*(n-len(free))/n)
    paired=[];unrelated=[];systematic=[]
    for f in free:
        others=[i for i in free if i!=f]
        for values in itertools.product((0,1),repeat=len(others)):
            base=dict(zip(others,values));a={**base,f:0};b={**base,f:1}
            # Missing entries are an error: this diagnostic must not add model calls.
            before=cache[stage,tuple(sorted(a.items()))]
            after=cache[stage,tuple(sorted(b.items()))]
            for j in dependent:
                if example['family']=='paired_parity':
                    matches=[c for row,c in zip(example['matrix_rows'],example['syndrome']) if row==((1<<f)|(1<<j))]
                    if matches:
                        target_after=1^matches[0]
                        paired.append(math.exp(after[j][target_after])-math.exp(before[j][target_after]))
                    else:
                        unrelated.append(abs(math.exp(after[j][1])-math.exp(before[j][1])))
                elif example['family']=='systematic':
                    systematic.append(abs(math.exp(after[j][1])-math.exp(before[j][1])))
                else:raise ValueError('unsupported family')
    def average(values):return statistics.mean(values) if values else None
    return dict(paired_signed_response=average(paired),off_pair_absolute_response=average(unrelated),
                systematic_absolute_response=average(systematic),paired_comparisons=len(paired),
                off_pair_comparisons=len(unrelated),systematic_comparisons=len(systematic),additional_model_calls=0)
