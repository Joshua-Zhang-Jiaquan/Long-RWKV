"""Analytic predictions for the controlled pair posterior, not model results."""
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path


def prediction(groups):
    flat = [i for group in groups for i in group]
    if sorted(flat) != list(range(8)) or any(not group for group in groups):
        raise ValueError('requires an ordered partition of eight positions')
    location = {i: k for k, group in enumerate(groups) for i in group}
    same_group_pairs = sum(location[i] == location[i+4] for i in range(4))
    return dict(calls=len(groups), same_group_pairs=same_group_pairs,
                dependence_nats=same_group_pairs*math.log(2),
                oracle_valid_mass=2.**(-same_group_pairs))


def main():
    histogram = Counter()
    for first in combinations(range(8), 4):
        second = [i for i in range(8) if i not in first]
        histogram[prediction([first, second])['same_group_pairs']] += 1
    result = dict(scope='Exact analytic oracle predictions; no learned model measurements',
                  uniform_four_four_partition_count=70, same_group_pair_histogram=dict(histogram),
                  expected_dependence_nats=sum(k*n for k,n in histogram.items())/70*math.log(2),
                  expected_oracle_valid_mass=sum(2.**(-k)*n for k,n in histogram.items())/70,
                  one=prediction([list(range(8))]),
                  information_set=prediction([list(range(4)),list(range(4,8))]),
                  sequential=prediction([[i] for i in range(8)]))
    destination=Path('results/long_context_mvp/analytic_partition_predictions.json')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
