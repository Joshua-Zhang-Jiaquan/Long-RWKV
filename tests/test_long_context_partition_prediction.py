from collections import Counter
from itertools import combinations
import math
import pytest
from theory_mvp.focused_paper.partition_prediction import prediction
from lrwkv_evidence.long_context_mvp import tasks as T


def test_all_seventy_partitions_match_enumerated_oracle_law():
    ex=T.problem('dev',20270923,0,'dependent');histogram=Counter();total_valid=0.
    for first in combinations(range(8),4):
        groups=[first,[i for i in range(8) if i not in first]]
        probabilities=[]
        for y in ex['support']:
            visible={};q=1.
            for group in groups:
                marginal=T.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8)
                for i in group:
                    b=(y>>i)&1;q*=marginal[i] if b else 1-marginal[i]
                visible.update({i:(y>>i)&1 for i in group})
            probabilities.append(q)
        expected=prediction(groups);kl=-math.log(16)-sum(math.log(q) for q in probabilities)/16
        assert expected['dependence_nats']==pytest.approx(kl,abs=1e-12)
        assert expected['oracle_valid_mass']==pytest.approx(sum(probabilities))
        histogram[expected['same_group_pairs']]+=1;total_valid+=sum(probabilities)
    assert histogram=={0:16,2:48,4:6}
    assert total_valid/70==pytest.approx(227/560)


def test_equal_call_policies_can_have_different_dependence():
    good=prediction([list(range(4)),list(range(4,8))])
    bad=prediction([[0,1,4,5],[2,3,6,7]])
    assert good['calls']==bad['calls']==2
    assert good['dependence_nats']==0 and bad['dependence_nats']==4*math.log(2)
    assert good['oracle_valid_mass']==1 and bad['oracle_valid_mass']==1/16
