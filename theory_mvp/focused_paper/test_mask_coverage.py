import math
import pytest
from theory_mvp.focused_paper.mask_coverage import ordinary_query_law,policy_query_law,density_ratio,results

def test_weighted_mask_law_normalization():
    for n in (2,4,8):
        law=ordinary_query_law(n)
        assert sum(law.values())==pytest.approx(1)
        for t in range(1,n+1):assert sum(v for (s,m,i),v in law.items() if s==t)==pytest.approx(1/n)

def test_sharp_balanced_history_ratio():
    for n in (2,4,8):
        ordinary=ordinary_query_law(n)
        policy=policy_query_law([list(range(n//2)),list(range(n//2,n))],n)
        assert density_ratio(policy,ordinary)==pytest.approx((n//2)*2**n)
        key=max(policy,key=lambda k:policy[k]/ordinary[k])
        # Error concentrated at the maximizing query attains the ratio.
        assert policy[key]/ordinary[key]==pytest.approx(density_ratio(policy,ordinary))

def test_policy_mixture_does_not_cover_other_histories_equally():
    r=results()
    assert r['information_set_under_mixture']<2
    assert r['sequential_under_mixture']>1900
    assert r['pair_preserving_under_mixture']==pytest.approx(2048)
    assert r['sequential_under_permuted_information_set_mixture']==pytest.approx(2048)
    info=policy_query_law([list(range(4)),list(range(4,8))])
    sequential=policy_query_law([[i] for i in range(8)])
    assert math.isinf(density_ratio(sequential,info))
