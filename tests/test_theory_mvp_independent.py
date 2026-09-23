"""Independent probability enumeration: no rank/TC formula in reference sampler."""
from collections import defaultdict
from itertools import product
import math
import pytest
from lrwkv_evidence.theory_mvp.exact import optimal_binary_compression
from lrwkv_evidence.theory_mvp.linear_compression import kernel
from lrwkv_evidence.theory_mvp.reveal_schedule import exact_schedule


def oracle_endpoint(rows,n,steps):
    support=kernel(rows,n)
    states={(0,0):1.} # (known_mask, known_values)
    for remaining in range(steps,0,-1):
        w=1/remaining;out=defaultdict(float)
        for (known,values),mass in states.items():
            unknown=((1<<n)-1)^known
            posterior=[x for x in support if x&known==values]
            ids=[i for i in range(n) if unknown>>i&1]
            marg={i:(sum(x>>i&1 for x in posterior)/len(posterior) if posterior else .5) for i in ids}
            batch=unknown
            while True:
                chosen=[i for i in ids if batch>>i&1]
                bp=w**len(chosen)*(1-w)**(len(ids)-len(chosen))
                for bits in product([0,1],repeat=len(chosen)):
                    bitmask=sum(b<<i for i,b in zip(chosen,bits))
                    prob=math.prod(marg[i] if b else 1-marg[i] for i,b in zip(chosen,bits))
                    out[known|batch,values|bitmask]+=mass*bp*prob
                if batch==0:break
                batch=(batch-1)&unknown
        states=out
    q={x:states.get(((1<<n)-1,x),0) for x in range(1<<n)}
    assert math.isclose(sum(q.values()),1,abs_tol=1e-12)
    return support,q


@pytest.mark.parametrize('rows,n', [([7],3),([3,6],3),([3,12],4),([1,6],4)])
def test_probability_tree_matches_rank_dp(rows,n):
    for steps in [1,2,3]:
        support,q=oracle_endpoint(rows,n,steps)
        mass=sum(q[x] for x in support)
        kl=sum(math.log2((1/len(support))/q[x])/len(support) for x in support)
        tv=.5*sum(abs((1/len(support) if x in support else 0)-p) for x,p in q.items())
        r=exact_schedule(rows,n,steps)
        assert math.isclose(mass,r['valid_probability'],abs_tol=1e-12)
        assert math.isclose(kl,r['endpoint_forward_kl_bits'],abs_tol=1e-12)
        assert math.isclose(tv,r['total_variation'],abs_tol=1e-12)


def test_guard_against_accidental_giant_codebook_search():
    with pytest.raises(ValueError):optimal_binary_compression(5,4)
