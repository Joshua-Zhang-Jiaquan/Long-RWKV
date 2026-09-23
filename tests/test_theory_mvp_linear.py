from lrwkv_evidence.theory_mvp.linear_compression import *


def test_equal_information_different_parallel_error():
    systematic=[1,2];parity=[3,12]
    assert rank(systematic)==rank(parity)==2
    assert tc_formula(systematic,15,15,4)==0
    assert tc_formula(parity,15,15,4)==2
    assert entropy(kernel(systematic,4))==entropy(kernel(parity,4))==2


def test_information_set_two_round_exact():
    rows=[3,6,12];support=kernel(rows,4);info=information_set(rows,4)
    assert support==[0,15]
    assert info.bit_count()==1
    assert direct_tc(support,info,4)==0
    assert len({x&info for x in support})==len(support)


def test_partial_posterior_formula():
    rows=[3,12]
    assert tc_formula(rows,14,14,4)==1
    assert tc_formula(rows,14,2,4)==0
    assert tc_formula(rows,14,12,4)==1


def test_all_rank_two_rowspaces():
    assert len(list(enumerate_rowspaces(4,2)))==35


def test_pairwise_independence_does_not_certify_product_joint():
    support=kernel([15,51],6)
    assert all(direct_tc(support,(1<<a)|(1<<b),6)==0 for a,b in combinations(range(6),2))
    assert direct_tc(support,63,6)==2
