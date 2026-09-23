import math
from lrwkv_evidence.theory_mvp.reveal_schedule import exact_schedule


def test_single_pair_collision_closed_form():
    # Two parity-linked bits: collision probability1/T; correctmass1-1/(2T).
    for t in [1,2,8,64]:
        r=exact_schedule([3],2,t)
        assert math.isclose(r['valid_probability'],1-1/(2*t),abs_tol=1e-12)
        assert math.isclose(r['path_dependence_bits'],1/t,abs_tol=1e-12)


def test_two_disjoint_pairs_factorize():
    for t in [1,2,8]:
        r=exact_schedule([3,12],4,t)
        assert math.isclose(r['valid_probability'],(1-1/(2*t))**2,abs_tol=1e-12)
        assert math.isclose(r['path_dependence_bits'],2/t,abs_tol=1e-12)


def test_systematic_code_always_exact():
    for t in [1,2,8]:
        r=exact_schedule([1,2],4,t)
        assert abs(r['endpoint_forward_kl_bits'])<1e-12
        assert r['path_dependence_bits']==0
