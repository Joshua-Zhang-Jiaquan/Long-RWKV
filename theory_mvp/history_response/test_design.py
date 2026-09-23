import math
import pytest
from lrwkv_evidence.history_response import design as H
from lrwkv_evidence.distance_intervention import tasks as T


@pytest.mark.parametrize('index',range(32))
def test_all_histories_constraints_and_frozen_probe(index):
    ex=T.example(index)
    assert len({tuple(v.items()) for v in H.histories(ex)})==16
    for r in range(4):
        cf=H.flip(ex,r)
        assert cf['index']==index
        for visible in H.histories(ex):
            m=H.meta(ex,cf,visible,r)
            assert m['original_target'] != m['flipped_target']
            assert len(m['unaffected_targets'])==3
    frozen,meta=T.counterfactual(ex)
    assert H.flip(ex,index%4)==frozen
    assert H.meta(ex,frozen,H.histories(ex)[0],index%4)==meta


def test_oracle_and_insensitive_predictors():
    ex=T.example(0);eps=1e-6
    initial=[[math.log(.5)]*2 for _ in range(8)]
    def oracle(e,v):
        return [[math.log(1-eps if b==gold else eps) for b in range(2)] for gold in H.targets(e,v)]
    original=[oracle(ex,v) for v in H.histories(ex)]
    flipped=[[oracle(H.flip(ex,r),v) for v in H.histories(ex)] for r in range(4)]
    result=H.summarize(ex,initial,original,flipped)
    assert result['second_error']==pytest.approx(-4*math.log(1-eps))
    assert result['response']['mean_correct_conditional_ce']==pytest.approx(-math.log(1-eps))
    assert result['response']['centering_penalty']==pytest.approx(0,abs=1e-12)
    assert result['response']['both_variants_correct']==1
    assert result['max_unaffected_change']==0
    insensitive=H.summarize(ex,initial,original,[original]*4)
    assert insensitive['response']['response_ce_lower_bound']==pytest.approx(math.log(2))
    assert insensitive['response']['centering_penalty']>6
    assert insensitive['response']['both_variants_correct']==0


def test_reject_missing_or_unnormalized_calls():
    ex=T.example(0);lp=[[math.log(.5)]*2 for _ in range(8)]
    with pytest.raises(ValueError):H.summarize(ex,lp,[lp]*15,[[lp]*16]*4)
    with pytest.raises(ValueError):H.validate_lp([[0.,0.]]*8)
