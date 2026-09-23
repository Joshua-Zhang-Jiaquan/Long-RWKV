import math
import pytest
from tools.verify_evidence import example,header_metrics,estimate,raw_records

def test_one_call_floor_and_oracle_two_pass():
    ex=example(0,2026092301);initial=[[math.log(.5)]*2 for _ in range(8)];conditionals=[];eps=1e-8
    for h in range(16):
        visible={i:(h>>(3-j))&1 for j,i in enumerate(ex['information_set'])}
        target=next(y for y in ex['support'] if all((y>>i)&1==v for i,v in visible.items()))
        conditionals.append([[math.log(1-eps if (target>>i)&1==b else eps) for b in (0,1)] for i in range(8)])
    result=header_metrics(ex,initial,conditionals)
    assert result['one_kl']==pytest.approx(4*math.log(2))
    assert result['info_kl']==pytest.approx(-4*math.log(1-eps))
    assert result['valid_mass_info']==pytest.approx((1-eps)**4)
    chance=header_metrics(ex,initial,[initial]*16)
    assert chance['gain_over_one']==pytest.approx(0)

def test_missing_history_is_rejected():
    with pytest.raises(ValueError,match='all histories'):header_metrics(example(0,2026092301),[],[[]]*15)

def test_unnormalized_probabilities_rejected():
    with pytest.raises(ValueError,match='unnormalized'):header_metrics(example(0,2026092301),[[0.,0.]]*8,[[[0.,0.]]*8]*16)

def test_missing_rank_census_rejected():
    with pytest.raises(ValueError,match='census'):raw_records({'files':[]})

def test_incomplete_bootstrap_panel_rejected():
    with pytest.raises(ValueError,match='incomplete'):estimate([0.]*31,2026092302)

def test_new_public_prompts_are_disjoint():
    a={example(i,20271021)['prompt'] for i in range(32)}
    b={example(i,2026092301)['prompt'] for i in range(32)}
    assert len(a)==len(b)==32 and not a&b
