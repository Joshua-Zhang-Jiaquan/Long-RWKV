import copy
import pytest
from theory_mvp.long_context_final.paired import CONTEXTS, analyze, bootstrap_mean


def panel():
    rows=[]
    for family in ('retrieval','dependent'):
        for index in range(4):
            for length,position in CONTEXTS:
                # Known per-problem gain; identical across positions so their
                # repetition must not narrow uncertainty or triple sample n.
                gain=(index+1)/10 if length is not None else .05
                metrics=[dict(method=method,joint_kl_nats=1-gain if method=='information_set' else 1,
                              exact_valid_mass=.5+gain if method=='information_set' else .5)
                         for method in ('one','information_set','pair_preserving_halves')]
                rows.append(dict(family=family,index=index,requested_context_tokens=length,
                                 position=position,exact=dict(results=metrics)))
    return rows


def test_pairing_and_position_aggregation_preserve_the_base_unit():
    result=analyze(panel(),range(4),repeats=1000)
    expected=bootstrap_mean([.1,.2,.3,.4],repeats=1000)
    for row in result['primary_estimands']:
        assert row['base_problems']==4
        assert row['mean']==pytest.approx(.25)
        assert row['pointwise_95_percentile_interval']==pytest.approx(expected['pointwise_95_percentile_interval'])
    assert all(r['mean']==pytest.approx(.2) for r in result['length_interactions'])


def test_identical_paired_gains_have_no_between_problem_variation():
    result=bootstrap_mean([.2]*8,repeats=1000)
    assert result['pointwise_95_percentile_interval']==pytest.approx([.2,.2])


@pytest.mark.parametrize('change',['missing','duplicate','extra_index','invalid','policy'])
def test_no_silent_filtering_of_bad_or_missing_cells(change):
    rows=panel()
    if change=='missing':rows.pop()
    elif change=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif change=='extra_index':rows[0]['index']=19
    elif change=='invalid':rows[0]['exact']['results'][0]['exact_valid_mass']=float('nan')
    elif change=='policy':rows[0]['exact']['results'].pop()
    with pytest.raises(ValueError):analyze(rows,range(4),repeats=100)
