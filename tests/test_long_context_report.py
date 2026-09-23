import copy
import math
import pytest
from lrwkv_evidence.long_context_eval.exact import audit, partitions
from lrwkv_evidence.long_context_eval.report import summarize
from lrwkv_evidence.long_context_mvp import tasks as T


def report(kind, scale):
    ex = T.problem('dev',20270923,0,'dependent')
    def predict(visible):
        p = T.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8)
        p = [min(1-1e-12,max(1e-12,v)) for v in p]
        return [[math.log1p(-v),math.log(v)] for v in p]
    row = dict(family='dependent',index=0,requested_context_tokens=1024,context_tokens=1024,
               position='far',records=70,exact=audit(ex,predict),
               costs=[dict(method=k,seconds=[scale*len(groups)]*3) for k,groups in partitions(ex).items()])
    return dict(execution_complete=True,development_only=True,comparator=kind,checkpoint_sha256=kind,
                competence_gate_passed=True,conditions=[row])


def test_joint_certificate_requires_both_error_and_measured_cost():
    main=report('rwkv',1.);attention=report('attention',3.)
    row=summarize([main,attention])['comparisons'][0]
    assert row['pair_preserving_minus_information_set_kl']==pytest.approx(4*math.log(2),abs=1e-10)
    assert row['attention']['joint_kl_certificate_and_budget_band']
    assert row['attention']['mean_timing_budget_lower_inclusive_seconds']==3.
    assert row['attention']['mean_timing_budget_upper_exclusive_seconds']==6.
    attention=report('attention',.5)
    row=summarize([main,attention])['comparisons'][0]
    assert row['product_kl_floor_certificate_margin']>0
    assert not row['attention']['joint_kl_certificate_and_budget_band']


def test_missing_comparison_remains_missing():
    row=summarize([report('rwkv',1.)])['comparisons'][0]
    assert row['attention']==dict(status='not_measured_for_this_condition')
    assert row['causal_rwkv']==dict(status='not_measured_for_this_condition')


def test_duplicate_checkpoint_and_unpaired_example_refused():
    main=report('rwkv',1.)
    with pytest.raises(ValueError):summarize([main,copy.deepcopy(main)])
    other=report('attention',3.)
    other['conditions'][0]['exact']['instance_id']='unpaired'
    with pytest.raises(ValueError):summarize([main,other])


def test_diagnostic_milestone_cannot_replace_terminal_checkpoint():
    main=report('rwkv',1.);main['diagnostic_step200']=True
    with pytest.raises(ValueError):summarize([main])
