import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'theory_mvp/distance_intervention/report.py'
spec=importlib.util.spec_from_file_location('distance_report',path)
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


def fake(shift):
    rows=[]
    for i in range(32):
        for p in R.POSITIONS:
            error=shift+(i/10 if p=='far' else i/20)
            rows.append(dict(index=i,primary=True,serial=dict(position=p),
                             exact=dict(results=[dict(method='information_set',estimation_nats=error)]),
                             stages=dict(second_round_estimation_nats=error),
                             counterfactual=dict(response=dict(mean_correct_conditional_ce=error,
                                                               signed_logit_contrast=-error,centering_penalty=error))))
    return dict(conditions=rows)


def test_pairing_keeps_identical_problem_noise_and_reports_seeds():
    data={}
    for seed,shift in zip(R.SEEDS,(1.,2.,3.)):
        data[f'near_seed{seed}']=fake(shift)
        data[f'balanced_seed{seed}']=fake(0.)
    out=R.contrasts(data)
    effect=out['seed_averaged_problem_effects']['far_conditional_error_reduction']
    assert effect['mean']==pytest.approx(2)
    assert effect['ci95']==pytest.approx([2,2])
    assert out['primary_far_direction_positive_all_seeds']
    assert set(out['per_adaptation_seed'])==set(map(str,R.SEEDS))
    assert out['seed_averaged_problem_effects']['far_minus_near_error_reduction']['mean']==pytest.approx(0,abs=1e-12)


def test_seed_reversal_cannot_be_hidden_by_positive_average():
    data={}
    for seed,shift in zip(R.SEEDS,(3.,3.,-1.)):
        data[f'near_seed{seed}']=fake(shift)
        data[f'balanced_seed{seed}']=fake(0.)
    out=R.contrasts(data)
    assert out['primary_far_error_reduction_ci_positive']
    assert not out['primary_far_direction_positive_all_seeds']


def test_estimator_rejects_accidentally_pooled_seed_problem_array():
    with pytest.raises(ValueError):R.estimate(np.zeros((3,32)))
