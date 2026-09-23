import importlib.util
from pathlib import Path
import sys
import pytest

folder=Path(__file__).resolve().parents[1]/'theory_mvp/distance_intervention';sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location('distance_cost_certificate',folder/'cost_certificate.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
METHODS=('one','information_set','pair_preserving_halves')


def fixture(kl=1.,latency=1.3):
    attention=dict(summary=[dict(method=m,mean_request_seconds=1. if m=='one' else 2.) for m in METHODS],
                   conditions=[dict(index=i) for i in range(8)])
    recurrent=dict(role='fixture',checkpoint_sha256='test',conditions=[],timings=[])
    for i in range(32):
        for pos in ('far','middle','near'):
            recurrent['conditions'].append(dict(index=i,primary=True,serial=dict(position=pos),
                exact=dict(results=[dict(method=m,joint_kl_nats=kl if i<8 else 0.) for m in METHODS])))
            if i<8:
                for m in METHODS:recurrent['timings'].append(dict(index=i,position=pos,method=m,seconds=[latency]*5))
    return attention,recurrent


def test_certificate_requires_both_quality_and_all_cost_conditions():
    a,r=fixture();assert M.certificate(a,r)['empirical_sufficient_certificate']
    a,r=fixture(kl=3.);assert not M.certificate(a,r)['empirical_sufficient_certificate']
    a,r=fixture(latency=1.6);assert not M.certificate(a,r)['empirical_sufficient_certificate']
    a,r=fixture();a['summary'][1]['mean_request_seconds']=1.4
    assert not M.certificate(a,r)['empirical_sufficient_certificate']


def test_quality_uses_only_exactly_matched_eight_problem_law():
    a,r=fixture(kl=3.)
    result=M.certificate(a,r)
    assert result['quality']['information_set']['mean']==pytest.approx(3.)
    assert not result['empirical_sufficient_certificate']
    a['conditions'][-1]['index']=20
    with pytest.raises(ValueError,match='matched attention'):M.certificate(a,r)
