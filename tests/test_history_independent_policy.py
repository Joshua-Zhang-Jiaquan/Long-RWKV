"""A history-independent predictor cannot exploit a reveal schedule."""
import math
import pytest
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04dev.evaluate import endpoint_metrics

@pytest.mark.parametrize('n',[2,4,8])
@pytest.mark.parametrize('fair',[False,True])
def test_endpoint_law_invariant_to_partition_without_history(n,fair):
    ex=C.example('dev',51917,1,n)
    probabilities=[.5 if fair else .15+.7*(i+1)/(n+1) for i in range(n)]
    def predict(example,visible,stage):
        return [[math.log1p(-p),math.log(p)] for p in probabilities]
    rows=[endpoint_metrics(ex,method,predict) for method in ('one','information_set','random_halves')]
    for r in rows:
        assert r['endpoint_log_probabilities']==pytest.approx(rows[0]['endpoint_log_probabilities'],abs=1e-12)
        assert r['joint_kl_nats']==pytest.approx(rows[0]['joint_kl_nats'],abs=1e-12)
    if fair:
        for r in rows:
            assert r['joint_kl_nats']==pytest.approx(n/2*math.log(2),abs=1e-12)
            assert r['exact_valid_mass']==pytest.approx(2**(-n/2),abs=1e-12)
            assert r['estimation_error_nats']==pytest.approx(n/2*math.log(2)-r['dependence_cost_nats'],abs=1e-12)
