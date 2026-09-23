import math
import pytest
from lrwkv_evidence.long_context_mvp import tasks as T
from lrwkv_evidence.long_context_eval.exact import audit


@pytest.mark.parametrize('epsilon',[1e-3,1e-9])
def test_perfect_two_round_predictions_do_not_bound_uncovered_sequential_error(epsilon):
    ex=T.problem('dev',20270923,999,'dependent')
    def predict(visible):
        probabilities=T.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8)
        probabilities=[min(1-1e-12,max(1e-12,p)) for p in probabilities]
        if set(visible)=={0}:probabilities[1]=epsilon
        return [[math.log1p(-p),math.log(p)] for p in probabilities]
    results={r['method']:r for r in audit(ex,predict)['results']}
    assert results['information_set']['joint_kl_nats']==pytest.approx(0.,abs=1e-10)
    predicted=-math.log(2)-.5*math.log(epsilon*(1-epsilon))
    assert results['sequential']['joint_kl_nats']==pytest.approx(predicted,abs=1e-10)
    assert results['sequential']['dependence_nats']==pytest.approx(0.,abs=1e-10)
