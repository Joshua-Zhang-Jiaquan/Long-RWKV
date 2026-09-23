import math
import pytest
from lrwkv_evidence.long_context_baseline_eval.exact import causal_audit
from lrwkv_evidence.long_context_mvp import tasks as T


def test_causal_fair_predictor_has_correct_exact_mass():
    ex = T.problem('dev', 20270923, 0, 'dependent')
    result = causal_audit(ex, lambda y: [[-math.log(2)]*2 for _ in range(8)])['results'][0]
    assert result['dependence_nats'] == 0
    assert result['joint_kl_nats'] == pytest.approx(4*math.log(2))
    assert result['exact_valid_mass'] == pytest.approx(1/16)
    assert result['within_valid_kl_nats'] == pytest.approx(0., abs=1e-12)


def test_causal_exact_conditionals_preserve_uniform_valid_law():
    ex = T.problem('dev', 20270923, 0, 'dependent')
    def path(y):
        rows = []
        for i in range(8):
            previous = {j: (y>>j)&1 for j in range(i)}
            p = T.oracle_marginals(ex['matrix_rows'], ex['syndrome'], previous, 8)[i]
            p = min(1-1e-12, max(1e-12, p))
            rows.append([math.log1p(-p), math.log(p)])
        return rows
    result = causal_audit(ex, path)['results'][0]
    assert result['joint_kl_nats'] == pytest.approx(0., abs=1e-10)
    assert result['exact_valid_mass'] == pytest.approx(1., abs=1e-10)
    assert result['within_valid_kl_nats'] == pytest.approx(0., abs=1e-10)


def test_causal_rejects_unnormalized_logits():
    ex = T.problem('dev', 20270923, 0, 'dependent')
    with pytest.raises(ValueError): causal_audit(ex, lambda y: [[0., 0.]]*8)
