import math
import pytest
from lrwkv_evidence.long_context_mvp import tasks as T
from lrwkv_evidence.long_context_eval.exact import audit


def test_fair_model_is_policy_invariant():
    for family in T.FAMILIES:
        ex=T.problem('dev',17,2,family)
        result=audit(ex,lambda v:[[-math.log(2)]*2 for _ in range(8)])
        expected=math.log(256/len(ex['support']))
        for row in result['results']:
            assert row['joint_kl_nats']==pytest.approx(expected)
            assert row['exact_valid_mass']==pytest.approx(len(ex['support'])/256)
            assert row['within_valid_kl_nats']==pytest.approx(0,abs=1e-12)


def test_near_oracle_restores_information_set_dependence():
    ex=T.problem('dev',17,2,'dependent')
    def predict(visible):
        possible=[y for y in ex['support'] if all(((y>>i)&1)==b for i,b in visible.items())]
        result=[]
        for i in range(8):
            p=sum((y>>i)&1 for y in possible)/len(possible)
            p=min(1-1e-12,max(1e-12,p));result.append([math.log1p(-p),math.log(p)])
        return result
    rows={r['method']:r for r in audit(ex,predict)['results']}
    assert rows['one']['dependence_nats']==pytest.approx(4*math.log(2))
    assert rows['one']['exact_valid_mass']==pytest.approx(1/16)
    assert rows['pair_preserving_halves']['dependence_nats']==pytest.approx(4*math.log(2))
    assert rows['pair_preserving_halves']['exact_valid_mass']==pytest.approx(1/16,abs=1e-10)
    for name in ('information_set','sequential'):
        assert rows[name]['joint_kl_nats']==pytest.approx(0,abs=1e-10)
        assert rows[name]['exact_valid_mass']==pytest.approx(1,abs=1e-10)
        assert rows[name]['dependence_nats']==pytest.approx(0,abs=1e-12)


def test_invalid_prediction_refused():
    ex=T.problem('dev',17,2,'dependent')
    with pytest.raises(ValueError):audit(ex,lambda v:[[0.,0.]]*8)
