import math
import pytest
from lrwkv_evidence.long_context_eval.numerical import specifications,validate


@pytest.mark.parametrize('kind', ['rwkv','attention','causal_rwkv'])
def test_actual_lengths_positions_and_history_modes_required(kind):
    rows=[dict(s,within_repeat_max_logprob_diff=0.,cross_rank_max_logprob_diff=1e-6) for s in specifications(kind)]
    validate(rows,kind)
    assert len(rows)==(6 if kind=='causal_rwkv' else 12)
    with pytest.raises(ValueError):validate(rows[:-1],kind)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,.01])
def test_invalid_or_excessive_numerical_difference_rejected(value):
    rows=[dict(s,within_repeat_max_logprob_diff=0.,cross_rank_max_logprob_diff=0.) for s in specifications()]
    rows[-1]['cross_rank_max_logprob_diff']=value
    with pytest.raises(ValueError):validate(rows)
