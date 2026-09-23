import copy
import math
import pytest
from lrwkv_evidence.train04dev.collect_evaluation import validate_endpoint

def fair_row():
    return dict(n=2,endpoint_log_probabilities=[-math.log(4)]*2,valid_endpoints=[[0,0],[1,1]],joint_kl_nats=math.log(2),exact_valid_mass=.5,dependence_cost_nats=0.,estimation_error_nats=math.log(2),conditional_valid_kl_nats=0.,decomposition_residual=0.)

def test_valid_uniform_law_passes():
    validate_endpoint(fair_row())

@pytest.mark.parametrize('change',[
    {'conditional_valid_kl_nats':.1},
    {'endpoint_log_probabilities':[float('nan'),-math.log(4)]},
    {'valid_endpoints':[[0,0],[0,0]]},
    {'estimation_error_nats':0.},
    {'endpoint_log_probabilities':[-math.log(4)]},
])
def test_corrupted_probability_evidence_rejected(change):
    row=copy.deepcopy(fair_row());row.update(change)
    with pytest.raises(ValueError):validate_endpoint(row)
