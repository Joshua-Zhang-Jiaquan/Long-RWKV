import math
import pytest
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04 import tasks
from lrwkv_evidence.train04dev.evaluate import endpoint_metrics
from lrwkv_evidence.train04dev.history_response import history_response

@pytest.mark.parametrize('n',[2,4,8])
@pytest.mark.parametrize('family_index',[0,1])
def test_oracle_and_constant_interventions(n,family_index):
    ex=C.example('dev',51917,family_index,n)
    for oracle in [False,True]:
        def predict(ex,visible,stage):
            probabilities=tasks.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,n) if oracle else [.3]*n
            return [[math.log(x) if x else -math.inf for x in (1-p,p)] for p in probabilities]
        cache={};endpoint_metrics(ex,'information_set',predict,cache)
        count=len(cache);r=history_response(ex,cache);assert len(cache)==count
        if ex['family']=='paired_parity':
            assert r['paired_signed_response']==pytest.approx(1. if oracle else 0.)
            if n>2:assert r['off_pair_absolute_response']==pytest.approx(0.)
        else:assert r['systematic_absolute_response']==pytest.approx(0.)
        assert r['additional_model_calls']==0

def test_missing_cache_cannot_silently_request_more_inference():
    with pytest.raises(KeyError):history_response(C.example('dev',51917,1,8),{})
