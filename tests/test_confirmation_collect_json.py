import copy
import json
import math
import pytest
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04dev.evaluate import endpoint_metrics
from theory_mvp.train04confirm.collect import METHODS,summarize


def evidence():
    # Development inputs only: never evaluate even a dummy predictor on the
    # locked confirmation panel while preparing its evaluation code.
    panel=[];rows=[]
    for index in (0,1):
        ex=C.example('dev',51917,index,8)
        record=dict(index=index,family=ex['family'],structure=sorted(ex['matrix_rows']),instance_id=ex['instance_id'])
        panel.append((record,ex))
        for method in METHODS:
            metric=endpoint_metrics(ex,method,lambda e,v,s:[[-math.log(2)]*2 for _ in range(8)])
            rows.append(dict(n=8,**record,**metric))
    return json.loads(json.dumps(rows)),panel


def test_fixed_product_baseline_has_no_schedule_benefit():
    rows,panel=evidence();result=summarize(rows,panel)
    assert len(result['family_summary'])==8
    for row in result['family_summary']:
        assert row['exact_valid_mass']==pytest.approx(1/16)
        assert row['joint_kl_nats']==pytest.approx(math.log(16))
    for row in result['structure_contrasts']:
        assert row['random_minus_information_set_kl']==pytest.approx(0)


@pytest.mark.parametrize('mutation',['missing','duplicate','wrong_support','wrong_identity'])
def test_rejects_incomplete_or_mislabeled_endpoint_laws(mutation):
    rows,panel=evidence();rows=copy.deepcopy(rows)
    if mutation=='missing':rows.pop()
    if mutation=='duplicate':rows.append(rows[0])
    if mutation=='wrong_identity':rows[0]['instance_id']='wrong'
    if mutation=='wrong_support':
        # Uniform endpoint metrics remain algebraically consistent, but this
        # permuted support order is not the frozen public condition's support.
        rows[0]['valid_endpoints']=list(reversed(rows[0]['valid_endpoints']))
    with pytest.raises(ValueError):summarize(rows,panel)
