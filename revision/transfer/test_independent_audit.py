"""Checks the independent endpoint audit against fully enumerable fixtures."""
import copy,math
import numpy as np
import pytest
from lrwkv_evidence.predictive_transfer import tasks as T,prediction as P
from tools.verify_transfer_evidence import endpoint,parse_public,features,fit_source,choice_from_source
from revision.transfer.test_prediction import fake_rows


def test_complete_joint_mass_and_validity():
    ex=P.example(P.panel('heldout')[0]);ys=T.support(ex)
    initial=[[-math.log(2)]*2 for _ in range(8)]
    conditional=[[[[math.log(.9 if b==y[i] else .1) for b in (0,1)] for i in range(8)] for y in ys] for _ in (0,1)]
    measured,public=endpoint(ex['prompt'],initial,conditional)
    assert public[2]==[sum(b<<i for i,b in enumerate(y)) for y in ys]
    for m in measured:
        assert abs(m['kl']+4*math.log(.9))<1e-12
        assert abs(m['valid_mass']-.9**4)<1e-12
    malformed=copy.deepcopy(conditional);malformed[0][0][0][0]+=1.
    with pytest.raises(ValueError,match='normalization'):endpoint(ex['prompt'],initial,malformed)
    # Normalized but wrong history indexing must alter exact posterior fidelity.
    permuted=copy.deepcopy(conditional);permuted[0]=list(reversed(permuted[0]))
    changed,_=endpoint(ex['prompt'],initial,permuted)
    assert changed[0]['kl']>measured[0]['kl']


def test_public_feature_and_forecast_reconstruction():
    source=fake_rows();records=[]
    for row in source:
        ex=P.example(row['cell']);public=parse_public(ex['prompt'])
        records.append(dict(**row,instance_id=ex['instance_id'],prompt=ex['prompt'],public=public))
        for policy in (0,1):assert features(public[0],public[1],policy,row['cell']['position'])==T.features(ex,policy,row['cell']['position'])
    independently_fitted=fit_source(records);frozen_fitted=P.fit(source)
    for cell in P.panel('heldout'):
        ex=P.example(cell);public=parse_public(ex['prompt'])
        own=choice_from_source(independently_fitted,dict(cell=cell,prompt=ex['prompt'],public=public))
        expected=P.predict(frozen_fitted,cell)
        assert own=={k:v for k,v in expected.items() if k!='cell'}


def test_unsupported_fallback_retains_the_cell():
    source=fake_rows();fitted=P.fit(source);cell=P.panel('heldout')[0];ex=P.example(cell)
    feature=T.features(ex,0,cell['position'])[0][1];fitted['bins'].pop(P.key(feature))
    predicted=P.predict(fitted,cell)
    assert not predicted['supported'] and predicted['chosen']==predicted['dependence_only']
    assert P.key(feature) in predicted['unsupported_bins']
