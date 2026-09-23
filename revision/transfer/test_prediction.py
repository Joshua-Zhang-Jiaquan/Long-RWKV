import copy,math
import pytest
from lrwkv_evidence.predictive_transfer import tasks as T
from lrwkv_evidence.predictive_transfer import prediction as P


def fake_rows():
    rows=[]
    for cell in P.panel('calibration'):
        ex=P.example(cell)
        metrics=[]
        for policy in (0,1):
            errors={str(i): .01 if f[0]=='fair' else .1*f[1]+.02*f[2] for i,f in T.features(ex,policy,cell['position'])}
            metrics.append(dict(errors=errors,kl=sum(errors.values())))
        rows.append(dict(cell=cell,metrics=metrics))
    return rows


def test_exact_endpoint_and_prediction_contract():
    ex=P.example(P.panel('heldout')[0]);ys=T.support(ex)
    init=[[-math.log(2)]*2 for _ in range(8)]
    # Smoothed oracle permits finite exhaustive probabilities and exact known KL.
    cond=[[[[math.log(.9 if b==y[i] else .1) for b in (0,1)] for i in range(8)] for y in ys] for _ in (0,1)]
    scores=P.metrics(ex,init,cond)
    for score in scores:
        assert abs(score['kl']+4*math.log(.9))<1e-12
        assert abs(score['valid_mass']-.9**4)<1e-12
    rows=fake_rows();fit=P.fit(rows)
    preds=[P.predict(fit,c) for c in P.panel('heldout')]
    assert len(preds)==240
    for pred in preds:
        if not pred['supported']:assert pred['chosen']==pred['dependence_only']
    with pytest.raises(ValueError):P.fit(rows[:-1])
    bad=copy.deepcopy(rows);bad[0]['cell']=P.panel('heldout')[0]
    with pytest.raises(ValueError):P.fit(bad)


def test_bases_are_public_and_gold_independent():
    for cell in P.panel('heldout'):
        ex=P.example(cell);changed=copy.deepcopy(ex)
        changed['bits']=[1-b for b in ex['bits']];changed['bases']=list(reversed(ex['bases']))
        for policy in (0,1):
            assert T.features(ex,policy,cell['position'])==T.features(changed,policy,cell['position'])
