import itertools
from types import SimpleNamespace
import numpy as np
from lrwkv_evidence.train04 import tasks
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04paired import core as P

TOK=SimpleNamespace(encode=lambda s:list(s.encode()),binary_ids=(256,257),mask_id=258)


def test_complement_is_feasible_involution_and_preserves_public_condition():
    for n in (2,4,8):
        for index in range(12):
            ex=C.example('train',17,index,n);free=ex['information_set']
            for values in itertools.product((0,1),repeat=len(free)):
                a=P.with_free_values(ex,values);b=P.with_free_values(ex,[1-v for v in values])
                assert a['prompt']==b['prompt']==ex['prompt']
                assert a['matrix_rows']==b['matrix_rows'] and a['syndrome']==b['syndrome']
                y=sum(v<<i for i,v in enumerate(b['bits']))
                assert all((y&r).bit_count()%2==c for r,c in zip(ex['matrix_rows'],ex['syndrome']))
                assert [a['bits'][i] for i in free]==list(values)
                if ex['family']=='paired_parity':assert b['bits']==[1-v for v in a['bits']]
                else:assert all(a['bits'][i]==b['bits'][i] for i in range(n) if i not in free)


def test_both_history_couplings_preserve_population_risk_and_gradient():
    # Arbitrary history-dependent scalar losses and gradients: each member has
    # exactly the uniform marginal, though their covariance need not decrease.
    rng=np.random.default_rng(57)
    for m in (1,2,4):
        count=2**m;values=rng.normal(size=(count,7))
        iid=np.mean([(values[a]+values[b])/2 for a in range(count) for b in range(count)],axis=0)
        anti=np.mean([(values[a]+values[a^(count-1)])/2 for a in range(count)],axis=0)
        assert np.allclose(iid,values.mean(0)) and np.allclose(anti,values.mean(0))


def test_batches_match_conditions_rounds_and_token_budget():
    for update in (1,2,19):
        for rank in range(8):
            a=P.training_canvases(update,rank,TOK,'independent')
            b=P.training_canvases(update,rank,TOK,'complementary')
            assert len(a)==len(b)==4
            assert [c['family'] for c in a]==['systematic']*2+['paired_parity']*2
            for x,y in zip(a,b):
                for field in ('prefix','stage','masked','loss_mask','loss_scale','target_positions','position_codes','position_times','instance_id'):
                    assert x[field]==y[field]
                assert len(x['ids'])==len(y['ids']) and x['loss_scale']==.25
                assert all(not v or x['masked'][i] for i,v in enumerate(x['loss_mask']))
            assert a[0]['instance_id']==a[1]['instance_id'] and a[2]['instance_id']==a[3]['instance_id']
            assert a[0]['ids']==b[0]['ids'] and a[2]['ids']==b[2]['ids']


def test_complementary_gradient_covariance_has_no_universal_ordering():
    signs=np.array(list(itertools.product((-1.,1.),repeat=2)))
    gradients=np.stack((signs[:,0]*signs[:,1],signs[:,0]),axis=1)
    partner=gradients[::-1]
    even=(gradients+partner)/2;odd=(gradients-partner)/2
    def covariance(x):
        centered=x-x.mean(0)
        return centered.T@centered/len(x)
    single=covariance(gradients)
    independent=np.array([(a+b)/2 for a in gradients for b in gradients])
    assert np.allclose(single,covariance(even)+covariance(odd))
    assert np.allclose(covariance(independent),single/2)
    difference=covariance(even)-covariance(independent)
    assert np.allclose(difference,(covariance(even)-covariance(odd))/2)
    assert np.allclose(np.linalg.eigvalsh(difference),[-.5,.5])
