import math
from collections import Counter
import pytest
from lrwkv_evidence.distance_intervention import training as R, tasks as T
from lrwkv_evidence.train04 import tasks as O
from lrwkv_evidence.posterior_context_probe.exact import audit


class Tokens:
    binary_ids=(48,49)
    mask_id=999
    def encode(self, text): return [ord(c) for c in text]


@pytest.mark.parametrize('step,rank', [(1,0),(2,3),(3,7)])
def test_matched_arms_and_correct_conditional_supervision(step, rank):
    tok=Tokens(); a=R.records(step,rank,tok,'near',R.SEEDS[0]); b=R.records(step,rank,tok,'balanced',R.SEEDS[0])
    assert len(a)==len(b)==4
    for x,y in zip(a,b):
        for key in ('gold','oracle','masked','stage','loss_mask','loss_scale','policy','instance_id','round_index'):
            assert x[key]==y[key]
        assert len(x['ids'])==len(y['ids'])
        assert Counter(x['ids'][:x['prefix']])==Counter(y['ids'][:y['prefix']])
        assert x['ids'][x['prefix']:]==y['ids'][y['prefix']:]
        assert sum(x['loss_mask'])==4
        ex=O.make_example('train',R.SEEDS[0],R.RECIPE['index_offset']+(step-1)*16+rank*2+(0 if x['family']=='systematic' else 1))
        visible={i:x['gold'][i] for i,m in enumerate(x['masked']) if not m}
        assert x['oracle']==O.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8)
    assert Counter(x['policy'] for x in a)=={'information_set':2,'pair_preserving_halves':2}


def test_counterfactual_changes_one_target_and_one_token():
    tok=Tokens()
    for index in range(32):
        ex=T.example(index); cf,meta=T.counterfactual(ex)
        assert set(ex['support']).isdisjoint(cf['support'])
        for position in ('far','middle','near'):
            a=T.serialize(ex,tok,16384,position); b=T.serialize(cf,tok,16384,position)
            assert T.validate_layout(a,b)>=a['public_block_span'][0]
        assert meta['original_target'] != meta['flipped_target']


def test_oracle_endpoints_and_exact_decomposition():
    ex=T.example(0)
    def predict(visible):
        ps=O.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8)
        return [[math.log(max(1-p,1e-100)),math.log(max(p,1e-100))] for p in ps]
    rows={r['method']:r for r in audit(ex,predict)['results']}
    assert rows['information_set']['joint_kl_nats']==pytest.approx(0,abs=1e-10)
    assert rows['information_set']['exact_valid_mass']==pytest.approx(1)
    for name in ('one','pair_preserving_halves'):
        assert rows[name]['joint_kl_nats']==pytest.approx(4*math.log(2))
        assert rows[name]['estimation_nats']==pytest.approx(0,abs=1e-10)
        assert rows[name]['exact_valid_mass']==pytest.approx(1/16)


def test_binary_response_floor_distinguishes_centering():
    from lrwkv_evidence.distance_intervention.diagnostics import evidence_response, softplus
    meta=dict(target_coordinate=0,original_target=0,flipped_target=1,unaffected_targets=[1,2,3])
    def lp(x):return [-softplus(x),-softplus(-x)]
    centered=evidence_response([lp(-2)]+[lp(0)]*7,[lp(2)]+[lp(0)]*7,meta)
    biased=evidence_response([lp(3)]+[lp(0)]*7,[lp(7)]+[lp(0)]*7,meta)
    assert centered['signed_logit_contrast']==pytest.approx(4)
    assert centered['centering_penalty']==pytest.approx(0)
    assert biased['signed_logit_contrast']==pytest.approx(4)
    assert biased['centering_penalty']>1
    assert centered['both_variants_correct'] and not biased['both_variants_correct']
    assert biased['unaffected_max_abs_probability_change']==0
