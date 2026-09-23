"""Check the implemented loss against exact endpoint error, with nontrivial histories."""
import math
import pytest
import torch
from lrwkv_evidence.distance_intervention import tasks as T, training as R
from lrwkv_evidence.train04 import tasks as O
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.posterior_context_probe.exact import audit


class Tokens:
    binary_ids=(48,49)
    mask_id=999
    def encode(self,text):return list(text.encode())


def test_balanced_excess_risk_equals_sum_of_policy_errors_over16():
    ex=T.example(3);tok=Tokens();canvases=[];logits=[]
    def model(visible):
        return [[0.,.11*(i-3)+.07*sum((j+1)*(2*v-1) for j,v in visible.items())] for i in range(8)]
    def lp(visible):return torch.tensor(model(visible),dtype=torch.float64).log_softmax(-1).tolist()
    exact={r['method']:r for r in audit(ex,lp)['results']}
    for policy in ('information_set','pair_preserving_halves'):
        groups=T.partitions(ex)[policy]
        for round_index in (0,1):
            for y in ex['support']:
                gold=[(y>>i)&1 for i in range(8)]
                visible={} if round_index==0 else {i:gold[i] for i in groups[0]}
                masked=[i not in visible for i in range(8)];prefix=tok.encode(ex['prompt'])
                c=dict(ids=prefix+[tok.mask_id if masked[i] else tok.binary_ids[visible[i]] for i in range(8)],
                       prefix=len(prefix),stage=8 if round_index==0 else 4,masked=masked,gold=gold,
                       oracle=O.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,8),
                       loss_mask=[i in groups[round_index] for i in range(8)],loss_scale=2/8,branch='policy')
                canvases.append(B.decorate(ex,c,tok));logits.append(model(visible))
    batch=B.pack(canvases,'cpu');floor=float(M.oracle_entropy(batch))
    risk=float(R.loss(torch.tensor(logits),batch))-floor
    info=exact['information_set'];pair=exact['pair_preserving_halves']
    assert floor==pytest.approx(.75*math.log(2),abs=1e-7)
    assert 16*risk==pytest.approx(info['estimation_nats']+pair['estimation_nats'],abs=2e-6)
    residual=pair['joint_kl_nats']-info['joint_kl_nats']-4*math.log(2)
    assert abs(residual)<=16*risk+2e-6
