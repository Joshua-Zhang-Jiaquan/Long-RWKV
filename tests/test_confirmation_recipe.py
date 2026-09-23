from types import SimpleNamespace
import pytest
import torch
from lrwkv_evidence.train04confirm import recipe as R
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.train04paired import core as P

TOK=SimpleNamespace(encode=lambda s:list(s.encode()),binary_ids=(256,257),mask_id=258)


@pytest.mark.parametrize('step',[1,300,301,700,701,1500,1501,2500])
def test_seed17_reproduces_development_data_and_loss_at_transitions(step):
    rank=3;mode='complementary'
    if step>1500:
        expected=P.training_canvases(step-1500,rank,TOK,mode)
    else:
        n=C.curriculum_n(step)
        examples=[C.example('train',17,(step-1)*32+rank*4+j,n) for j in range(4)]
        expected=[C.make_canvas(ex,TOK,17) for ex in examples] if step<=300 else [B.decorate(ex,M.canvas(ex,TOK,17),TOK) for ex in examples]
    assert R.canvases(step,rank,TOK,17,mode)==expected
    batch=R.batch(step,rank,TOK,17,mode,'cpu')
    logits=torch.linspace(-2,2,2*batch['gold'].numel()).reshape(*batch['gold'].shape,2).requires_grad_()
    actual=R.loss(step,logits,batch)
    reference=(C.loss if step<=300 else M.loss)(logits,batch,'rao_blackwell')
    assert torch.equal(actual,reference)
    assert torch.equal(torch.autograd.grad(actual,logits)[0],torch.autograd.grad(reference,logits)[0])


def test_fresh_seeds_change_data_but_preserve_matching_between_arms():
    ids=[]
    for seed in R.SEEDS:
        a=R.canvases(300,0,TOK,seed,'independent')
        assert a==R.canvases(300,0,TOK,seed,'complementary')
        ids.extend(c['instance_id'] for c in a)
        for step in (301,1500):
            assert R.canvases(step,0,TOK,seed,'independent')==R.canvases(step,0,TOK,seed,'complementary')
        a=R.canvases(1501,0,TOK,seed,'independent');b=R.canvases(1501,0,TOK,seed,'complementary')
        assert a[0]==b[0] and a[2]==b[2]
        for x,y in zip(a,b):
            for key in ('instance_id','stage','loss_mask','loss_scale','target_positions'):
                assert x[key]==y[key]
            assert len(x['ids'])==len(y['ids'])
    assert len(set(ids))==len(ids)
