from types import SimpleNamespace
import torch
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M

tok=SimpleNamespace(encode=lambda text:[7,8,9],binary_ids=(1,2),mask_id=99)

def test_policy_canvas_does_not_reveal_unscored_targets():
    for n in (2,4,8):
        ex=C.example('train',17,1,n)
        first=M.canvas(ex,tok,17,branch='policy',round_index=0)
        assert first['ids'][-n:]==[99]*n
        assert sum(first['loss_mask'])==n//2 and all(first['masked'])
        batch=M.pack([first],'cpu');assert batch['codes'][0,-n:].tolist()==[3]*n
        second=M.canvas(ex,tok,17,branch='policy',round_index=1)
        assert sum(second['masked'])==n//2
        assert second['masked']==second['loss_mask']
        assert set(second['oracle'])<={0.,1.}


def test_corruption_branch_preserves_original_law_and_gradient():
    ex=C.example('train',17,1,4);c=M.canvas(ex,tok,17,branch='corruption')
    original=C.make_canvas(ex,tok,17)
    assert all(c[k]==v for k,v in original.items())
    batch=M.pack([c],'cpu');logits=torch.randn(1,4,2,requires_grad=True)
    for objective in ('hard','rao_blackwell'):
        a=C.loss(logits,batch,objective);b=M.loss(logits,batch,objective)
        assert torch.allclose(a,b)
        x=torch.autograd.grad(a,logits,retain_graph=True)[0];y=torch.autograd.grad(b,logits,retain_graph=True)[0]
        assert torch.allclose(x,y)


def test_policy_round_weighting_and_gradient_support():
    ex=C.example('train',17,1,8)
    canvases=[M.canvas(ex,tok,17,branch='policy',round_index=k) for k in (0,1)]
    batch=M.pack(canvases,'cpu');logits=torch.randn(2,8,2,requires_grad=True)
    value=M.loss(logits,batch,'rao_blackwell');value.backward()
    assert not logits.grad[~batch['loss_mask']].any()
    assert torch.allclose(M.oracle_entropy(batch),torch.tensor(.5)*torch.log(torch.tensor(2.)))
    assert batch['loss_scale'].tolist()==[.25,.25]
