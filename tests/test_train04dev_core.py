import itertools
from types import SimpleNamespace
import torch
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04 import tasks


def test_curriculum_ends_on_original_task():
    assert C.curriculum_n(300)==2 and C.curriculum_n(301)==4 and C.curriculum_n(701)==8
    assert C.example('dev',17,5,8)==tasks.make_example('dev',17,5)
    for n in (2,4):
        e=C.example('train',17,0,n);y=sum(b<<i for i,b in enumerate(e['bits']))
        assert all((y&r).bit_count()%2==v for r,v in zip(e['matrix_rows'],e['syndrome']))


def test_logodds_preserves_softmax_and_gradients():
    weight=torch.randn(8,5,requires_grad=True);x=torch.randn(3,5,requires_grad=True)
    old=SimpleNamespace(weight=torch.nn.Parameter(weight.detach().clone()),binary_ids=torch.tensor([2,6]))
    head=C.StableBinaryHead(old);ref=torch.nn.functional.linear(x,head.weight)[:,[2,6]]
    new=head(x)
    assert torch.allclose(ref.softmax(-1),new.softmax(-1),atol=2e-7)
    a=torch.autograd.grad(ref.log_softmax(-1)[:,1].sum(),(x,head.weight),retain_graph=True)
    b=torch.autograd.grad(new.log_softmax(-1)[:,1].sum(),(x,head.weight))
    assert all(torch.allclose(u,v,atol=1e-6) for u,v in zip(a,b))


def test_rao_blackwell_gradient_equals_conditional_hard_average():
    e=tasks.make_example('train',17,1);n=8
    visible={0:e['bits'][0],3:e['bits'][3]};masked=torch.tensor([[i not in visible for i in range(n)]])
    p=tasks.oracle_marginals(e['matrix_rows'],e['syndrome'],visible,n)
    endpoints=[]
    for bits in itertools.product((0,1),repeat=n):
        y=sum(b<<i for i,b in enumerate(bits))
        if all(bits[i]==v for i,v in visible.items()) and all((y&r).bit_count()%2==v for r,v in zip(e['matrix_rows'],e['syndrome'])):endpoints.append(bits)
    logits=torch.randn(1,n,2,requires_grad=True);batch=dict(oracle=torch.tensor([p]),masked=masked,stage=torch.tensor([4.]))
    gradients=[]
    for bits in endpoints:
        batch['gold']=torch.tensor([bits]);gradients.append(torch.autograd.grad(C.loss(logits,batch,'hard'),logits,retain_graph=True)[0])
    rb=torch.autograd.grad(C.loss(logits,batch,'rao_blackwell'),logits)[0]
    assert torch.allclose(rb,torch.stack(gradients).mean(0),atol=1e-7)
    assert torch.stack(gradients).var(0,unbiased=False).sum()>0


def test_pack_preserves_document_boundaries_and_empty_masks():
    tok=SimpleNamespace(encode=lambda text:[7,8,9],mask_id=99,binary_ids=(1,2))
    examples=[C.make_canvas(C.example('train',17,i,4),tok,17,stage=8) for i in range(4)]
    p=C.pack(examples,'cpu')
    assert p['doc_starts'].tolist()==[[0,7,14,21]]
    assert p['gather_idx'].tolist()==[3,4,5,6,10,11,12,13,17,18,19,20,24,25,26,27]
    assert p['codes'][0,:3].tolist()==[1,1,1]
    p['masked'].zero_();z=torch.randn(4,4,2,requires_grad=True)
    value=C.loss(z,p,'rao_blackwell');value.backward()
    assert value==0 and not z.grad.any()


def test_serial_resets_boundaries_and_preserves_gradient():
    class Backbone(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,input_ids,doc_starts,gather_idx,**kwargs):
            assert doc_starts.tolist()==[[0]]
            x=input_ids.float().cumsum(-1)[0,gather_idx]*self.scale
            return torch.stack((x,-x),-1)
    model=torch.nn.Module();model.backbone=Backbone()
    tok=SimpleNamespace(encode=lambda text:[7,8,9],mask_id=99,binary_ids=(1,2))
    canvases=[C.make_canvas(C.example('train',17,i,4),tok,17,stage=8) for i in range(4)]
    net=C.SerialDenoiser(model);out=net(C.pack(canvases,'cpu'))
    separate=torch.cat([net(C.pack([c],'cpu')) for c in canvases])
    assert torch.equal(out,separate)
    out[...,0].sum().backward()
    assert model.backbone.scale.grad>0
