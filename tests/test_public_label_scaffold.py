import copy
from types import SimpleNamespace
import pytest
import torch
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.train04binding import core as B

TOK=SimpleNamespace(encode=lambda text:[100+ord(x) for x in text],mask_id=99,binary_ids=(1,2))

@pytest.mark.parametrize('n',[2,4,8])
def test_scaffold_preserves_values_masks_oracles_and_clamps_public_labels(n):
    ex=C.example('train',17,1,n)
    original=M.canvas(ex,TOK,17,branch='policy',round_index=1)
    decorated=B.decorate(ex,original,TOK);batch=B.pack([decorated],'cpu')
    positions=decorated['target_positions']
    assert [decorated['ids'][i] for i in positions]==original['ids'][original['prefix']:]
    for i,(code,time) in enumerate(zip(decorated['position_codes'],decorated['position_times'])):
        if i not in positions:assert code==1 and time==0
    for field in ['gold','oracle','masked','loss_mask','loss_scale','branch']:assert decorated[field]==original[field]
    # Alter all hidden supervision fields: serialized inputs must not change.
    changed=copy.deepcopy(original);changed['gold']=[1-b for b in changed['gold']];changed['oracle']=[.123]*n
    other=B.decorate({**ex,'bits':[1-b for b in ex['bits']]},changed,TOK)
    assert other['ids']==decorated['ids']
    assert batch['gather_idx'].tolist()==positions


def test_scaffold_serial_wrapper_handles_noncontiguous_targets_and_gradients():
    class Backbone(torch.nn.Module):
        def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.))
        def forward(self,input_ids,doc_starts,gather_idx,**kwargs):
            assert doc_starts.tolist()==[[0]]
            values=input_ids.float().cumsum(-1)[0,gather_idx]*self.scale
            return torch.stack((values,-values),-1)
    model=torch.nn.Module();model.backbone=Backbone();net=C.SerialDenoiser(model)
    canvases=[]
    for i in range(2):
        ex=C.example('train',17,i,8);canvases.append(B.decorate(ex,C.make_canvas(ex,TOK,17,stage=8),TOK))
    together=net(B.pack(canvases,'cpu'));separate=torch.cat([net(B.pack([c],'cpu')) for c in canvases])
    assert torch.equal(together,separate)
    together[...,0].sum().backward();assert model.backbone.scale.grad>0
