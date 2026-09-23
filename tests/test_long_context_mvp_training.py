import torch
from lrwkv_evidence.long_context_mvp import training as R

class Tokenizer:
    binary_ids=[1001,1002];mask_id=1003
    def encode(self,text):return list(text.encode())


def test_rank_coverage_and_curriculum():
    tok=Tokenizer();ids=set()
    for rank in range(8):
        records=R.records(201,rank,tok)
        assert [c['family'] for c in records]==['retrieval','dependent']*2
        assert all(len(c['ids'])==1032 for c in records)
        assert all(c['instance_id'] not in ids for c in records)
        ids.update(c['instance_id'] for c in records)
    assert len(ids)==32
    assert all(len(c['ids'])<1032 for c in R.records(200,0,tok))


def test_policy_loss_supervises_only_current_reveal_group():
    records=R.records(201,0,Tokenizer());policy=records[2:]
    for c in policy:
        selected=[i for i,v in enumerate(c['loss_mask']) if v]
        assert selected in (list(range(4)),list(range(4,8)))
        assert all(c['masked'][i] for i in selected)
        assert c['loss_scale']==.25
    b=R.batch(policy,'cpu');logits=torch.zeros(2,8,2,requires_grad=True)
    loss=R.loss(logits,b);loss.backward()
    assert torch.isclose(loss,torch.tensor(2.).log())
    assert (logits.grad[~b['loss_mask']]==0).all()


def test_empty_supervision_has_zero_loss_and_zero_gradient():
    records=R.records(1,0,Tokenizer())
    for c in records:c['loss_mask']=[False]*8
    b=R.batch(records,'cpu');logits=torch.randn(4,8,2,requires_grad=True)
    loss=R.loss(logits,b);loss.backward()
    assert loss.detach().item()==0 and (logits.grad==0).all()
