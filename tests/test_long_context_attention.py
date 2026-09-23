import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from lrwkv_evidence.long_context_baselines.models import AttentionDenoiser
from lrwkv_evidence.train04dev.core import pack


def model():
    torch.manual_seed(3)
    config = GPTNeoXConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                          intermediate_size=64, vocab_size=64, rotary_pct=.5,
                          attention_dropout=0., hidden_dropout=0.)
    return AttentionDenoiser(GPTNeoXForCausalLM(config), (2, 3))


def test_attention_is_bidirectional_and_gradients_reach_backbone():
    m = model().train()
    ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
    gather = torch.tensor([1])
    original = m.document(ids, gather)
    changed = ids.clone(); changed[0, -1] = 11
    assert (original - m.document(changed, gather)).abs().max() > 1e-6
    original.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.backbone.layers[0].attention.query_key_value.weight.grad.abs().sum() > 0


def test_serial_documents_do_not_leak():
    m = model().eval()
    rows = [dict(ids=[4, 5, 6, 7, 8], prefix=3, stage=8, masked=[True]*2,
                 gold=[0, 1], oracle=[.5, .5]) for _ in range(2)]
    batch = pack(rows, 'cpu')
    original = m(batch)
    poisoned = dict(batch); poisoned['input_ids'] = batch['input_ids'].clone()
    poisoned['input_ids'][0, :5] = 11
    changed = m(poisoned)
    assert (original[0] - changed[0]).abs().max() > 1e-6
    assert torch.equal(original[1], changed[1])
