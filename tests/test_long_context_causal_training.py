from lrwkv_evidence.long_context_baselines import training as B
from lrwkv_evidence.long_context_mvp import training as R


class Tokenizer:
    binary_ids = [1001, 1002]
    mask_id = 1003
    def encode(self, text): return list(text.encode())


def test_causal_supervision_conditions_only_on_past_bits():
    rows = B.records('causal_rwkv', 201, 0, Tokenizer())
    matched = R.records(201, 0, Tokenizer())
    for row, other in zip(rows, matched):
        assert row['instance_id'] == other['instance_id']
        assert row['ids'][:1024] == other['ids'][:1024]
        assert row['loss_mask'] == [True]*8 and row['loss_scale'] == 1/8
        assert row['ids'][1024:] == [Tokenizer.binary_ids[b] for b in row['gold']]
        if row['family'] == 'dependent':
            assert row['oracle'][:4] == [.5]*4
            assert row['oracle'][4:] == row['gold'][4:]
        else:
            assert row['oracle'] == row['gold']


def test_attention_uses_identical_corruption_recipe():
    assert B.records('attention', 1, 0, Tokenizer()) == R.records(1, 0, Tokenizer())
