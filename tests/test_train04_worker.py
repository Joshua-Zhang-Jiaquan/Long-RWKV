"""CPU qualification of the exact loss, clamped corruption, and binary adapter."""
import itertools
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.train04 import worker as W


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arm = 'A1'
        self.spec = SimpleNamespace(bidirectional=True, hidden_loop=False, gate_mod=False)
        self.embeddings = torch.nn.Embedding(10, 4)
        self.lm_head = torch.nn.Linear(4, 10, bias=False)
        self.kwargs = None
    def forward(self, ids, **kwargs):
        self.kwargs = kwargs
        return self.lm_head(self.embeddings(ids)[0].index_select(0, kwargs['gather_idx']))


class WorkerTests(unittest.TestCase):
    def test_exact_stage_importance_and_empty_masks(self):
        logits = torch.tensor([[0.2, -0.3], [0.4, 0.6]], requires_grad=True)
        labels = torch.tensor([0, 1])
        expected = torch.nn.functional.cross_entropy(logits, labels).detach().item()
        for stage in range(1, 9):
            probability = stage / 8
            expectation = 0
            for pattern in itertools.product((False, True), repeat=2):
                probability_mask = math.prod(probability if bit else 1 - probability for bit in pattern)
                value = W.exact_absorbing_loss(logits, labels, torch.tensor(pattern), stage)
                expectation += probability_mask * float(value.detach())
            self.assertAlmostEqual(expectation, expected, places=6)
        empty = W.exact_absorbing_loss(logits, labels, torch.tensor([False, False]), 1)
        self.assertEqual(float(empty.detach()), 0)
        empty.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_prefix_clamped_and_no_forced_mask(self):
        clean = torch.tensor([[8, 9, 3, 5]])
        corrupted, mask, stage = W.corrupt_targets(clean, (2, 4), 7,
            torch.Generator().manual_seed(3), stage=8)
        self.assertEqual(corrupted.tolist(), [[8, 9, 7, 7]])
        self.assertEqual(clean.tolist(), [[8, 9, 3, 5]])
        self.assertTrue(bool(mask.all()))
        empty_seen = False
        for seed in range(10):
            _, mask, _ = W.corrupt_targets(clean, (2, 4), 7,
                torch.Generator().manual_seed(seed), stage=1)
            empty_seen |= not bool(mask.any())
        self.assertTrue(empty_seen)

    def test_gather_same_positions_and_restricted_head_exact(self):
        backbone = ToyBackbone()
        original_weight = backbone.lm_head.weight
        hidden = torch.randn(2, 4)
        reference = torch.nn.functional.linear(hidden, original_weight)[:, [3, 5]]
        adapter = W.BinaryDenoiser(backbone, [3, 5], 7)
        self.assertIs(adapter.backbone.lm_head.weight, original_weight)
        self.assertTrue(torch.allclose(backbone.lm_head(hidden), reference, atol=1e-7))
        logits = adapter(torch.tensor([[8, 9, 7, 5]]), (2, 4), 4)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(backbone.kwargs['gather_idx'].tolist(), [2, 3])
        self.assertEqual(backbone.kwargs['codes'].tolist(), [[1, 1, 3, 2]])
        self.assertEqual(backbone.kwargs['block_t'].tolist(), [[0, 0, .5, .5]])
        self.assertFalse(backbone.kwargs['force_forward'])
        self.assertEqual(backbone.kwargs['R'], 1)

    def test_task_tokenization_preserves_explicit_bit_positions(self):
        tokenizer = SimpleNamespace(encode=lambda text: [8, 9], binary_ids=(3, 5), mask_id=7)
        result = W.tokenize_example({'prompt': 'independent public constraints', 'bits': [1, 0, 1]}, tokenizer)
        self.assertEqual(result['input_ids'], [8, 9, 5, 3, 5])
        self.assertEqual(result['target_span'], (2, 5))
        self.assertEqual(result['gold_bits'], [1, 0, 1])
        result['input_ids'][-1] = 3
        with self.assertRaises(ValueError):
            W.validate_example(result, tokenizer.binary_ids, tokenizer.mask_id)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
