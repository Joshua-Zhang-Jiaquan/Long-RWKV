"""The exploratory prompt set is fixed; gold references never fill targets."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.e3 import positive_canary as P


class CanaryTests(unittest.TestCase):
    def test_fixed_prompt_set_and_balanced_assignment(self):
        self.assertEqual(len(P.PROMPTS), 8)
        self.assertEqual(len({p[0] for p in P.PROMPTS}), 8)
        self.assertEqual([sum(i % 4 == r for i in range(8)) for r in range(4)], [2] * 4)
        self.assertEqual(P.PROMPTS, tuple(P.PROMPTS))
        self.assertEqual(P.BUDGET, 32)
        self.assertEqual(P.CONFIG['steps'], 8)

    def test_canvas_masked_independent_of_reference(self):
        encoder = SimpleNamespace(encode_text=lambda text: [ord(c) for c in text])
        for name, prompt, reference in P.PROMPTS:
            ids, (lo, hi) = P.make_canvas(encoder, prompt, 999)
            self.assertEqual(ids[:lo], encoder.encode_text(prompt))
            self.assertEqual(ids[lo:hi], [999] * 32)
            self.assertEqual(hi - lo, 32)
            self.assertNotEqual(ids[lo:hi], encoder.encode_text(reference))

    def test_causal_first_step_receives_only_visible_prefix(self):
        class Model:
            seen = None
            def __call__(self, *, input_ids, use_cache):
                self.seen = input_ids.tolist()
                return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 12))
        model = Model()
        ids = torch.tensor([[2, 3, 11, 11]])
        encoder = SimpleNamespace(decode_sample_ids=lambda ids: str(ids))
        result = P.first_step('r0', model, ids, (2, 4), list(range(10)), [2], encoder)
        self.assertEqual(model.seen, [[2, 3]])
        self.assertEqual(result['position'], 'last_visible_prefix')
        self.assertEqual(len(result['top10']), 10)
        self.assertEqual(result['expected_first_token']['rank'], 1)


if __name__ == '__main__':
    unittest.main()
