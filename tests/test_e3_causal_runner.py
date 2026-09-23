"""CPU tests for AR shift, no target leakage, and qualification refusal."""
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.e3 import causal_runner as C


class ToyCausal:
    def __init__(self):
        self.seen = []
    def __call__(self, *, input_ids, use_cache):
        assert use_cache is False
        self.seen.append(input_ids.clone())
        logits = torch.zeros((*input_ids.shape, 7))
        logits[..., 0] = 100  # illegal token, must be suppressed
        logits[:, -1, int(input_ids[0, -1]) + 1] = 10
        return SimpleNamespace(logits=logits)


class CausalTests(unittest.TestCase):
    def test_next_logits_and_no_gold_leak(self):
        model = ToyCausal()
        ids = torch.tensor([[1, 2, 6, 6]])
        tokens, calls = C.greedy_suffix(model, ids, (2, 4), [1, 2, 3, 4, 5, 6])
        self.assertEqual(tokens, [3, 4])
        self.assertEqual(calls, 2)
        self.assertEqual([x.tolist() for x in model.seen], [[[1, 2]], [[1, 2, 3]]])
        self.assertEqual(ids.tolist(), [[1, 2, 6, 6]])

    def test_previous_full_logits_released_before_next_forward(self):
        class MemoryChecked(ToyCausal):
            previous = None
            def __call__(self, **kwargs):
                if self.previous is not None:
                    assert self.previous() is None, 'previous full logits still live'
                result = super().__call__(**kwargs)
                self.previous = weakref.ref(result.logits)
                return result
        tokens, calls = C.greedy_suffix(MemoryChecked(), torch.tensor([[1, 2, 6, 6]]),
                                        (2, 4), [1, 2, 3, 4, 5, 6])
        self.assertEqual(tokens, [3, 4])

    def test_non_suffix_refused(self):
        with self.assertRaises(ValueError):
            C.greedy_suffix(ToyCausal(), torch.tensor([[1, 2, 3, 4]]), (1, 3), [1])

    def test_token_map_mismatch_refused(self):
        encoder = SimpleNamespace(_tokenizer=SimpleNamespace(idx2token={1: b'a'}))
        wrapper = SimpleNamespace(trie_tokenizer=SimpleNamespace(idx2token={1: b'b'}))
        with self.assertRaises(ValueError):
            C.qualify_tokenizer(encoder, wrapper)

    def test_confirmation_qualified_same_model_policy_source(self):
        with self.assertRaises(ValueError):
            C.require_qualification({}, {'model_identity': 'model'})
        receipt = {'qualified': True, 'n': 72, 'policy': C.POLICY,
                   'model_identity': 'model', 'dependencies': {'source': 'digest'}, 'runner_sha256': C.file_sha(Path(C.__file__))}
        C.require_qualification(receipt, {'model_identity': 'model', 'dependencies': {'source': 'digest'}})
        with self.assertRaises(ValueError):
            C.require_qualification(receipt, {'model_identity': 'other model'})

    def test_wrapper_difference_disclosed_without_retokenizing(self):
        enc = SimpleNamespace(decode_ids=lambda ids: 'ab')
        native = SimpleNamespace(decode=lambda ids: ['ab'], encode=lambda text: [[1, 2]])
        wrapper = SimpleNamespace(encode=lambda *a, **kw: [99])
        result = C.verify_prefix([1, 2], enc, native, wrapper)
        self.assertTrue(result['native_roundtrip_exact'])
        self.assertFalse(result['hf_wrapper_roundtrip_exact'])
        native_wrong = SimpleNamespace(decode=lambda ids: ['wrong'], encode=native.encode)
        with self.assertRaises(ValueError):
            C.verify_prefix([1, 3], enc, native_wrong, wrapper)

    def test_noncanonical_filler_segmentation_accepted_without_retokenizing(self):
        class Encoder:
            def decode_ids(self, ids):
                return ''.join({1: 'x', 2: 'xx'}[i] for i in ids)
        class Native:
            def decode(self, rows):
                return [Encoder().decode_ids(ids) for ids in rows]
            def encode(self, text):
                return [[2] * (len(text) // 2)]
        class Wrapper:
            def encode(self, text, **kwargs):
                return [2] * (len(text) // 2)
        banked = [1, 1, 1, 1]
        check = C.verify_prefix(banked, Encoder(), Native(), Wrapper())
        self.assertTrue(check['native_decode_equal'])
        self.assertFalse(check['native_roundtrip_exact'])
        self.assertFalse(check['hf_wrapper_roundtrip_exact'])
        self.assertEqual(check['prefix_ids_sha256'], C.G.ids_sha256([1, 1, 1, 1]))
        self.assertEqual(banked, [1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()
