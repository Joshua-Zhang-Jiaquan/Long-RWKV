"""Independent receipt audit rejects token/text and reconstructed input mismatches."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.mvp import audit_lookup as A


class AuditTests(unittest.TestCase):
    def fixture(self):
        item = A.P.make_items()[0]
        encoder = SimpleNamespace(encode_text=lambda text: list(text.encode()),
                                  decode_sample_ids=lambda ids: 'red' if ids == [1] * 32 else 'blue')
        provenance = {'fixture': 'synthetic test only'}
        canvas, span = A.P.canvas_for(encoder, item['prompt'], 65535)
        row = {'model': 'f2', 'rank': 0, 'scope': A.P.CONTRACT['scope'], 'status': 'ok',
            'item_id': item['item_id'], 'depth': 1, 'gold': 'red', 'prompt': item['prompt'],
            'prompt_sha256': A.P.U.digest(item['prompt']), 'input_ids_sha256': A.P.G.ids_sha256(canvas),
            'target_span': list(span), 'prompt_tokens': span[0], 'total_tokens': len(canvas),
            'padding_tokens': 0, 'decode_seed': 17, 'provenance_sha256': A.P.U.digest(provenance),
            'output_ids': [1] * 32, 'text': 'red', 'prediction': 'red', 'correct': True,
            'substring_correct': True, 'tokenizer_check': {'native_decode_equal': True,
                'prefix_ids_sha256': A.P.G.ids_sha256(canvas[:span[0]])},
            'actual_nfe': 8, 'commits_per_step': [4] * 8, 'wall_seconds': 1.0,
            'peak_allocated_bytes': 100, 'peak_reserved_bytes': 200}
        return row, item, encoder, provenance

    def test_valid_receipt_then_output_tampering(self):
        row, item, encoder, provenance = self.fixture()
        A.validate_item(row, item, 0, encoder, 65535, provenance)
        row['output_ids'][0] = 2
        with self.assertRaisesRegex(ValueError, 'native decoding'):
            A.validate_item(row, item, 0, encoder, 65535, provenance)

    def test_input_sha_span_rank_and_trace_checks(self):
        row, item, encoder, provenance = self.fixture()
        for key, wrong in [('input_ids_sha256', 'wrong'), ('target_span', [0, 32]),
                           ('rank', 1), ('commits_per_step', [3] * 8)]:
            changed = copy.deepcopy(row); changed[key] = wrong
            with self.assertRaises(ValueError):
                A.validate_item(changed, item, 0, encoder, 65535, provenance)

    def test_wilson_nonzero_uncertainty_at_extremes(self):
        self.assertGreater(A.wilson(0, 120)[1], 0)
        self.assertLess(A.wilson(120, 120)[0], 1)
        self.assertLess(A.wilson(96, 120)[0], 0.8)
        with self.assertRaises(ValueError):
            A.wilson(121, 120)


if __name__ == '__main__':
    unittest.main()
