"""Fixed balanced tasks, no target leakage, and a fail-closed competence gate."""
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.mvp import lookup_pilot as P


class PilotTests(unittest.TestCase):
    def test_balanced_fresh_deterministic_items(self):
        items = P.make_items()
        self.assertEqual(items, P.make_items())
        self.assertEqual(len(items), 240)
        self.assertEqual(len({r['prompt'] for r in items}), 240)
        for depth in (1, 2):
            subset = [r for r in items if r['depth'] == depth]
            self.assertEqual(Counter(r['gold'] for r in subset), {c: 5 for c in P.COLORS})
            for row in subset:
                self.assertTrue(row['prompt'].startswith('User:'))
                self.assertTrue(row['prompt'].endswith('Assistant:'))
                self.assertEqual(sum(f"favorite color is {row['gold']}." in f for f in row['logical_facts']), 1)

    def test_no_gold_target_or_gold_length(self):
        encoder = SimpleNamespace(encode_text=lambda text: list(text.encode()))
        for item in P.make_items()[:3]:
            ids, span = P.canvas_for(encoder, item['prompt'], 999)
            self.assertEqual(ids[span[0]:], [999] * 32)
            self.assertEqual(span[1] - span[0], 32)
            self.assertEqual(ids[:span[0]], list(item['prompt'].encode()))

    def test_exact_parser_distinct_from_substring(self):
        self.assertEqual(P.parse_answer('\nAnswer: "BLUE".\nExplanation'), 'blue')
        self.assertIsNone(P.parse_answer('The answer is blue.'))
        self.assertTrue(P.substring_correct('The answer is blue.', 'blue'))
        self.assertFalse(P.substring_correct('blueberry', 'blue'))
        self.assertIsNone(P.parse_answer('red or blue'))

    def test_gate_requires_complete_every_model_depth(self):
        rows = []
        for model in ('f2', 'r0'):
            for item in P.make_items():
                # Exactly96/120correct in everygroup; allremaining25?24 fail.
                correct = item['index'] < 96
                text = item['gold'] if correct else 'I do not know.'
                rows.append({'model': model, 'item_id': item['item_id'], 'status': 'ok',
                    'depth': item['depth'], 'gold': item['gold'], 'text': text,
                    'prediction': P.parse_answer(text), 'correct': correct,
                    'substring_correct': P.substring_correct(text, item['gold']),
                    'prompt_sha256': P.U.digest(item['prompt']), 'output_ids': [1] * 32,
                    'wall_seconds': 1.0})
        self.assertTrue(P.gate_summary(rows)['gate_passed'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            P.gate_summary(rows[:-1])
        row = rows[0]; row.update(text='unsure', prediction=None, correct=False, substring_correct=False)
        self.assertFalse(P.gate_summary(rows)['gate_passed'])


if __name__ == '__main__':
    unittest.main()
