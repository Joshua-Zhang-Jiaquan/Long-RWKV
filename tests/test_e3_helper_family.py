"""Dataflow partition preserves canonical item ranks at both supported lengths."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.e3.helper_family import select_family


class FamilyTests(unittest.TestCase):
    def test_rank_preservation_and_exact_family(self):
        for length in (32768, 65536):
            cells = [SimpleNamespace(length=length, cell_id=str(i),
                family='associative_recall' if i < 36 else 'overwrite_delayed_query' if i < 63 else 'code_dataflow') for i in range(108)]
            def read(cell):
                return {'status': 'ok' if int(cell.cell_id) < 90 else 'unsupported',
                        'total_instances': 200, 'instances': [None] * 200}
            supported, chosen = select_family(cells, read, length=length, family='code_dataflow')
            self.assertEqual(len(chosen), 27)
            self.assertTrue(all(c.family == 'code_dataflow' for c in chosen))
            for new_cell, cell in enumerate(chosen):
                old_cell = supported.index(cell)
                for index in range(200):
                    self.assertEqual((old_cell * 200 + index) % 8, (new_cell * 200 + index) % 8)

    def test_invalid_length_family_or_incomplete_panel_refused(self):
        for length, family in ((16384, 'code_dataflow'), (65536, 'associative_recall'), (65536, 'code_dataflow')):
            with self.assertRaises(ValueError):
                select_family([], lambda c: {}, length=length, family=family)


if __name__ == '__main__':
    unittest.main()
