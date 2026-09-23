"""Scheduling-only qualification: exact disjoint cover and preserved item ranks."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.e3.helper_suffix import select_partition


def fake_grid():
    cells = [SimpleNamespace(length=65536, cell_id=f'cell{i}',
             family='associative_recall' if i < 36 else 'overwrite_delayed_query' if i < 63 else 'code_dataflow') for i in range(108)]
    def record(cell):
        i = int(cell.cell_id[4:])
        return {'status': 'ok' if i < 90 else 'unsupported',
                'total_instances': 200, 'instances': [None] * 200}
    return cells, record


class HelperTests(unittest.TestCase):
    def test_partition_disjoint_exhaustive_and_rank_invariant(self):
        cells, record = fake_grid()
        primary, helper = select_partition(cells, record)
        self.assertEqual(len(primary), 36)
        self.assertEqual(len(helper), 54)
        self.assertEqual(primary + helper, cells[:90])
        for local_cell, cell in enumerate(helper):
            original_cell = cells.index(cell)
            for index in range(200):
                self.assertEqual((original_cell * 200 + index) % 8,
                                 (local_cell * 200 + index) % 8)

    def test_nondivisible_cells_refused(self):
        cells, record = fake_grid()
        def bad(cell):
            rec = record(cell)
            rec['instances'] = [None] * 199
            return rec
        with self.assertRaisesRegex(ValueError, '200 instances'):
            select_partition(cells, bad)
        with self.assertRaises(ValueError):
            select_partition(cells, record, world=3)

    def test_missing_cell_or_family_reordering_refused(self):
        cells, record = fake_grid()
        with self.assertRaisesRegex(ValueError, 'complete 64K grid'):
            select_partition(cells[:-1], record)
        cells[0].family = 'code_dataflow'
        with self.assertRaisesRegex(ValueError, '36 associative'):
            select_partition(cells, record)


if __name__ == '__main__':
    unittest.main()
