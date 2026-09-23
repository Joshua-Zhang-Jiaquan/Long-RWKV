"""Queued zero-allocation jobs count against the 32-GPU submission cap."""
import importlib.util
from pathlib import Path
import sys
import unittest
qz = Path(__file__).resolve().parents[1] / 'qz'
sys.path.insert(0, str(qz))
spec = importlib.util.spec_from_file_location('submit_mvp', qz / 'submit_mvp_lookup.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class CapacityTests(unittest.TestCase):
    def test_queued_jobs_cannot_hide_as_zero_gpu(self):
        self.assertEqual(M.CAP, 32)
        active = next(s for s in M.C.LIVE_STATUSES if s != 'job_running')
        census = M.live_usage([{'name': f'lrwkv-job{i}', 'status': active, 'gpu_count': 0} for i in range(4)])
        self.assertEqual(census['live_reserved_gpus'], 32)
        self.assertGreater(census['live_reserved_gpus'] + 8, M.CAP)

    def test_terminal_jobs_do_not_consume_cap(self):
        census = M.live_usage([{'name': 'lrwkv-done', 'status': 'job_succeeded', 'gpu_count': 8},
                              {'name': 'lrwkv-live', 'status': 'job_running', 'gpu_count': 16}])
        self.assertEqual(census['live_reserved_gpus'], 16)


if __name__ == '__main__':
    unittest.main()
