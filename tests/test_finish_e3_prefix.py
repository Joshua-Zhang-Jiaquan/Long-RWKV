"""Stop gate must reject missing IDs and never repeat a reserved request."""
import importlib.util
from pathlib import Path
import tempfile
import hashlib
import json
import unittest
from unittest.mock import patch
spec = importlib.util.spec_from_file_location('finish_prefix', Path(__file__).resolve().parents[1] / 'qz/finish_e3_prefix.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class FinishTests(unittest.TestCase):
    def test_subset_is_not_complete(self):
        result = F.exact_coverage(['a', 'b', 'c'], ['a', 'b'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['missing'], 1)

    def test_same_count_wrong_ids_is_not_complete(self):
        result = F.exact_coverage(['a', 'b'], ['a', 'wrong'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['extra'], 1)
        self.assertEqual(result['missing'], 1)

    def test_duplicate_ids_refused(self):
        with self.assertRaises(ValueError):
            F.exact_coverage(['a', 'b'], ['a', 'a'])
        self.assertTrue(F.exact_coverage(['a', 'b'], ['b', 'a'])['complete'])

    def test_helper_stop_identity_requires_unique_matching_output(self):
        job = {'out': '/primary', 'submission': {'job_id': 'job-primary'}}
        helper = {'out': '/helper'}
        good = {'out': '/helper', 'submission': {'job_id': 'job-helper', 'returncode': 0}}
        self.assertEqual(F.resolve_job_id(job, {'out': '/primary'}, []), 'job-primary')
        self.assertEqual(F.resolve_job_id(job, helper, [good]), 'job-helper')
        with self.assertRaises(ValueError):
            F.resolve_job_id(job, helper, [{'out': '/other', 'submission': good['submission']}])
        other = {'out': '/helper', 'submission': {'job_id': 'job-conflicting', 'returncode': 0}}
        with self.assertRaises(ValueError):
            F.resolve_job_id(job, helper, [good, other])

    def test_old_helper_subset_needs_explicit_policy(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'qualified.py'
            source.write_text('# unchanged')
            out = root / 'helper'
            (out / 'helper_schedule').mkdir(parents=True)
            job = {'out': str(root / 'primary'), 'length': 65536, 'model': 'f2'}
            part = {'out': str(out), 'cell_ids': ['overwrite'], 'helper_wrapper_sha256': 'a' * 64}
            primary = {'cell_ids': ['recall']}
            side = {'scope': 'partial_confirmation_helper_only', 'rank': 0,
                'world_size': 8, 'length': 65536, 'model': 'f2',
                'primary_out': job['out'], 'helper_out': str(out),
                'primary_cells': ['recall'], 'helper_cells': ['overwrite', 'dataflow'],
                'wrapper_sha256': 'a' * 64, 'instances_per_cell': 200,
                'expected_helper_items': 400,
                'qualified_sources': {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
            (out / 'helper_schedule/rank0.json').write_text(json.dumps(side))
            with self.assertRaisesRegex(ValueError, 'sidecar'):
                F.validate_schedule(job, part, primary, 0)
            part['schedule_policy'] = 'accepted_subset_of_recorded_schedule'
            F.validate_schedule(job, part, primary, 0)
            part['cell_ids'] = ['unassigned']
            with self.assertRaisesRegex(ValueError, 'sidecar'):
                F.validate_schedule(job, part, primary, 0)

    def test_unready_never_stops(self):
        with tempfile.TemporaryDirectory() as d, patch.object(F.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                F.stop_once({'ready': False, 'job_id': 'example'}, Path(d))
            run.assert_not_called()

    def test_reserved_receipt_never_repeats(self):
        with tempfile.TemporaryDirectory() as d, patch.object(F.subprocess, 'run') as run:
            (Path(d) / 'example.json').write_text('{"status":"request_reserved"}')
            result = F.stop_once({'ready': True, 'job_id': 'example'}, Path(d))
            self.assertEqual(result['status'], 'not_repeated')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
