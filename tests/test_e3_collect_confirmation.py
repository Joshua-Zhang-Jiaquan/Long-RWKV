import json
from pathlib import Path
import tempfile
import unittest
from lrwkv_evidence.e3 import collect_confirmation as C


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.panel = self.root / 'panel'
        self.panel.mkdir()
        self.put(self.panel / 'grid_manifest.json', {'split': 'iclr2027_grid'})
        self.plan = {'panel': str(self.panel), 'jobs': []}
        self.params = dict(expected_cells=3, expected_supported=2, expected_unsupported=1, instances_per_cell=2)
        for ci in range(3):
            cell = {'cell_id': f'c{ci}', 'family': 'associative_recall', 'history_length': 16384,
                    'status': 'ok' if ci < 2 else 'unsupported', 'unsupported_reason': None if ci < 2 else 'infeasible',
                    'instances': [], 'total_instances': 2 if ci < 2 else 0}
            if ci < 2:
                cell['instances'] = [dict(cell_id=f'c{ci}', data_seed=101, instance_index=i,
                                          input_ids_sha256=f'prompt{ci}-{i}', answers=['v1000'], target_span=[8,11]) for i in range(2)]
            self.put(self.panel / f'c{ci}.json', cell)
        for m in ('f2', 'r0'):
            out = self.root / m
            out.mkdir()
            self.plan['jobs'].append({'model': m, 'length': 16384, 'out': str(out)})
            prov = {'mode': 'confirm', 'split': 'iclr2027_grid', 'configs': [{'steps': 8}],
                    'policy': {'name': 'greedy'}, 'length_filter': 16384,
                    'sources': {str(Path(C.__file__)): __import__('hashlib').sha256(Path(C.__file__).read_bytes()).hexdigest()}}
            self.put(out / 'provenance_rank0.json', prov)
            for ci in range(2):
                for i in range(2):
                    row = {'cell_id': f'c{ci}', 'data_seed': 101, 'instance_index': i,
                        'status': 'ok', 'split': 'iclr2027_grid', 'correct': (ci == 0 if m == 'f2' else i == 0),
                        'prompt_hash_verified': True, 'input_ids_sha256': f'prompt{ci}-{i}',
                        'config': {'steps': 8}, 'policy': {'name': 'greedy'},
                        'provenance_sha256': C.digest(prov), 'actual_nfe': 3, 'wall_seconds': 0.1,
                        'peak_allocated_bytes': 10, 'peak_reserved_bytes': 12}
                    row.update(gold=['v1000'], text='v1000' if row['correct'] else 'v1001',
                               prediction='v1000' if row['correct'] else 'v1001', parse_ok=True,
                               joint_exact=row['correct'], output_ids=[1,2,3], rank=0)
                    dest = out / ('config' if m == 'f2' else '') / f'c{ci}__101__{i}.json'
                    self.put(dest, row)

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_complete_paired_support_preserved_not_zero_filled(self):
        result = C.collect(self.plan, **self.params)
        self.assertEqual(result['supported_cells'], 2)
        self.assertEqual(result['excluded_cells'], 1)
        self.assertEqual(result['overall'], {'F2': 0.5, 'R0': 0.5, 'difference': 0.0})
        self.assertEqual([r['difference'] for r in result['cell_scores']], [0.5, -0.5])
        self.assertEqual(sum(r['F2_only'] for r in result['cell_scores']), 1)
        self.assertNotIn('F2', result['unsupported_cells'][0])
        self.assertIn('2/3', C.render_tex(result))

    def test_wall_resource_quantiles_inclusive_linear(self):
        for path, wall in zip(sorted((self.root / 'r0').glob('c*__*.json')), [1, 2, 3, 100]):
            row = C.read(path)
            row['wall_seconds'] = wall
            self.put(path, row)
        result = C.collect(self.plan, **self.params)
        resource = result['resources']['R0']['16384']
        self.assertEqual(resource['wall_seconds_sum'], 106)
        self.assertEqual(resource['wall_seconds_mean'], 26.5)
        self.assertEqual(resource['wall_seconds_median'], 2.5)
        self.assertAlmostEqual(resource['wall_seconds_p95'], 85.45)
        self.assertIn('inclusive linear', resource['wall_quantile_method'])
        self.assertIn('asymmetric warmup', result['resource_scope'])
        self.assertIn('uncached AR', result['resource_scope'])

    def test_bootstrap_equal_panel_and_cell_weights_not_pooled_items(self):
        cells = [dict(cell_id='a', family='A', length=16384, n=2,
                      F2_only=2, R0_only=0, both_correct=0),
                 dict(cell_id='b', family='B', length=16384, n=10,
                      F2_only=0, R0_only=0, both_correct=0),
                 dict(cell_id='c', family='B', length=16384, n=100,
                      F2_only=0, R0_only=0, both_correct=0)]
        result = C.paired_bootstrap(cells)
        self.assertEqual(result['overall_ci95']['F2'], [0.5, 0.5])
        self.assertEqual(result['overall_ci95']['difference'], [0.5, 0.5])
        cells[1]['F2_only'] = 10
        result = C.paired_bootstrap(cells)
        self.assertEqual(result['overall_ci95']['F2'], [0.75, 0.75])

    def test_bootstrap_reproducible_and_preserves_pairing(self):
        cells = [dict(cell_id='a', family='A', length=16384, n=20,
                      F2_only=0, R0_only=0, both_correct=10)]
        first = C.paired_bootstrap(cells)
        self.assertEqual(first, C.paired_bootstrap(cells))
        self.assertEqual(first['overall_ci95']['difference'], [0.0, 0.0])
        self.assertLess(first['overall_ci95']['F2'][0], 0.5)
        self.assertGreater(first['overall_ci95']['F2'][1], 0.5)
        self.assertEqual(first['overall_ci95']['F2'], first['overall_ci95']['R0'])

    def test_missing_refused_status_no_accuracy(self):
        (self.root / 'f2/config/c0__101__0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete confirmation'):
            C.collect(self.plan, **self.params)
        partial = C.collect(self.plan, status_only=True, **self.params)
        self.assertEqual(partial['missing_items']['F2'], 1)
        self.assertNotIn('overall', partial)
        self.assertEqual(partial['excluded_cells'], 1)
        with self.assertRaises(ValueError):
            C.render_tex(partial)

    def test_duplicate_config_receipt_refused(self):
        self.put(self.root / 'f2/other/c0__101__0.json', C.read(self.root / 'f2/config/c0__101__0.json'))
        with self.assertRaisesRegex(ValueError, 'duplicate outcome'):
            C.collect(self.plan, **self.params)

    def test_unsupported_outcome_cannot_become_zero(self):
        row = C.read(self.root / 'r0/c0__101__0.json')
        row['cell_id'] = 'c2'
        row['correct'] = False
        self.put(self.root / 'r0/c2__101__0.json', row)
        with self.assertRaisesRegex(ValueError, 'unsupported outcome'):
            C.collect(self.plan, **self.params)

    def partition_fixture(self):
        job = self.plan['jobs'][0]
        helper = self.root / 'helper'
        prov = C.read(self.root / 'f2/provenance_rank0.json')
        self.put(helper / 'provenance_rank0.json', prov)
        for i in range(2):
            self.put(helper / f'config/c1__101__{i}.json', C.read(self.root / f'f2/config/c1__101__{i}.json'))
        job['partitions'] = [{'out': job['out'], 'cell_ids': ['c0']},
                             {'out': str(helper), 'cell_ids': ['c1'], 'helper_wrapper_sha256': 'a'*64}]
        self.put(helper / 'helper_schedule/rank0.json', {
            'scope': 'partial_confirmation_helper_only', 'model': 'f2', 'length': 16384,
            'primary_out': job['out'], 'helper_out': str(helper), 'primary_cells': ['c0'],
            'helper_cells': ['c1'], 'instances_per_cell': 2, 'expected_helper_items': 2,
            'wrapper_sha256': 'a'*64, 'rank': 0, 'world_size': 8, 'qualified_sources': prov['sources']})
        return job

    def test_partition_combines_disjoint_sources_ignores_primary_extras(self):
        self.partition_fixture()
        result = C.collect(self.plan, **self.params)
        self.assertEqual(result['observed_items'], {'F2': 4, 'R0': 4})
        self.assertEqual(result['partitions'][0]['unused_receipts_outside_assignment'], 2)
        self.assertEqual(result['overall']['F2'], 0.5)

    def test_partition_overlap_and_missing_cover_refused(self):
        job = self.partition_fixture()
        job['partitions'][1]['cell_ids'] = ['c0', 'c1']
        with self.assertRaisesRegex(ValueError, 'partition overlap'):
            C.collect(self.plan, **self.params)
        job['partitions'] = job['partitions'][:1]
        with self.assertRaisesRegex(ValueError, 'do not cover'):
            C.collect(self.plan, **self.params)

    def test_helper_schedule_superset_requires_explicit_policy(self):
        job = self.partition_fixture()
        path = self.root / 'helper/helper_schedule/rank0.json'
        sidecar = C.read(path)
        sidecar['helper_cells'] = ['c0', 'c1']
        sidecar['expected_helper_items'] = 4
        self.put(path, sidecar)
        with self.assertRaisesRegex(ValueError, 'orchestration sidecar'):
            C.collect(self.plan, **self.params)
        job['partitions'][1]['schedule_policy'] = 'accepted_subset_of_recorded_schedule'
        self.put(self.root / 'helper/config/c0__101__0.json',
                 C.read(self.root / 'f2/config/c0__101__0.json'))
        result = C.collect(self.plan, **self.params)
        self.assertEqual(result['observed_items'], {'F2': 4, 'R0': 4})
        self.assertEqual(result['partitions'][1]['scheduled_cells'], 2)
        self.assertEqual(result['partitions'][1]['unused_receipts_outside_assignment'], 1)
        job['partitions'] = job['partitions'][1:]
        with self.assertRaisesRegex(ValueError, 'omits primary'):
            C.collect(self.plan, **self.params)

    def test_helper_sidecar_required(self):
        self.partition_fixture()
        (self.root / 'helper/helper_schedule/rank0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'orchestration sidecar'):
            C.collect(self.plan, **self.params)

    def test_forged_correctness_refused(self):
        path = self.root / 'r0/c0__101__0.json'
        row = C.read(path)
        row['correct'] = not row['correct']
        self.put(path, row)
        with self.assertRaisesRegex(ValueError, 'parsed outcome'):
            C.collect(self.plan, **self.params)

    def test_prompt_and_provenance_mismatch_refused(self):
        path = self.root / 'r0/c0__101__0.json'
        row = C.read(path)
        row['input_ids_sha256'] = 'wrong'
        self.put(path, row)
        with self.assertRaisesRegex(ValueError, 'invalid measured receipt'):
            C.collect(self.plan, **self.params)
        row['input_ids_sha256'] = 'prompt0-0'
        row['provenance_sha256'] = 'wrong'
        self.put(path, row)
        with self.assertRaisesRegex(ValueError, 'invalid measured receipt'):
            C.collect(self.plan, **self.params)


if __name__ == '__main__':
    unittest.main()
