"""CPU qualification of the real sampler and confirmation refusal gates."""
import sys
import tempfile
import json
from unittest.mock import patch
from types import SimpleNamespace
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.e3 import gpu_runner as U, dev_grid as D


class ToyModel:
    def __init__(self):
        self.seen = []
    def __call__(self, ids):
        self.seen.append(ids.clone())
        logits = torch.zeros((*ids.shape, 6))
        logits[..., 2] = 2.0
        return logits


class SamplerTests(unittest.TestCase):
    def test_reuse_audit_rejects_nonrepeatable_model(self):
        class ChangingModel(ToyModel):
            def __call__(self, ids):
                logits = super().__call__(ids)
                logits[..., 2] += len(self.seen)
                return logits
        with self.assertRaisesRegex(ValueError, 'different target logits'):
            U.sample_target(ChangingModel(), torch.zeros((1, 3), dtype=torch.long),
                (1, 3), D.sweep_configs()[8], mask_id=0, pad_id=1,
                generator=torch.Generator().manual_seed(1), repeat_logit_audit={})

    def test_rotating_config_assignment_balances_nominal_work(self):
        work = [0] * 8
        owned = [[] for _ in range(8)]
        for item in range(72):
            for index, config in enumerate(D.sweep_configs()):
                rank = U.assigned_rank(item, index, 8)
                work[rank] += config['steps']
                owned[rank].append((item, index))
        self.assertEqual(len(set(work)), 1)
        self.assertEqual(len(set(pair for shard in owned for pair in shard)), 72 * 16)

    def test_all_configs_reproducible_visible_immutable_and_actual_calls(self):
        ids = torch.tensor([[3, 4, 5, 5, 5, 3]])
        for config in D.sweep_configs():
            model = ToyModel()
            out = U.sample_target(model, ids, (2, 5), config, mask_id=0, pad_id=1,
                                  generator=torch.Generator().manual_seed(12))
            again = U.sample_target(ToyModel(), ids, (2, 5), config, mask_id=0, pad_id=1,
                                    generator=torch.Generator().manual_seed(12))
            self.assertEqual(out, again)
            self.assertTrue(torch.equal(model.seen[0][0, 2:5], torch.zeros(3, dtype=torch.long)))
            self.assertEqual(out[1], len(model.seen))
            self.assertEqual(sum(out[2]), 3)
            self.assertTrue(all(t not in (0, 1) for t in out[0]))
            for canvas in model.seen:
                self.assertTrue(torch.equal(canvas[:, [0, 1, 5]], ids[:, [0, 1, 5]]))
            self.assertTrue(torch.equal(ids, torch.tensor([[3, 4, 5, 5, 5, 3]])))

    def test_cache_exact_for_all_configs_and_invalidated_after_commits(self):
        class CanvasModel(ToyModel):
            def __call__(self, ids):
                logits = super().__call__(ids)
                # Every commit changes the next forward's prediction distribution.
                logits[..., 2] = 0
                logits[..., 2 + int(ids.sum()) % 4] = 8
                return logits
        ids = torch.tensor([[3, 0, 0, 0, 4]])
        for config in D.sweep_configs():
            original, cached = CanvasModel(), CanvasModel()
            common = dict(mask_id=0, pad_id=1)
            a = U.sample_target(original, ids, (1, 4), config,
                generator=torch.Generator().manual_seed(99), **common)
            b = U.sample_target(cached, ids, (1, 4), config,
                generator=torch.Generator().manual_seed(99), reuse_unchanged_logits=True, **common)
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[2], b[2])
            self.assertLessEqual(b[1], 3)
            self.assertLessEqual(b[1], a[1])
            unique_canvases = []
            for canvas in original.seen:
                if not unique_canvases or not torch.equal(canvas, unique_canvases[-1]):
                    unique_canvases.append(canvas)
            self.assertEqual(len(cached.seen), len(unique_canvases))
            for got, expected in zip(cached.seen, unique_canvases):
                self.assertTrue(torch.equal(got, expected))

    def test_confidence_linear_schedule(self):
        cfg = D.sweep_configs()[8]
        _, calls, trace = U.sample_target(ToyModel(), torch.zeros((1, 16), dtype=torch.long),
            (0, 16), cfg, mask_id=0, pad_id=1, generator=torch.Generator().manual_seed(1))
        self.assertEqual(calls, 8)
        self.assertEqual(trace, [2] * 8)

    def test_bernoulli_does_not_force_fixed_budget(self):
        cfg = D.sweep_configs()[0]
        _, _, trace = U.sample_target(ToyModel(), torch.zeros((1, 80), dtype=torch.long),
            (0, 80), cfg, mask_id=0, pad_id=1, generator=torch.Generator().manual_seed(1))
        self.assertNotEqual(trace, [10] * 8)
        self.assertEqual(sum(trace), 80)

    def test_confirmation_frozen_and_dev_selection_required(self):
        with self.assertRaises(SystemExit):
            U.configs_for('confirm', {})
        cfg = D.sweep_configs()[0]
        with self.assertRaises(ValueError):
            U.configs_for('confirm', {'quality': {'decoder': cfg}})
        self.assertEqual(U.configs_for('confirm', {'quality': {'decoder': {
            **cfg, 'dev_panel': 'lrwkv_evidence.e3.dev_grid'}}}), [cfg])

    def test_collector_refuses_missing_configuration_and_mixed_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            panel, out = Path(root) / 'panel', Path(root) / 'out'
            U.atomic_json(panel / 'dev_manifest.json', {'split': 'dev', 'instances': 72,
                'instances_per_seed': 8})
            cells = D.panel_cells()
            instances = {c.cell_id: [SimpleNamespace(data_seed=seed, instance_index=i,
                input_ids_sha256=f'{c.cell_id}-{seed}-{i}') for seed in (201, 202, 203)
                for i in range(8)] for c in cells}
            provenance = {'split': 'dev', 'manifest_sha256': U.file_sha(panel / 'dev_manifest.json')}
            U.atomic_json(out / 'provenance_rank0.json', provenance)
            def load(path):
                return instances[Path(path).stem]
            with patch.object(U.L, 'load_cell_record', side_effect=load), patch.object(U.L, 'instances_of', side_effect=lambda x: x), patch.object(D, 'dev_seeds', return_value=(201, 202, 203)):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    U.collect_dev(out, panel)
                for config in D.sweep_configs():
                    for c in cells:
                        for inst in instances[c.cell_id]:
                            path = out / D.config_id(config) / f'{c.cell_id}__{inst.data_seed}__{inst.instance_index}.json'
                            U.atomic_json(path, {'split': 'dev', 'config': config, 'status': 'ok',
                                'cell_id': c.cell_id, 'data_seed': inst.data_seed,
                                'instance_index': inst.instance_index, 'input_ids_sha256': inst.input_ids_sha256,
                                'prompt_hash_verified': True, 'correct': False, 'actual_nfe': 1,
                                'wall_seconds': 0.1, 'provenance_sha256': U.digest(provenance)})
                self.assertEqual(len(U.collect_dev(out, panel)), 16)
                row = json.loads(path.read_text())
                row['provenance_sha256'] = 'wrong'
                U.atomic_json(path, row)
                with self.assertRaisesRegex(ValueError, 'different run provenance'):
                    U.collect_dev(out, panel)

    def test_provenance_change_preserves_original_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'provenance_rank0.json'
            U.pin_provenance(path, {'checkpoint': 'old'})
            with self.assertRaisesRegex(ValueError, 'changed run provenance'):
                U.pin_provenance(path, {'checkpoint': 'new'})
            self.assertEqual(json.loads(path.read_text()), {'checkpoint': 'old'})

    def test_resume_rejects_error_receipt_even_with_matching_hash(self):
        instance = SimpleNamespace(cell_id='c', data_seed=201, instance_index=0,
                                   input_ids_sha256='prompt')
        with self.assertRaisesRegex(ValueError, 'resume receipt'):
            U.validate_resume({'status': 'error', 'provenance_sha256': 'p',
                               'input_ids_sha256': 'prompt'}, provenance_hash='p',
                              instance=instance, config=D.sweep_configs()[0], split='dev')

    def test_item_seed_split_disjoint(self):
        self.assertNotEqual(U.item_seed('dev', 'c', 201, 0), U.item_seed('iclr2027_grid', 'c', 201, 0))
        self.assertEqual(U.item_seed('dev', 'c', 201, 0), U.item_seed('dev', 'c', 201, 0))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
