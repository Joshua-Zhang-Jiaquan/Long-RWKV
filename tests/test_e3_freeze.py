import json
from pathlib import Path
import tempfile
import unittest

from lrwkv_evidence.e3 import dev_grid as D, gpu_runner as U


class CalibrationFreezeTests(unittest.TestCase):
    def test_changed_measurements_and_wrong_winner_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'measurements.json'
            rows = [{**config, 'accuracy': 0.0, 'n': 72} for config in D.sweep_configs()]
            path.write_text(json.dumps(rows))
            decoder = {**D.select_decoder(rows), 'dev_measurements_path': str(path),
                       'dev_measurements_sha256': U.file_sha(path),
                       'dev_provenance_sha256': 'a' * 64,
                       'checkpoint_sha256': 'b' * 64, 'dev_panel_sha256': 'c' * 64}
            protocol = {'quality': {'decoder': decoder}}
            self.assertEqual(U.verify_confirmation_freeze(protocol)['steps'], 8)
            decoder['steps'] = 64
            with self.assertRaisesRegex(ValueError, 'selection rule'):
                U.verify_confirmation_freeze(protocol)
            decoder['steps'] = 8
            rows[1]['accuracy'] = 1.0
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'changed after freeze'):
                U.verify_confirmation_freeze(protocol)

    def test_partial_sweep_cannot_be_frozen_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'measurements.json'
            rows = [{**config, 'accuracy': 0.0, 'n': 72} for config in D.sweep_configs()[:-1]]
            path.write_text(json.dumps(rows))
            decoder = {**D.select_decoder(rows), 'dev_measurements_path': str(path),
                       'dev_measurements_sha256': U.file_sha(path),
                       'dev_provenance_sha256': 'a' * 64,
                       'checkpoint_sha256': 'b' * 64, 'dev_panel_sha256': 'c' * 64}
            with self.assertRaisesRegex(ValueError, 'complete declared sweep'):
                U.verify_confirmation_freeze({'quality': {'decoder': decoder}})
