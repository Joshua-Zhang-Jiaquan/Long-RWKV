"""Freeze only a complete verified calibration and retain its measured provenance."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from . import dev_grid as D, gpu_runner as U


def freeze_completed(out: Path, panel: Path, protocol: Path):
    rows = U.collect_dev(out, panel)
    measured = out / 'dev_measurements.json'
    selected = D.select_decoder(rows)
    provenance = json.loads((out / 'provenance_rank0.json').read_text())
    selected.update({
        'dev_measurements_path': str(measured.resolve()),
        'dev_measurements_sha256': U.file_sha(measured),
        'dev_provenance_sha256': U.digest(provenance),
        'checkpoint_sha256': provenance['checkpoint_sha256'],
        'dev_panel_sha256': provenance['panel_sha256'],
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'law_note': 'Bernoulli refers to reveal law. Temperature and legal-token '
                    'restriction define the predictive head; confidence is a separate policy.',
    })
    if not protocol.exists():
        U.atomic_json(protocol, {
            'schema': 'long_rwkv_f2_execution_v1',
            'quality': {'decoder': None},
            'long_context': {
                'lengths': [16384, 32768, 65536], 'declared_cells': 324,
                'generated_supported_cells': 264, 'generator_unsupported_cells': 60,
                'examples_per_supported_cell': 200,
                'data_seeds': [101, 102, 103, 104, 105],
                'inference_seed': 17, 'access_mode': 'full_canvas',
                'target_policy': 'gold span length supplied; all target tokens masked',
                'status': 'not_measured',
            },
        })
    D.freeze(protocol, selected)
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(freeze_completed(args.out, args.panel, args.protocol), indent=2))


if __name__ == '__main__':
    main()
