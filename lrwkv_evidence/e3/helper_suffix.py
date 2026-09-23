"""Run a disclosed 64K suffix partition through an immutable qualified runner.

This file must be invoked by absolute path, not imported from the qualified
package. No model, sampler, prompt or per-item seed logic is changed. Helpers
write a separate output directory; the caller owns primary-prefix completion
checks and final union validation. A helper alone is never full-grid evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def select_partition(cells, read_record, world=8):
    supported, all_at_length = [], []
    for cell in cells:
        if cell.length != 65536:
            continue
        all_at_length.append(cell)
        record = read_record(cell)
        if record['status'] != 'ok':
            continue
        if record.get('total_instances') != 200 or len(record['instances']) != 200 or 200 % world:
            raise ValueError('partition rank preservation requires 200 instances/cell divisible by world size')
        supported.append(cell)
    if len(supported) != 90 or len(all_at_length) != 108:
        raise ValueError('expected the complete 64K grid: 90 supported / 108 declared cells')
    primary, helper = supported[:36], supported[36:]
    if any(c.family != 'associative_recall' for c in primary):
        raise ValueError('primary prefix is not the 36 associative-recall cells')
    if any(c.family == 'associative_recall' for c in helper):
        raise ValueError('helper suffix overlaps the associative-recall prefix')
    return primary, helper


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qualified-root', type=Path, required=True)
    ap.add_argument('--model', choices=['f2', 'r0'], required=True)
    ap.add_argument('--primary-out', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--protocol', type=Path)
    ap.add_argument('--qualification', type=Path)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args(argv)
    if args.out.resolve() == args.primary_out.resolve():
        raise ValueError('helper output must differ from the primary output')
    if int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('partition helper requires exactly eight workers')
    root = args.qualified_root.resolve()
    sys.path.insert(0, str(root))
    from lrwkv_evidence.e3 import grid324 as G, gpu_runner as U
    if not Path(G.__file__).resolve().is_relative_to(root) or not Path(U.__file__).resolve().is_relative_to(root):
        raise ValueError('import did not resolve inside the immutable qualified source snapshot')
    cells = G.all_cells()
    primary, helper = select_partition(cells, lambda cell: json.loads(
        (args.panel / f'{cell.cell_id}.json').read_text()))
    rank = int(os.environ['RANK'])
    source_paths = [Path(G.__file__), Path(U.__file__)]
    if args.model == 'f2':
        module = U
    else:
        from lrwkv_evidence.e3 import causal_runner as module
        source_paths.append(Path(module.__file__))
    sidecar = {'scope': 'partial_confirmation_helper_only', 'length': 65536,
        'model': args.model, 'qualified_root': str(root), 'rank': rank, 'world_size': 8,
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'qualified_sources': {str(p): U.file_sha(p) for p in source_paths},
        'primary_out': str(args.primary_out.resolve()), 'helper_out': str(args.out.resolve()),
        'primary_cells': [c.cell_id for c in primary],
        'helper_cells': [c.cell_id for c in helper],
        'instances_per_cell': 200, 'expected_helper_items': 10800,
        'rank_preservation': 'removed prefix has 7200 items, divisible by 8; within-cell order unchanged',
        'execution_change': 'G.all_cells returns only fixed 54 supported suffix cells; all inference functions unchanged'}
    sidecar_path = args.out / 'helper_schedule' / f'rank{rank}.json'
    if sidecar_path.exists() and json.loads(sidecar_path.read_text()) != sidecar:
        raise ValueError('helper schedule changed across resume')
    U.atomic_json(sidecar_path, sidecar)
    # Sole execution patch. Keep the original module __file__ and all model and
    # sampling functions intact; sidecar above discloses this narrowed traversal.
    G.all_cells = lambda: list(helper)
    common = ['--mode', 'confirm', '--panel', str(args.panel), '--out', str(args.out),
        '--length', '65536', '--cpu-threads', '4', '--model-dir', str(args.model_dir)]
    if args.model == 'f2':
        if not args.protocol or not args.checkpoint:
            ap.error('F2 requires --protocol and --checkpoint')
        common += ['--protocol', str(args.protocol), '--checkpoint', str(args.checkpoint),
                   '--seed', str(args.seed), '--reuse-unchanged-logits']
    else:
        if not args.qualification:
            ap.error('R0 requires --qualification')
        common += ['--qualification', str(args.qualification)]
    module.main(common)


if __name__ == '__main__':
    main()
