"""Explicit partial F2/R0 confirmation helper for one dataflow family/length.

Invoke by absolute path outside the immutable qualified stage. Only G.all_cells
traversal is filtered; model, sampler, item RNG and rank assignments are intact.
A separate output and scheduling sidecars disclose this partial execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def select_family(cells, read_record, *, length, family, world=8):
    if length not in (32768, 65536) or family != 'code_dataflow' or world != 8:
        raise ValueError('only 32K/64K dataflow with eight workers is qualified for this helper')
    supported, chosen = [], []
    declared = 0
    for cell in cells:
        if cell.length != length:
            continue
        declared += 1
        rec = read_record(cell)
        if rec['status'] != 'ok':
            continue
        if rec.get('total_instances') != 200 or len(rec['instances']) != 200:
            raise ValueError('rank preservation requires 200 instances for every supported cell')
        supported.append(cell)
        if cell.family == family:
            chosen.append(cell)
    if declared != 108 or len(supported) != 90 or len(chosen) != 27:
        raise ValueError('expected complete 108 declared / 90 supported / 27 dataflow cells')
    return supported, chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['f2', 'r0'], default='f2')
    ap.add_argument('--qualification', type=Path)
    ap.add_argument('--qualified-root', type=Path, required=True)
    ap.add_argument('--primary-out', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--length', type=int, choices=[32768, 65536], required=True)
    ap.add_argument('--family', choices=['code_dataflow'], default='code_dataflow')
    ap.add_argument('--protocol', type=Path)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args(argv)
    if args.out.resolve() == args.primary_out.resolve():
        raise ValueError('family helper must use a separate output directory')
    if int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('exactly eight workers are required')
    root = args.qualified_root.resolve()
    sys.path.insert(0, str(root))
    from lrwkv_evidence.e3 import grid324 as G, gpu_runner as U
    if not all(Path(m.__file__).resolve().is_relative_to(root) for m in (G, U)):
        raise ValueError('runner import escaped the immutable qualified snapshot')
    supported, chosen = select_family(G.all_cells(), lambda c: json.loads(
        (args.panel / f'{c.cell_id}.json').read_text()), length=args.length, family=args.family)
    rank = int(os.environ['RANK'])
    sources = [G, U]
    module = U
    if args.model == 'r0':
        from lrwkv_evidence.e3 import causal_runner as module
        sources.append(module)
    # 64K already has an overwrite suffix worker; 32K has only the original
    # worker and this new dataflow worker. Record their separate responsibilities.
    primary = [c for c in supported if c.family == 'associative_recall'
               or (args.length == 32768 and c.family == 'overwrite_delayed_query')]
    others = [c for c in supported if c not in primary and c not in chosen]
    sidecar = {'scope': 'partial_confirmation_helper_only', 'helper_kind': 'family_partition',
        'model': args.model, 'length': args.length, 'family': args.family,
        'qualified_root': str(root), 'rank': rank, 'world_size': 8,
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'qualified_sources': {str(Path(m.__file__)): U.file_sha(Path(m.__file__)) for m in sources},
        'primary_out': str(args.primary_out.resolve()), 'helper_out': str(args.out.resolve()),
        'primary_cells': [c.cell_id for c in primary],
        'helper_cells': [c.cell_id for c in chosen],
        'other_partition_cells': [c.cell_id for c in others],
        'instances_per_cell': 200, 'expected_helper_items': 5400,
        'rank_preservation': 'every removed cell has 200 items divisible by 8; within-cell item order unchanged',
        'execution_change': 'G.all_cells filtered to 27 supported dataflow cells; inference functions unchanged'}
    side_path = args.out / 'helper_schedule' / f'rank{rank}.json'
    if side_path.exists() and json.loads(side_path.read_text()) != sidecar:
        raise ValueError('helper schedule changed across resume')
    U.atomic_json(side_path, sidecar)
    G.all_cells = lambda: list(chosen)
    common = ['--mode', 'confirm', '--panel', str(args.panel), '--out', str(args.out),
        '--length', str(args.length), '--cpu-threads', '4', '--model-dir', str(args.model_dir)]
    if args.model == 'f2':
        if not args.checkpoint or not args.protocol:
            ap.error('F2 helper requires --checkpoint and --protocol')
        common += ['--checkpoint', str(args.checkpoint), '--protocol', str(args.protocol),
                   '--seed', str(args.seed), '--reuse-unchanged-logits']
    else:
        if not args.qualification:
            ap.error('R0 helper requires --qualification')
        common += ['--qualification', str(args.qualification)]
    module.main(common)


if __name__ == '__main__':
    main()
