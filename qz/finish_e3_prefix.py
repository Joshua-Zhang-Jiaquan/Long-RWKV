"""Validate completed accepted partitions; optionally stop their original worker jobs.

Dry-run by default. --stop requires all accepted canonical partition outcomes plus
validated companion partition execution before making one StopJob request. A durable local
request receipt prevents automatic duplicate stop requests, even after failures.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence.e3 import collect_confirmation as C, grid324 as G


def exact_coverage(expected, present):
    """ID equality is necessary; a count alone cannot certify a finished prefix."""
    expected, present = list(expected), list(present)
    if len(set(expected)) != len(expected) or len(set(present)) != len(present):
        raise ValueError('duplicate canonical or observed item IDs')
    missing = set(expected) - set(present)
    extra = set(present) - set(expected)
    return {'complete': not missing and not extra, 'expected': len(expected),
            'observed': len(present), 'missing': len(missing), 'extra': len(extra)}


def eight_manifests(out):
    paths = [out / f'provenance_rank{r}.json' for r in range(8)]
    if not all(p.is_file() for p in paths):
        raise ValueError(f'not all eight provenance manifests exist: {out}')
    manifests = [C.read(p) for p in paths]
    if len({C.digest(m) for m in manifests}) != 1:
        raise ValueError('rank provenance differs')
    return manifests[0]


def resolve_job_id(job, partition, submission_plans):
    """Resolve output -> submitted job, refusing ambiguous helper submissions."""
    target = Path(partition['out']).resolve()
    if target == Path(job['out']).resolve():
        return job['submission']['job_id']
    matches = set()
    for plan in submission_plans:
        for candidate in plan.get('jobs', [plan]):
            if not candidate.get('out') or Path(candidate['out']).resolve() != target:
                continue
            submission = candidate.get('submission', {})
            if submission.get('returncode') == 0 and submission.get('job_id'):
                matches.add(submission['job_id'])
    if len(matches) != 1:
        raise ValueError('helper output has missing or ambiguous submitted job identity')
    return next(iter(matches))


def validate_schedule(job, part, primary, rank):
    """Validate all sidecars, even ranks that have not written an outcome yet."""
    out = Path(part['out'])
    side = C.read(out / 'helper_schedule' / f'rank{rank}.json')
    assigned, scheduled = set(part['cell_ids']), set(side.get('helper_cells', []))
    primary_assigned, primary_scheduled = set(primary['cell_ids']), set(side.get('primary_cells', []))
    subset = part.get('schedule_policy') == 'accepted_subset_of_recorded_schedule'
    if (not scheduled or len(scheduled) != len(side.get('helper_cells', []))
        or not assigned <= scheduled or (not subset and assigned != scheduled)
        or not primary_assigned <= primary_scheduled or (not subset and primary_assigned != primary_scheduled)
        or scheduled & primary_scheduled
        or side.get('scope') != 'partial_confirmation_helper_only' or side.get('rank') != rank
        or side.get('world_size') != 8 or side.get('length') != job['length']
        or side.get('model', '').upper() != job['model'].upper()
        or Path(side.get('primary_out', '')).resolve() != Path(job['out']).resolve()
        or Path(side.get('helper_out', '')).resolve() != out.resolve()
        or not part.get('helper_wrapper_sha256')
        or side.get('wrapper_sha256') != part['helper_wrapper_sha256']
        or side.get('instances_per_cell') != 200
        or side.get('expected_helper_items') != len(scheduled) * 200
        or not side.get('qualified_sources')):
        raise ValueError('invalid helper scheduling sidecar')
    for name, sha in side['qualified_sources'].items():
        import hashlib
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
            raise ValueError('helper qualified source hash mismatch')


def assess_job(job, panel, coverage, *, partition_out=None, submission_plans=()):
    model = job['model'].upper()
    if job['length'] not in (32768, 65536):
        raise ValueError('stop gate only supports 32K/64K partitioned jobs')
    parts = job.get('partitions', [])
    primary_parts = [p for p in parts if Path(p['out']).resolve() == Path(job['out']).resolve()]
    if len(primary_parts) != 1 or len(parts) < 2:
        raise ValueError('requires an explicit partition plan with a primary and helpers')
    primary = primary_parts[0]
    target = Path(partition_out or job['out']).resolve()
    selected = [p for p in parts if Path(p['out']).resolve() == target]
    if len(selected) != 1:
        raise ValueError('selected output is not a unique assigned partition')
    part = selected[0]
    assigned = set(part['cell_ids'])
    expected = []
    for cid in part['cell_ids']:
        cell = C.read(panel / f'{cid}.json')
        if (cell['history_length'] != job['length'] or cell['status'] != 'ok'
            or cell.get('total_instances') != 200 or len(cell['instances']) != 200):
            raise ValueError('invalid canonical partition cell')
        expected.extend(C.item_id(i) for i in cell['instances'])
    if len(assigned) != len(part['cell_ids']) or not expected:
        raise ValueError('empty or duplicated partition assignment')
    out = Path(part['out'])
    present = []
    paths = out.glob('*/*__*__*.json') if model == 'F2' else out.glob('*__*__*.json')
    for path in paths:
        row = C.read(path)
        if row.get('cell_id') in assigned:
            present.append(C.item_id(row))
    check = exact_coverage(expected, present)
    selected_status = None
    companion_progress = []
    provenance = None
    for companion in parts:
        cout = Path(companion['out'])
        status = next(p for p in coverage['partitions'] if p['model'] == model
            and p['length'] == job['length'] and Path(p['out']).resolve() == cout.resolve())
        cm = eight_manifests(cout)
        if provenance is None:
            provenance = cm
        elif provenance != cm:
            raise ValueError('companion model/decoder provenance differs')
        if cout.resolve() != Path(job['out']).resolve():
            for rank in range(8):
                validate_schedule(job, companion, primary, rank)
        if cout.resolve() == target:
            selected_status = status
        else:
            companion_progress.append({'out': str(cout), 'validated_receipts': status['used_receipts'],
                                       'expected': len(companion['cell_ids']) * 200})
    if check['complete'] and selected_status['used_receipts'] != len(expected):
        raise ValueError('coverage changed during audit; rerun after receipts settle')
    ready = check['complete'] and all(c['validated_receipts'] > 0 for c in companion_progress)
    return {'model': model, 'length': job['length'], 'partition_out': str(target),
            'job_id': resolve_job_id(job, part, submission_plans), 'ready': ready,
            'primary_coverage': check, 'accepted_partition_coverage': check,
            'companion_progress': companion_progress,
            'unused_primary_suffix_receipts': selected_status['unused_receipts_outside_assignment'],
            'reason': 'accepted partition complete and companion execution validated' if ready
                      else 'accepted partition incomplete or a companion has no validated outcomes'}


def stop_once(summary, receipt_dir):
    if not summary.get('ready'):
        raise ValueError('not ready to stop')
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"{summary['job_id']}.json"
    record = {'status': 'request_reserved', 'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'validation': summary, 'command': ['qz', 'train', 'StopJob', '--job-id', summary['job_id'], '-o', 'json']}
    # Reserve before requesting StopJob: a timeout must not trigger a blind retry.
    try:
        with path.open('x') as f:
            json.dump(record, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        return {'status': 'not_repeated', 'receipt': str(path)}
    try:
        result = subprocess.run(record['command'], capture_output=True, text=True, timeout=60)
        record.update({'status': 'request_succeeded' if result.returncode == 0 else 'request_failed',
                       'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    except Exception as exc:
        record.update({'status': 'request_outcome_unknown', 'error': str(exc)})
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(tmp, path)
    return {'status': record['status'], 'receipt': str(path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, default=Path('results/e3_confirmation_plan.json'))
    ap.add_argument('--model', choices=['f2', 'r0'])
    ap.add_argument('--length', type=int, choices=[32768, 65536], default=65536)
    ap.add_argument('--partition-out', type=Path, help='assigned output to stop; default original job output')
    ap.add_argument('--stop', action='store_true')
    ap.add_argument('--receipt-dir', type=Path, default=Path('results/e3_prefix_stop_receipts'))
    args = ap.parse_args(argv)
    plan = C.read(args.plan)
    jobs = [j for j in plan['jobs'] if j['length'] == args.length
            and (args.model is None or j['model'].lower() == args.model)
            and (args.partition_out is None or any(Path(p['out']).resolve() == args.partition_out.resolve()
                 for p in j.get('partitions', [])))]
    submission_plans = [C.read(p) for p in args.plan.parent.glob('e3*helper*plan.json')]
    if not jobs:
        raise SystemExit('no matching partitioned jobs')
    try:
        coverage = C.collect(plan, status_only=True)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(json.dumps({'ready': False, 'reason': f'confirmation validation incomplete: {exc}'}, indent=2))
        return
    summaries = []
    for job in jobs:
        try:
            summary = assess_job(job, Path(plan.get('panel', G.OUT_ROOT)), coverage,
                                 partition_out=args.partition_out, submission_plans=submission_plans)
        except (ValueError, FileNotFoundError, KeyError, StopIteration) as exc:
            summary = {'model': job['model'].upper(), 'ready': False, 'reason': str(exc)}
        if args.stop and summary['ready']:
            summary['stop'] = stop_once(summary, args.receipt_dir)
        summaries.append(summary)
    print(json.dumps({'dry_run': not args.stop, 'jobs': summaries}, indent=2))


if __name__ == '__main__':
    main()
