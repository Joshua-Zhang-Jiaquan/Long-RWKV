"""Fail-closed collection of paired F2/R0 confirmation receipts.

--status emits coverage only. Complete reports condition on the generated cells;
unsupported cells remain explicit records with no accuracy assigned. CIs are
computed by paired within-cell resampling with fixed equal-cell/family/length
weights; intervals condition on the generated cells and the executed decodes.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from lrwkv_evidence.parallel_io import read_json_parallel

MODELS = ('F2', 'R0')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def item_id(inst):
    return f"{inst['cell_id']}__{inst['data_seed']}__{inst['instance_index']}"


def paired_bootstrap(cells, *, draws=10000, seed=20270921):
    """Fixed-cell stratified paired percentile intervals with bounded memory.

    Each cell's paired (F2,R0) indicators have four possible outcomes. Sampling
    their joint multinomial counts is exactly equivalent to resampling example
    pairs within that cell. Cells and the decoder seed are not resampled.
    """
    import numpy as np
    if draws < 1 or not cells:
        raise ValueError('bootstrap requires positive draws and nonempty cells')
    groups = defaultdict(list)
    for cell in cells:
        groups[(cell['length'], cell['family'])].append(cell)
    rng = np.random.default_rng(seed)
    overall = np.zeros((draws, 3), dtype=np.float64)
    panels = []
    def intervals(values):
        q = np.quantile(values, [0.025, 0.975], axis=0)
        return {m: [float(q[0, i]), float(q[1, i])] for i, m in enumerate(('F2', 'R0', 'difference'))}
    for (length, family), group in sorted(groups.items()):
        values = np.zeros_like(overall)
        for cell in sorted(group, key=lambda c: c['cell_id']):
            n = cell['n']
            f, r, both = cell['F2_only'], cell['R0_only'], cell['both_correct']
            neither = n - f - r - both
            if n <= 0 or any(type(k) is not int or k < 0 for k in (f, r, both, neither)):
                raise ValueError('invalid paired contingency counts')
            sampled = rng.multinomial(n, np.array([neither, f, r, both]) / n, size=draws)
            scale = 1 / (n * len(group))
            values[:, 0] += (sampled[:, 1] + sampled[:, 3]) * scale
            values[:, 1] += (sampled[:, 2] + sampled[:, 3]) * scale
            values[:, 2] += (sampled[:, 1] - sampled[:, 2]) * scale
        overall += values / len(groups)
        panels.append({'length': length, 'family': family, 'ci95': intervals(values)})
    return {'method': 'paired within-cell multinomial percentile bootstrap',
        'draws': draws, 'seed': seed, 'confidence': 0.95,
        'weighting': 'equal family/length panels; equal supported cells within each panel',
        'resampling_unit': 'paired examples within each fixed cell',
        'scope': 'conditional on generated cells, fixed checkpoints and executed decoder seeds; no execution or unsupported-cell uncertainty',
        'degeneracy_note': 'All-zero or constant paired outcomes can give zero-width percentile intervals; this is not proof of equivalence or zero population error.',
        'family_length': panels, 'overall_ci95': intervals(overall)}


def collect(plan, *, status_only=False, expected_cells=324, expected_supported=264,
            expected_unsupported=60, instances_per_cell=200):
    from . import grid324 as G, runner as R
    file_hashes = {}
    def verify_sources(mapping):
        if not mapping:
            raise ValueError('missing frozen source hashes')
        for name, claimed in mapping.items():
            if name not in file_hashes:
                file_hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
            if file_hashes[name] != claimed:
                raise ValueError(f'frozen source hash mismatch: {name}')
    panel = Path(plan.get('panel', G.OUT_ROOT))
    manifest = read(panel / 'grid_manifest.json')
    if manifest.get('split') != 'iclr2027_grid' or manifest.get('partial_run', False):
        raise ValueError('not a complete confirmation panel')
    cells = [read(p) for p in sorted(panel.glob('*.json')) if p.name != 'grid_manifest.json']
    cells = [c for c in cells if 'cell_id' in c]
    if len(cells) != expected_cells or len({c['cell_id'] for c in cells}) != expected_cells:
        raise ValueError('missing or duplicate canonical cells')
    supported = [c for c in cells if c['status'] == 'ok']
    unsupported = [c for c in cells if c['status'] != 'ok']
    if len(supported) != expected_supported or len(unsupported) != expected_unsupported:
        raise ValueError('canonical supported/unsupported counts differ')
    expected = {}
    by_cell = {}
    for cell in supported:
        if len(cell['instances']) != instances_per_cell or cell.get('total_instances') != instances_per_cell:
            raise ValueError('incomplete canonical cell')
        by_cell[cell['cell_id']] = cell
        for inst in cell['instances']:
            key = item_id(inst)
            if inst['cell_id'] != cell['cell_id'] or key in expected:
                raise ValueError('duplicate or misfiled canonical instance')
            expected[key] = inst
    jobs = {}
    for job in plan['jobs']:
        key = (job['model'].upper(), int(job['length']))
        if key in jobs or key[0] not in MODELS:
            raise ValueError('duplicate or unknown model/length job')
        jobs[key] = job
    lengths = sorted({c['history_length'] for c in cells})
    if set(jobs) != {(m, n) for m in MODELS for n in lengths}:
        raise ValueError('plan must cover every model and length')
    execution_sources = []
    for (model, length), job in jobs.items():
        eligible = {c['cell_id'] for c in supported if c['history_length'] == length}
        partitions = job.get('partitions')
        if partitions is None:
            execution_sources.append((model, length, job, {'out': job['out'], 'cell_ids': sorted(eligible)}))
            continue
        covered, outputs = set(), set()
        if not any(Path(p['out']).resolve() == Path(job['out']).resolve() for p in partitions):
            raise ValueError('partition plan omits primary output')
        for part in partitions:
            assigned = set(part['cell_ids'])
            if part.get('schedule_policy') not in (None, 'accepted_subset_of_recorded_schedule'):
                raise ValueError('unknown partition schedule policy')
            out = str(Path(part['out']).resolve())
            if (not assigned or len(assigned) != len(part['cell_ids'])
                    or assigned & covered or not assigned <= eligible or out in outputs):
                raise ValueError('partition overlap, duplicate output, or invalid cell assignment')
            covered.update(assigned)
            outputs.add(out)
            execution_sources.append((model, length, job, part))
        if covered != eligible:
            raise ValueError('partition assignments do not cover all supported cells')
    rows = {m: {} for m in MODELS}
    partition_coverage = []
    for model, length, job, part in execution_sources:
        out = Path(part['out'])
        assigned = set(part['cell_ids'])
        eligible_cells = {c['cell_id'] for c in supported if c['history_length'] == length}
        helper = 'partitions' in job and out.resolve() != Path(job['out']).resolve()
        source_status = {'model': model, 'length': length, 'out': str(out),
                         'assigned_cells': len(assigned), 'schedule_policy': part.get('schedule_policy', 'exact_schedule'),
                         'used_receipts': 0,
                         'unused_receipts_outside_assignment': 0}
        partition_coverage.append(source_status)
        sidecars = {int(p.stem.removeprefix('rank')): read(p)
                    for p in (out / 'helper_schedule').glob('rank*.json')}
        cache_parity = {digest(read(p)): read(p) for p in (out / 'cache_parity').glob('*.json')}
        provenance = [read(p) for p in sorted(out.glob('provenance_rank*.json'))]
        hashes = {digest(p): p for p in provenance}
        if len(hashes) > 1:
            raise ValueError('mixed run provenance in one model/length output')
        for p in provenance:
            verify_sources(p.get('sources') or p.get('dependencies'))
            if p.get('mode') != 'confirm' or p.get('split') != 'iclr2027_grid':
                raise ValueError('development receipts cannot enter confirmation')
            if model == 'F2' and len(p.get('configs', [])) != 1:
                raise ValueError('confirmation requires exactly one frozen F2 decoder')
            if p.get('length_filter') not in (None, length):
                raise ValueError('wrong length provenance')
        # F2 stores config/item.json; R0 stores item.json. Restrict these paths
        # so unsupported copies and manifests can never become scored rows.
        paths = out.glob('*/*__*__*.json') if model == 'F2' else out.glob('*__*__*.json')
        for path, _raw, row in read_json_parallel(paths):
            key = f"{row.get('cell_id')}__{row.get('data_seed')}__{row.get('instance_index')}"
            if key not in expected or by_cell[row['cell_id']]['history_length'] != length:
                raise ValueError(f'unknown or unsupported outcome: {path}')
            if row['cell_id'] not in assigned:
                source_status['unused_receipts_outside_assignment'] += 1
                continue
            if helper:
                sidecar = sidecars.get(row.get('rank'))
                primary_cells = next(set(p['cell_ids']) for p in job['partitions']
                                     if Path(p['out']).resolve() == Path(job['out']).resolve())
                scheduled = set((sidecar or {}).get('helper_cells', []))
                schedule_matches = (assigned <= scheduled if part.get('schedule_policy') ==
                                    'accepted_subset_of_recorded_schedule' else assigned == scheduled)
                source_status['scheduled_cells'] = len(scheduled)
                if (not sidecar or sidecar.get('scope') != 'partial_confirmation_helper_only'
                        or sidecar.get('length') != length or sidecar.get('model', '').upper() != model
                        or Path(sidecar.get('helper_out', '')).resolve() != out.resolve()
                        or Path(sidecar.get('primary_out', '')).resolve() != Path(job['out']).resolve()
                        or not schedule_matches
                        or len(scheduled) != len(sidecar.get('helper_cells', []))
                        or not scheduled <= eligible_cells
                        or set(sidecar.get('primary_cells', [])) != primary_cells
                        or sidecar.get('instances_per_cell') != instances_per_cell
                        or sidecar.get('expected_helper_items') != len(scheduled) * instances_per_cell
                        or sidecar.get('rank') != row.get('rank')
                        or not isinstance(sidecar.get('world_size'), int)
                        or not 0 <= row.get('rank', -1) < sidecar['world_size']
                        or len(sidecar.get('wrapper_sha256', '')) != 64
                        or sidecar.get('wrapper_sha256') != part.get('helper_wrapper_sha256')
                        or not sidecar.get('qualified_sources')):
                    raise ValueError('missing or incompatible helper orchestration sidecar')
            if helper:
                verify_sources(sidecar['qualified_sources'])
            if key in rows[model]:
                raise ValueError(f'duplicate outcome: {key}')
            inst = expected[key]
            prov = hashes.get(row.get('provenance_sha256'))
            if (prov is None or row.get('status') != 'ok' or type(row.get('correct')) is not bool
                    or row.get('prompt_hash_verified') is not True or row.get('split') != 'iclr2027_grid'
                    or row.get('input_ids_sha256') != inst['input_ids_sha256']):
                raise ValueError(f'invalid measured receipt: {path}')
            answers = inst.get('answers')
            span = inst.get('target_span')
            family = G.FAMILY_MAP[by_cell[row['cell_id']]['family']]
            answer = R.extract_answer(row.get('text', ''), family)
            correct = len(answers or []) == 1 and answer == answers[0]
            if (not answers or row.get('gold') != answers or row.get('prediction') != answer
                    or row.get('parse_ok') is not (answer is not None)
                    or row['correct'] is not correct or row.get('joint_exact') is not correct
                    or not span or not isinstance(row.get('output_ids'), list)
                    or len(row['output_ids']) != span[1] - span[0]):
                raise ValueError('canonical gold, parsed outcome, or target length mismatch')
            if model == 'F2' and prov.get('reuse_unchanged_logits'):
                parity = cache_parity.get(row.get('logit_reuse_parity_sha256'))
                if (not parity or parity.get('qualified') is not True
                        or parity.get('provenance_sha256') != row['provenance_sha256']
                        or parity.get('rank') != row.get('rank')
                        or parity.get('config', {}).get('sampler') != row.get('config', {}).get('sampler')):
                    raise ValueError('missing or mismatched partition cache parity')
                runs = parity.get('runs', {})
                if (runs.get('cached', {}).get('output_ids') != runs.get('uncached', {}).get('output_ids')
                        or runs.get('cached', {}).get('commits_per_step') != runs.get('uncached', {}).get('commits_per_step')):
                    raise ValueError('cache parity outcome mismatch')
            if model == 'F2' and row.get('config') not in prov.get('configs', []):
                raise ValueError('F2 outcome decoder is not in provenance')
            if model == 'R0' and row.get('policy') != prov.get('policy'):
                raise ValueError('R0 outcome policy differs from provenance')
            if type(row.get('actual_nfe')) is not int or row['actual_nfe'] < 1:
                raise ValueError('invalid actual NFE')
            for k in ('wall_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes'):
                if type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0:
                    raise ValueError(f'invalid resource metric {k}')
            # Keep only sufficient summary fields, never generated text or
            # per-step logits/cache traces from all 105,600 receipts.
            source_status['used_receipts'] += 1
            rows[model][key] = {k: row[k] for k in ('cell_id', 'correct', 'actual_nfe',
                'wall_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}
    coverage = {'schema': 1, 'complete': all(len(rows[m]) == len(expected) for m in MODELS),
        'declared_cells': len(cells), 'supported_cells': len(supported),
        'excluded_cells': len(unsupported), 'expected_items_per_model': len(expected),
        'observed_items': {m: len(rows[m]) for m in MODELS},
        'partitions': partition_coverage,
        'missing_items': {m: len(expected) - len(rows[m]) for m in MODELS},
        'unsupported_cells': [{k: c.get(k) for k in ('cell_id', 'family', 'history_length', 'status', 'unsupported_reason')}
                              for c in unsupported]}
    if status_only:
        return {'report_kind': 'coverage_only', **coverage}
    if not coverage['complete']:
        raise ValueError(f"incomplete confirmation; missing {coverage['missing_items']}")
    family_rows = []
    paired_cells = []
    for cell in supported:
        keys = [item_id(i) for i in cell['instances']]
        scores = {m: statistics.mean(rows[m][k]['correct'] for k in keys) for m in MODELS}
        paired_cells.append({'cell_id': cell['cell_id'], 'family': cell['family'],
            'length': cell['history_length'], 'n': len(keys), **scores,
            'difference': scores['F2'] - scores['R0'],
            'F2_only': sum(rows['F2'][k]['correct'] and not rows['R0'][k]['correct'] for k in keys),
            'R0_only': sum(rows['R0'][k]['correct'] and not rows['F2'][k]['correct'] for k in keys),
            'both_correct': sum(rows['F2'][k]['correct'] and rows['R0'][k]['correct'] for k in keys)})
    families = sorted({c['family'] for c in cells})
    for length in lengths:
        for family in families:
            group = [c for c in paired_cells if c['length'] == length and c['family'] == family]
            excluded = sum(c['history_length'] == length and c['family'] == family for c in unsupported)
            family_rows.append({'length': length, 'family': family, 'supported_cells': len(group),
                'excluded_cells': excluded, 'items': sum(c['n'] for c in group),
                **{m: statistics.mean(c[m] for c in group) if group else None for m in (*MODELS, 'difference')}})
    if any(r['F2'] is None for r in family_rows):
        raise ValueError('cannot compute primary family/length macro with an empty family')
    resources = {}
    for model in MODELS:
        resources[model] = {}
        for length in lengths:
            group = [r for k, r in rows[model].items() if by_cell[r['cell_id']]['history_length'] == length]
            walls = [r['wall_seconds'] for r in group]
            resources[model][str(length)] = {'n': len(group),
                'nfe_mean': statistics.mean(r['actual_nfe'] for r in group),
                'nfe_median': statistics.median(r['actual_nfe'] for r in group),
                'wall_seconds_mean': statistics.mean(walls),
                'wall_seconds_median': statistics.median(walls),
                'wall_seconds_p95': statistics.quantiles(walls, n=100, method='inclusive')[94] if len(walls) > 1 else walls[0],
                'wall_quantile_method': 'inclusive linear interpolation at index (n-1)*p in sorted observations',
                'wall_seconds_sum': sum(walls),
                'peak_allocated_bytes_max': max(r['peak_allocated_bytes'] for r in group),
                'peak_reserved_bytes_max': max(r['peak_reserved_bytes'] for r in group)}
    return {'report_kind': 'complete_conditional_comparison', **coverage,
        'estimand': 'equal length, equal family, equal supported cell within each family/length; conditional on generated support',
        'family_length': family_rows, 'cell_scores': paired_cells,
        'overall': {m: statistics.mean(r[m] for r in family_rows) for m in (*MODELS, 'difference')},
        'uncertainty': paired_bootstrap(paired_cells), 'resources': resources,
        'resource_scope': ('Descriptive per-request elapsed times, with asymmetric warmup: R0 first requests include JIT compilation, '
                           'whereas F2 cache-parity qualification warms execution before measured calls. '
                           'R0 uses uncached AR prefix recomputation. These timings do not support a rigorous speedup claim '
                           'or an optimized AR speed comparison; request-time sums are not campaign elapsed time.')}


def render_tex(summary):
    if not summary.get('complete') or summary.get('report_kind') != 'complete_conditional_comparison':
        raise ValueError('coverage-only reports cannot render an accuracy table')
    lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{@{}llrrrr@{}}',
        r'\toprule', r'Length & Family & Cells & F2 & R0 & $\Delta$ [95\% CI]\\', r'\midrule']
    names = {'associative_recall': 'Recall', 'overwrite_delayed_query': 'Overwrite', 'code_dataflow': 'Dataflow'}
    ci_by_panel = {(r['length'], r['family']): r['ci95']['difference']
                   for r in summary['uncertainty']['family_length']}
    for r in summary['family_length']:
        low, high = ci_by_panel[(r['length'], r['family'])]
        lines.append(f"{r['length']//1024}K & {names.get(r['family'], r['family']).replace('_', ' ')} & "
            f"{r['supported_cells']}/{r['supported_cells']+r['excluded_cells']} & "
            f"{100*r['F2']:.3f} & {100*r['R0']:.3f} & {100*r['difference']:+.3f} [{100*low:+.3f},{100*high:+.3f}]" + r'\\')
    o = summary['overall']
    low, high = summary['uncertainty']['overall_ci95']['difference']
    lines += [r'\midrule', f"Overall & Conditional macro & {summary['supported_cells']}/{summary['declared_cells']} & "
        f"{100*o['F2']:.3f} & {100*o['R0']:.3f} & {100*o['difference']:+.3f} [{100*low:+.3f},{100*high:+.3f}]" + r'\\',
        r'\bottomrule', r'\end{tabular}',
        r'\caption{Long-context accuracy (\%) conditional on generated support. Cells gives supported/declared counts. '
        + f"All {summary['excluded_cells']} unsupported cells remain excluded, not scored as zero. "
        + r'The macro weights lengths, families, and supported cells equally at their respective levels. Differences and intervals are percentage points. Paired within-cell percentile bootstrap intervals use 10,000 draws; they condition on the generated cells and executed decodes. Zero-width intervals are not evidence of equivalence.}',
        r'\label{tab:e3-confirmation-conditional}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('results/e3_confirmation_plan.json'))
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('results/e3_confirmation_summary.json'))
    args = parser.parse_args()
    result = collect(read(args.plan), status_only=args.status)
    if args.status:
        print(json.dumps(result, indent=2))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        args.out.with_suffix('.tex').write_text(render_tex(result))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
