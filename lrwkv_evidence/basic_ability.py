"""Collect basic-ability panels with exact per-arm/seed coverage and source hashes.

Run: python -m lrwkv_evidence.basic_ability [--root OUTPUTS] [--out-dir results]
Duplicate IDs fail closed, including identical duplicates; arms are never pooled.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from .evidence import OUTPUTS

DATASETS = {'race': 4887, 'openbookqa': 500, 'mmlu_redux': 5700}
TRACKS = {
    'a29_c6_f2_s14000': ('F2, step 14000', 'fwdce'),
    'a29_c6loop_s9500': ('Loop, step 9500', 'fwdce'),
    'a29_c6loop_s20500': ('Loop, step 20500', 'fwdce'),
    'rel_c1_rwkv7_0p4b': ('Released RWKV-7, 0.4B', 'raw'),
    'rel_c1_rwkv7_1p5b': ('Released RWKV-7, 1.5B', 'raw'),
    'rel_c1_rwkv7_2p9b': ('Released RWKV-7, 2.9B', 'raw'),
}


def audit_file(path, expected_n, arm, seed=42):
    content = path.read_bytes()  # Hash exactly the bytes parsed.
    data = json.loads(content)
    records = data.get('records', [])
    chosen = [r for r in records if r.get('arm') == arm and r.get('seed') == seed]
    ids = Counter(r.get('document_id') for r in chosen)
    expected_ids = {f'{i:06d}' for i in range(expected_n)}
    missing = sorted(expected_ids - set(ids))
    unexpected = sorted(set(ids) - expected_ids, key=str)
    duplicates = {str(k): n for k, n in ids.items() if n > 1}
    errors = []
    if data.get('record_count') != len(records):
        errors.append('record_count disagrees with records length')
    if missing or unexpected or duplicates:
        errors.append('document IDs are not exact unique canonical coverage')
    values = [r.get('metrics', {}).get('correct') for r in chosen]
    valid = all(v in (0, 1) for v in values)
    if not valid:
        errors.append('correct metric is missing or non-binary')
    accuracy = sum(values) / len(values) if values and valid else None
    aggregates = [m for m in data.get('metrics', [])
                  if m.get('arm') == arm and m.get('metric') == 'correct']
    aggregate = aggregates[0] if len(aggregates) == 1 else None
    if aggregate is None:
        errors.append('missing or ambiguous correct aggregate')
    elif (aggregate.get('n_documents') != len(ids) or accuracy is None
          or not math.isclose(aggregate['mean'], accuracy, abs_tol=1e-12)):
        errors.append('aggregate disagrees with recomputed per-item accuracy')
    return dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                checkpoint=data.get('checkpoint'), expected_n=expected_n,
                arm=arm, seed=seed, all_records=len(records),
                selected_records=len(chosen), unique_documents=len(ids),
                excluded_records=len(records)-len(chosen),
                observed_arms=dict(Counter(r.get('arm') for r in records)),
                duplicate_documents=duplicates, missing_document_ids=missing,
                unexpected_document_ids=unexpected, errors=errors,
                status='complete' if not errors else 'incomplete',
                correct_count=int(sum(values)) if values and valid else None,
                accuracy=accuracy if not errors else None,
                reported_ci95=aggregate.get('ci95') if aggregate else None,
                bootstrap=data.get('bootstrap'),
                provenance={k: data.get(k) for k in
                            ('registry_hash', 'profile_sha256', 'condition_profile_hash')})


def collect(root):
    panels = []
    for track, (label, arm) in TRACKS.items():
        for dataset, n in DATASETS.items():
            directory = root / f'e1_{dataset}_{track}'
            files = sorted((directory / 'merged').glob('*.json'))
            row = dict(track=track, label=label, dataset=dataset, expected_n=n, arm=arm, seed=42)
            if len(files) == 1:
                row.update(audit_file(files[0], n, arm))
            else:
                row.update(status='missing' if not files else 'ambiguous', accuracy=None,
                           candidates=[str(p) for p in files],
                           available_shards=len(list(directory.glob('multichoice.*.shard*.json'))))
            panels.append(row)
    return dict(schema='lrwkv_basic_ability_v1',
                generated_at=datetime.now(timezone.utc).isoformat(), source_root=str(root),
                interpretation='Descriptive option-scoring accuracy; fwdce for adapted models, raw for released models. No generation, matched-training, or significance claim. Source provenance is recorded, including unpinned fields; hashes identify observed bytes.',
                panels=panels)


def latex(report):
    rows = {(r['track'], r['dataset']): r for r in report['panels']}
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Model & RACE & OpenBookQA & MMLU-Redux \\',
             r'Unique items & 4887 & 500 & 5700 \\', r'\midrule']
    for track, (label, _) in TRACKS.items():
        cells = [f"{100 * rows[track, ds]['accuracy']:.2f}"
                 if rows[track, ds]['status'] == 'complete' else r'---' for ds in DATASETS]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Basic-ability option-scoring accuracy (\%). RACE and OpenBookQA use validation splits; MMLU-Redux uses the test split. Each reported cell has exact unique item coverage at evaluation seed 42. Adapted checkpoints use \texttt{fwdce}; released checkpoints use \texttt{raw}. Dashes denote unavailable or unverified complete panels. These are descriptive checkpoint comparisons, with no claim of statistical significance or matched training budget.}',
              r'\label{tab:basic-ability-panels}', r'\end{table}', '']
    return '\n'.join(lines)



def paired_contrast(left, right, *, replicates=10000, bootstrap_seed=20270921, batch_size=128):
    """Paired document bootstrap of binary accuracies, conditional on checkpoints.

    Inputs map unique document IDs to correctness; ID sets must match exactly.
    Sampling is bounded to batch_size * n integer indices, rather than B * n.
    """
    import numpy as np
    if not left or set(left) != set(right):
        raise ValueError('paired contrast requires equal, nonempty document ID sets')
    if replicates < 1 or batch_size < 1:
        raise ValueError('replicates and batch_size must be positive')
    if any(v not in (0, 1) for v in (*left.values(), *right.values())):
        raise ValueError('paired correctness must be binary')
    ids = sorted(left)
    a = np.array([left[k] for k in ids], dtype=np.int8)
    b = np.array([right[k] for k in ids], dtype=np.int8)
    differences = a - b
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, len(ids), size=(stop-start, len(ids)))
        samples[start:stop] = differences[indices].mean(axis=1)
    return dict(n=len(ids), left_correct=int(a.sum()), right_correct=int(b.sum()),
                delta_percentage_points=float(differences.sum()) * 100 / len(ids),
                ci95_percentage_points=(np.quantile(samples, [.025, .975], method='linear') * 100).tolist(),
                left_only_correct=int(((a == 1) & (b == 0)).sum()),
                right_only_correct=int(((a == 0) & (b == 1)).sum()),
                both_correct=int(((a == 1) & (b == 1)).sum()),
                both_incorrect=int(((a == 0) & (b == 0)).sum()),
                bootstrap=dict(replicates=replicates, seed=bootstrap_seed,
                               unit='paired_document', method='percentile',
                               quantile_method='linear', generator='PCG64',
                               batch_size=batch_size, numpy_version=np.__version__))


def _paired_values(row):
    content = Path(row['path']).read_bytes()
    if hashlib.sha256(content).hexdigest() != row['sha256']:
        raise ValueError('source changed after auditing: ' + row['path'])
    records = [r for r in json.loads(content)['records']
               if r.get('arm') == row['arm'] and r.get('seed') == row['seed']]
    values = {r['document_id']: r['metrics']['correct'] for r in records}
    if len(values) != len(records):
        raise ValueError('duplicate document IDs in paired source')
    return values


def collect_contrasts(report):
    rows = {(r['track'], r['dataset']): r for r in report['panels']}
    contrasts = []
    for dataset in DATASETS:
        left = rows['a29_c6_f2_s14000', dataset]
        right = rows['rel_c1_rwkv7_2p9b', dataset]
        row = dict(dataset=dataset, direction='F2 step 14000 minus released RWKV-7 2.9B',
                   sources=[{k: r.get(k) for k in ('track', 'path', 'sha256', 'arm', 'seed', 'status')}
                            for r in (left, right)])
        if left['status'] != 'complete' or right['status'] != 'complete':
            row.update(status='unavailable', reason='both complete audited panels required')
        else:
            row.update(paired_contrast(_paired_values(left), _paired_values(right)), status='complete')
        contrasts.append(row)
    return dict(schema='lrwkv_basic_ability_contrasts_v1', generated_at=report['generated_at'],
                interpretation='Paired document resampling measures item-sampling uncertainty conditional on these fixed checkpoints and this scoring protocol. Pretraining and training budgets are unmatched; these are descriptive differences, not causal effects or training-seed uncertainty. No multiple-comparison adjustment.',
                contrasts=contrasts)


def contrasts_latex(report):
    names = dict(race='RACE', openbookqa='OpenBookQA', mmlu_redux='MMLU-Redux')
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Dataset & $n$ & $\Delta$ (pp; 95\% CI) & F2 only & RWKV only \\', r'\midrule']
    for r in report['contrasts']:
        if r['status'] != 'complete':
            lines.append(names[r['dataset']] + r' & --- & --- & --- & --- \\')
            continue
        lo, hi = r['ci95_percentage_points']
        lines.append(f"{names[r['dataset']]} & {r['n']} & {r['delta_percentage_points']:+.2f} [{lo:+.2f}, {hi:+.2f}] & {r['left_only_correct']} & {r['right_only_correct']}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{F2 step 14000 minus released RWKV-7 2.9B: accuracy differences in percentage points (pp), with 95\% paired-document percentile bootstrap intervals (10000 resamples; seed 20270921). The final columns count documents answered correctly only by the indicated model. Pairing uses exact document IDs at evaluation seed 42. Intervals describe item-sampling uncertainty for fixed checkpoints; pretraining and training budgets are unmatched, so differences are not causal effects.}',
              r'\label{tab:basic-ability-contrasts}', r'\end{table}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUTPUTS)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    report = collect(args.root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'basic_ability_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.out_dir / 'basic_ability_table.tex').write_text(latex(report))
    contrasts = collect_contrasts(report)
    (args.out_dir / 'basic_ability_contrasts.json').write_text(json.dumps(contrasts, indent=2) + '\n')
    (args.out_dir / 'basic_ability_contrasts.tex').write_text(contrasts_latex(contrasts))
    print(json.dumps(dict(Counter(r['status'] for r in report['panels'])), indent=2))


if __name__ == '__main__':
    main()
