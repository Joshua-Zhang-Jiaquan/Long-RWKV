"""Paired base-problem inference; never treat lengths or timing draws as samples.

This module does not authorize final evaluation or turn development data into
test data. The caller must validate raw shards against a frozen test manifest.
Training seeds are analyzed separately, rather than pooled as extra problems.
"""
import math
import numpy as np

CONTEXTS = ((None, 'evidence_only'),) + tuple(
    (length, position) for length in (1024, 4096, 16384)
    for position in ('far', 'middle', 'near'))
METRICS = ('joint_kl_nats', 'exact_valid_mass')
CONTRASTS = ('one', 'pair_preserving_halves')


def bootstrap_mean(values, *, seed=20270925, repeats=10000):
    """Pointwise percentile CI over paired base-problem differences."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('requires at least two finite paired observations')
    if type(repeats) is not int or repeats < 100:
        raise ValueError('insufficient bootstrap draws')
    generator = np.random.default_rng(seed)
    # Bound temporary storage when a larger panel is used.
    means = np.concatenate([
        x[generator.integers(0, len(x), size=(min(512, repeats-start), len(x)))].mean(axis=1)
        for start in range(0, repeats, 512)])
    lower, upper = np.quantile(means, [.025, .975])
    return dict(base_problems=len(x), mean=float(x.mean()),
                standard_deviation=float(x.std(ddof=1)),
                pointwise_95_percentile_interval=[float(lower), float(upper)],
                bootstrap_seed=seed, bootstrap_repeats=repeats)


def analyze(conditions, indices, *, bootstrap_seed=20270925, repeats=10000):
    """Analyze one checkpoint on a complete balanced test panel.

    Inputs follow the exact collector's condition schema. Positive differences
    mean information-set improvement: comparator KL minus information-set KL,
    or information-set validity minus comparator validity. Each bootstrap unit
    is a base index; all positions and lengths stay paired inside that unit.
    """
    indices = list(indices)
    if len(indices) < 2 or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('invalid frozen base indices')
    expected = {(family, index, length, position)
                for family in ('systematic', 'paired_parity') for index in indices
                for length, position in CONTEXTS}
    rows = {}
    for row in conditions:
        key = (row['family'], row['index'], row['requested_context_tokens'], row['position'])
        if key in rows:
            raise ValueError('duplicate condition')
        methods = {r['method']: r for r in row['exact']['results']}
        if len(methods) != len(row['exact']['results']) or not set(CONTRASTS + ('information_set',)) <= methods.keys():
            raise ValueError('missing or duplicate primary policy')
        for method in CONTRASTS + ('information_set',):
            values = [methods[method][metric] for metric in METRICS]
            if not all(math.isfinite(v) for v in values) or values[0] < -1e-7 or not 0 <= values[1] <= 1+1e-8:
                raise ValueError('invalid endpoint metric')
        rows[key] = methods
    if set(rows) != expected:
        raise ValueError('missing or unexpected frozen condition; no complete-case filtering')

    def differences(family, length, position, comparator, metric):
        sign = 1 if metric == 'joint_kl_nats' else -1
        return np.asarray([sign * (rows[family, i, length, position][comparator][metric]
                                  - rows[family, i, length, position]['information_set'][metric])
                           for i in indices])

    cells = []
    interactions = []
    for family in ('systematic', 'paired_parity'):
        for comparator in CONTRASTS:
            for metric in METRICS:
                short = differences(family, None, 'evidence_only', comparator, metric)
                for length, position in CONTEXTS:
                    delta = differences(family, length, position, comparator, metric)
                    identity = dict(family=family, comparator=comparator, metric=metric,
                                    context_tokens=length, position=position)
                    cells.append(dict(**identity, **bootstrap_mean(delta, seed=bootstrap_seed, repeats=repeats)))
                    if length is not None:
                        interactions.append(dict(**identity, **bootstrap_mean(delta-short, seed=bootstrap_seed, repeats=repeats)))
    # This proposed primary estimand must be fixed in the test manifest first.
    # Average positions WITHIN each base problem before resampling problems.
    primary = []
    for comparator in CONTRASTS:
        for metric in METRICS:
            by_problem = np.stack([differences('paired_parity', 16384, pos, comparator, metric)
                                   for pos in ('far', 'middle', 'near')]).mean(axis=0)
            primary.append(dict(family='paired_parity', context_tokens=16384,
                                position='equal_weight_mean_of_three_positions',
                                comparator=comparator, metric=metric,
                                **bootstrap_mean(by_problem, seed=bootstrap_seed, repeats=repeats)))
    return dict(primary_estimands=primary, cells=cells, length_interactions=interactions,
                inference_unit='base problem within one fixed trained checkpoint',
                limitations=['Intervals are pointwise, not simultaneous across cells or outcomes.',
                             'Training-seed variability is separate; do not pool checkpoints as independent test problems.',
                             'Exact endpoint probabilities remove rollout sampling noise, not held-out problem variability.',
                             'A positive long-context gain alone does not show improved retention; inspect the paired length interaction.',
                             'Cost and the restricted budget certificate require a separate measured-cost analysis.'])
