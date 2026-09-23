"""Prospective exhaustive diagnostic; all expectations use the target history law."""
import itertools
import math
from lrwkv_evidence.distance_intervention import tasks as T
from lrwkv_evidence.distance_intervention.diagnostics import evidence_response


def histories(ex):
    return [dict(zip(ex['information_set'], bits)) for bits in itertools.product((0, 1), repeat=4)]


def flip(ex, row):
    if row not in range(4):
        raise ValueError('invalid constraint')
    # The frozen helper uses index only to choose the constraint. Restore the
    # original example index before serialization; no example is regenerated.
    cf, _ = T.counterfactual(dict(ex, index=row))
    cf['index'] = ex['index']
    return cf


def targets(ex, visible):
    feasible = [y for y in ex['support'] if all((y >> i) & 1 == b for i, b in visible.items())]
    if len(feasible) != 1:
        raise ValueError('history must determine a unique valid output')
    return [(feasible[0] >> i) & 1 for i in range(8)]


def meta(ex, cf, visible, row):
    a, b = targets(ex, visible), targets(cf, visible)
    changed = [i for i in range(8) if a[i] != b[i]]
    if len(changed) != 1 or changed[0] in visible:
        raise ValueError('intervention must change exactly one unobserved target')
    i = changed[0]
    return dict(row_index=row, target_coordinate=i, visible=visible,
                original_target=a[i], flipped_target=b[i],
                unaffected_targets=[j for j in range(8) if j not in visible and j != i])


def validate_lp(lp):
    if len(lp) != 8:
        raise ValueError('eight coordinates required')
    for pair in lp:
        if len(pair) != 2 or not all(math.isfinite(x) and x <= 1e-8 for x in pair):
            raise ValueError('invalid binary log probabilities')
        if abs(sum(math.exp(x) for x in pair)-1) > 1e-6:
            raise ValueError('unnormalized probabilities')


def summarize(ex, initial, original, flipped):
    if len(original) != 16 or len(flipped) != 4 or any(len(x) != 16 for x in flipped):
        raise ValueError('all 16 histories and four constraints required')
    for lp in [initial, *original, *(v for row in flipped for v in row)]:
        validate_lp(lp)
    first = sum(-math.log(2)-sum(initial[i])/2 for i in ex['information_set'])
    second = 0.; flipped_error = 0.; responses = []; hist_errors = []
    variants = [flip(ex, r) for r in range(4)]
    for h, visible in enumerate(histories(ex)):
        gold = targets(ex, visible)
        error = -sum(original[h][i][gold[i]] for i in range(8) if i not in visible)
        second += error/16; hist_errors.append(error)
        for r, cf in enumerate(variants):
            m = meta(ex, cf, visible, r)
            response = evidence_response(original[h], flipped[r][h], m)
            flipped_error -= flipped[r][h][m['target_coordinate']][m['flipped_target']]/16
            responses.append(dict(history=h, constraint=r, **response))
    keys = ('mean_correct_conditional_ce', 'response_ce_lower_bound', 'centering_penalty',
            'signed_logit_contrast', 'both_variants_correct', 'signed_probability_shift',
            'unaffected_mean_abs_probability_change')
    means = {k: sum(r[k] for r in responses)/64 for k in keys}
    identity_error = abs(4*means['mean_correct_conditional_ce']-(second+flipped_error)/2)
    if identity_error > 1e-8:
        raise ValueError('symmetrized conditional-error identity failed')
    return dict(first_error=first, second_error=second, information_set_error=first+second,
                flipped_target_error=flipped_error, response=means,
                worst_history_second_error=max(hist_errors),
                worst_pair_ce=max(r['mean_correct_conditional_ce'] for r in responses),
                max_unaffected_change=max(r['unaffected_max_abs_probability_change'] for r in responses),
                identity_error=identity_error, pairs=responses)
