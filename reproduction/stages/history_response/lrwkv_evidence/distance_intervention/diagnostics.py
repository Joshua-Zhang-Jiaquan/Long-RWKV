"""Exact stage errors and paired binary-evidence response diagnostics."""
import math
from .tasks import partitions


def softplus(x):
    return max(x, 0.) + math.log1p(math.exp(-abs(x)))


def stage_errors(ex, predict):
    first, second = partitions(ex)['information_set']
    initial = predict({})
    first_error = sum(-math.log(2) - .5*(initial[i][0]+initial[i][1]) for i in first)
    second_error = 0.
    for y in ex['support']:
        visible = {i:(y>>i)&1 for i in first}
        lp = predict(visible)
        second_error -= sum(lp[i][(y>>i)&1] for i in second) / len(ex['support'])
    return dict(first_round_estimation_nats=first_error,
                second_round_estimation_nats=second_error,
                second_round_mean_bit_ce=second_error/len(second),
                total_estimation_nats=first_error+second_error)


def evidence_response(original, flipped, meta):
    i = meta['target_coordinate']; a = meta['original_target']; b = meta['flipped_target']
    # Orient logits so original target=0 and flipped target=1.
    sign = 1 if b == 1 else -1
    l0 = sign*(original[i][1]-original[i][0]); l1 = sign*(flipped[i][1]-flipped[i][0])
    delta = l1-l0
    ce = -(original[i][a]+flipped[i][b])/2
    floor = softplus(-delta/2)
    gap = ce-floor
    if gap < -1e-9: raise ValueError('binary evidence-response inequality violated')
    unaffected = [abs(math.exp(flipped[j][1])-math.exp(original[j][1])) for j in meta['unaffected_targets']]
    return dict(signed_logit_contrast=delta, common_logit_midpoint=(l0+l1)/2,
                mean_correct_conditional_ce=ce, response_ce_lower_bound=floor,
                centering_penalty=max(0.,gap),
                signed_probability_shift=sign*(math.exp(flipped[i][1])-math.exp(original[i][1])),
                both_variants_correct=l0 < 0 < l1,
                unaffected_mean_abs_probability_change=sum(unaffected)/len(unaffected),
                unaffected_max_abs_probability_change=max(unaffected),
                original_target_logprobs=original[i], flipped_target_logprobs=flipped[i])
