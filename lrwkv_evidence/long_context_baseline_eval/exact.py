"""Exact causal law using conditional probabilities from actual cached steps."""
import math
from lrwkv_evidence.long_context_eval.exact import logsumexp


def causal_audit(ex, cached_path_logprobs):
    support = ex['support']
    if not support or len(set(support)) != len(support):
        raise ValueError('invalid support')
    logq = []
    for y in support:
        if type(y) is not int or not 0 <= y < 256:
            raise ValueError('invalid answer')
        # The callback feeds only earlier bits at each cached causal step.
        lp = cached_path_logprobs(y)
        if len(lp) != 8 or any(len(row) != 2 or any(not math.isfinite(v) for v in row)
                              or abs(logsumexp(row)) > 1e-6 for row in lp):
            raise ValueError('invalid normalized causal conditionals')
        logq.append(sum(lp[i][(y >> i) & 1] for i in range(8)))
    kl = -math.log(len(support)) - sum(logq) / len(support)
    logvalid = logsumexp(logq)
    if kl < -1e-8 or logvalid > 1e-8 or kl + logvalid < -1e-8:
        raise ValueError('invalid causal endpoint law')
    result = dict(method='cached_causal', joint_kl_nats=kl, dependence_nats=0., estimation_nats=kl,
                  log_valid_mass=logvalid, exact_valid_mass=math.exp(logvalid),
                  within_valid_kl_nats=kl + logvalid, support_log_probabilities=logq)
    return dict(family=ex['family'], instance_id=ex['instance_id'], support=support, results=[result],
                scope='Exact support likelihood from cached causal steps; teacher forcing is used only to score target paths.')
