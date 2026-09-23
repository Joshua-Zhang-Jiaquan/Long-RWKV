"""Exact valid-support endpoint audit of a binary model's normalized conditionals."""
import math
from .tasks import partitions


def logsumexp(xs):
    m=max(xs)
    return m+math.log(sum(math.exp(x-m) for x in xs))


def audit(ex,predict_logprobs):
    """Callback receives visible bits only and returns eight normalized log pairs.

    Cache is shared across policies for audit speed, never a latency measurement.
    The model callback must construct input without oracle values or a gold draw.
    """
    support=ex['support'];count=len(support)
    if not support or len(set(support))!=count or any(type(y) is not int or not 0<=y<256 for y in support):
        raise ValueError('invalid support')
    cache={}
    def predict(visible):
        key=tuple(sorted(visible.items()))
        if key not in cache:
            lp=predict_logprobs(dict(visible))
            if len(lp)!=8 or any(len(row)!=2 or not all(math.isfinite(x) for x in row) or abs(logsumexp(row))>1e-6 for row in lp):
                raise ValueError('predictor must return finite normalized binary log probabilities')
            cache[key]=lp
        return cache[key]
    results=[]
    for method,groups in partitions(ex).items():
        logq=[];dependence=0.;estimation=0.
        for y in support:
            visible={};path=0.
            for group in groups:
                feasible=[z for z in support if all(((z>>i)&1)==b for i,b in visible.items())]
                target=[(y>>i)&1 for i in group]
                joint_count=sum(all(((z>>i)&1)==b for i,b in zip(group,target)) for z in feasible)
                logjoint=math.log(joint_count/len(feasible));logmarginal=0.;logmodel=0.
                lp=predict(visible)
                for i,b in zip(group,target):
                    probability=sum(((z>>i)&1)==b for z in feasible)/len(feasible)
                    logmarginal+=math.log(probability);logmodel+=lp[i][b]
                dependence+=(logjoint-logmarginal)/count
                estimation+=(logmarginal-logmodel)/count
                path+=logmodel;visible.update(zip(group,target))
            logq.append(path)
        kl=-math.log(count)-sum(logq)/count;logvalid=logsumexp(logq)
        if abs(kl-dependence-estimation)>1e-8 or logvalid>1e-8 or min(dependence,estimation,kl)<-1e-8:
            raise ValueError('invalid endpoint decomposition or normalization')
        results.append(dict(method=method,joint_kl_nats=kl,dependence_nats=dependence,
                            estimation_nats=estimation,exact_valid_mass=math.exp(logvalid),
                            log_valid_mass=logvalid,within_valid_kl_nats=kl+logvalid,
                            support_log_probabilities=logq))
    return dict(family=ex['family'],instance_id=ex['instance_id'],support=support,
                results=results,unique_audit_calls=len(cache),scope='Exact fixed-condition endpoint law; cached calls are not latency.')
