"""Prospective binned error model. No model outputs from target classes enter fit."""
import math
from collections import defaultdict
from . import tasks as T
POSITIONS=('far','middle','near')
CALIBRATION_SEED=202709237
HELDOUT_SEED=202709238
MIN_SUPPORT=8  # distinct source prompts per structural bin; histories are not replicates
SHRINKAGE=8.


def panel(split):
    if split=='calibration':classes=range(15);count=4;seed=CALIBRATION_SEED
    elif split=='heldout':classes=range(15,25);count=8;seed=HELDOUT_SEED
    else:raise ValueError(split)
    return [dict(index=c*100+j,class_id=c,seed=seed,position=p,split=split)
            for c in classes for j in range(count) for p in POSITIONS]


def example(cell):return T.make_example(cell['split'],cell['seed'],cell['index'],cell['class_id'])


def key(feature):return '|'.join(map(str,feature))


def tie(ex,position):return int(T.digest(('dependence_tie_v1',ex['prompt'],position)),16)%2


def metrics(ex,initial,conditional):
    """Exact KL, valid mass and per-output oracle-to-model errors."""
    ys=T.support(ex);result=[]
    for policy in (0,1):
        first=T.public_bases(ex)[policy];second=[i for i in range(8) if i not in first]
        errors={i:-(initial[i][0]+initial[i][1])/2-math.log(2) for i in first}
        for i in second:errors[i]=-sum(conditional[policy][h][i][y[i]] for h,y in enumerate(ys))/16
        logq=[sum(initial[i][y[i]] for i in first)+sum(conditional[policy][h][i][y[i]] for i in second) for h,y in enumerate(ys)]
        kl=-math.log(16)-sum(logq)/16
        if abs(kl-sum(errors.values()))>1e-8:raise ValueError('endpoint/error identity failed')
        result.append(dict(kl=kl,valid_mass=sum(math.exp(v) for v in logq),errors={str(k):v for k,v in errors.items()}))
    return result


def fit(rows):
    expected={(c['index'],c['position']) for c in panel('calibration')}
    if len(rows)!=len(expected) or {(r['cell']['index'],r['cell']['position']) for r in rows}!=expected:
        raise ValueError('incomplete/duplicate calibration panel')
    bins=defaultdict(list);prompts=defaultdict(set);pooled=defaultdict(list);policy_scores=[[],[]]
    for row in rows:
        cell=row['cell']
        if cell not in panel('calibration'):raise ValueError('non-source calibration')
        ex=example(cell)
        for policy in (0,1):
            policy_scores[policy].append(row['metrics'][policy]['kl'])
            for i,feature in T.features(ex,policy,cell['position']):
                value=row['metrics'][policy]['errors'][str(i)]
                if not math.isfinite(value) or value < -1e-9:raise ValueError('invalid calibration error')
                k=key(feature);parent=key((feature[0],feature[3]))
                bins[k].append(value);prompts[k].add(ex['instance_id']);pooled[parent].append(value)
    means={k:sum(v)/len(v) for k,v in pooled.items()}
    fitted={}
    for k,values in bins.items():
        parts=k.split('|');prior=means[key((parts[0],parts[3]))]
        fitted[k]=dict(mean=(sum(values)+SHRINKAGE*prior)/(len(values)+SHRINKAGE),
                       prompts=len(prompts[k]),outputs=len(values))
    return dict(bins=fitted,pooled=means,source_preference=int(sum(policy_scores[1])<sum(policy_scores[0])),
                min_support=MIN_SUPPORT,shrinkage=SHRINKAGE)


def predict(fitted,cell):
    if cell not in panel('heldout'):raise ValueError('undeclared target cell')
    ex=example(cell);scores=[];supported=True;missing=[];heuristic=[]
    for policy in (0,1):
        score=0.;fan=0
        for i,feature in T.features(ex,policy,cell['position']):
            k=key(feature);record=fitted['bins'].get(k)
            if record is None or record['prompts']<MIN_SUPPORT:
                supported=False;missing.append(k)
            score+=record['mean'] if record else fitted['pooled'][key((feature[0],feature[3]))]
            if feature[0]=='deterministic':fan+=feature[1]
        scores.append(score);heuristic.append(fan)
    fallback=tie(ex,cell['position'])
    choice=(int(scores[1]<scores[0]) if scores[0]!=scores[1] else fallback) if supported else fallback
    return dict(cell=cell,scores=scores,supported=supported,unsupported_bins=sorted(set(missing)),
                chosen=choice,dependence_only=fallback,unconditional=fitted['source_preference'],
                fanin_heuristic=int(heuristic[1]<heuristic[0]) if heuristic[0]!=heuristic[1] else fallback)
