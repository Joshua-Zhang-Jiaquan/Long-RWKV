"""Exact small-system checks; no claim of a general neural streaming lower bound.

An online walker retains the active node only, cannot cache off-path records,
and can update its active node during a scan. History is read-only. Each scan
is one directional traversal, not a forward/backward pair. Distinct chain
edges occur once; distractors cannot change chain nodes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import permutations, combinations
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def scan_chain(order, *, alternating=False, cache_records=0):
    """Simulate routing through opaque key/value records in chain order.

    With cache_records>0, off-path seen records enter a FIFO cache; this is a
    different algorithm, included to falsify transfer of the no-cache theorem.
    Return scan count and actually inspected records, stopping at target.
    """
    order=tuple(order); h=len(order)
    if sorted(order)!=list(range(h)) or h<1 or cache_records<0:
        raise ValueError('expected nonempty permutation and nonnegative cache')
    # Opaque node labels prevent the walker from obtaining a successor by
    # incrementing a known chain index. Only the currently scanned/cached record
    # exposes its successor. The construction's oracle order is not passed to
    # the traversal logic.
    nodes=[hashlib.sha256(f'node:{j}'.encode()).hexdigest() for j in range(h+1)]
    records=tuple((nodes[j],nodes[j+1]) for j in order)
    active=nodes[0]; completed=0; scans=0; inspected=0; cache=[]
    while completed<h:
        scans+=1
        traversal=records[::-1] if alternating and scans%2==0 else records
        for edge,value in traversal:
            inspected+=1
            if edge==active:
                active=value;completed+=1
            elif cache_records and not any(k==edge for k,_ in cache):
                cache.append((edge,value))
                if len(cache)>cache_records:cache.pop(0)
            while any(k==active for k,_ in cache):
                at=next(i for i,(k,_) in enumerate(cache) if k==active)
                _,active=cache.pop(at);completed+=1
            if completed==h:break
        if scans>h:raise AssertionError('termination bound violated')
    assert active==nodes[-1]
    return scans,inspected


def positions(order):
    result=[0]*len(order)
    for p,edge in enumerate(order):result[edge]=p
    return result


def predicted_scans(order,alternating=False):
    p=positions(order)
    if len(p)==1:return 1
    directions=[1 if b>a else -1 for a,b in zip(p,p[1:])]
    if not alternating:return 1+sum(d<0 for d in directions)
    return 1+int(directions[0]<0)+sum(a!=b for a,b in zip(directions,directions[1:]))


def eulerian(n):
    if n<1:raise ValueError('positive n required')
    a=[1]
    for m in range(2,n+1):
        a=[(k+1)*(a[k] if k<len(a) else 0)+(m-k)*(a[k-1] if k else 0) for k in range(m)]
    return a


def enumerate_scans(max_h=8):
    rows=[]
    for h in range(1,max_h+1):
        hist=Counter(); alt=Counter(); cached=Counter(); inspected=Counter()
        for order in permutations(range(h)):
            f,work=scan_chain(order);a,_=scan_chain(order,alternating=True)
            assert f==predicted_scans(order)
            assert a==predicted_scans(order,True)
            hist[f]+=1;alt[a]+=1;inspected['forward']+=work
            c,_=scan_chain(order,cache_records=1);cached[c]+=1
        assert [hist[i+1] for i in range(h)]==eulerian(h)
        count=math.factorial(h)
        mean=sum(k*v for k,v in hist.items())/count
        amean=sum(k*v for k,v in alt.items())/count
        assert math.isclose(mean,(h+1)/2)
        assert math.isclose(amean,1 if h==1 else 1.5+(h-2)*2/3)
        rows.append(dict(hops=h,permutations=count,forward_hist=dict(hist),alternating_hist=dict(alt),
            one_record_cache_hist=dict(cached),forward_mean_scans=mean,alternating_mean_scans=amean,
            one_record_cache_mean_scans=sum(k*v for k,v in cached.items())/count,
            forward_mean_records_inspected=inspected['forward']/count,
            forward_cdf=[sum(v for k,v in hist.items() if k<=budget)/count for budget in range(1,h+1)],
            alternating_cdf=[sum(v for k,v in alt.items() if k<=budget)/count for budget in range(1,h+1)]))
    return rows


def optimal_binary_compression(m,bits):
    """Exact codebook search for uniform binary records and uniform post-query.

    Each deterministic decoder maps one of 2**bits states plus index q to a bit;
    its responses across q form a reconstruction word. Given a codebook, the
    optimal encoder chooses a nearest word. Thus arbitrary B-bit encoders and
    decoders reduce exactly to this exhaustive Hamming codebook problem.
    This is a finite illustration of rate distortion, not a new lower bound.
    """
    if not 1<=m<=4 or not 0<=bits<=m:raise ValueError('enumeration only m<=4, 0<=bits<=m')
    n=1<<m;k=1<<bits
    if k==n:return dict(records=m,memory_bits=bits,codebooks_examined=1,min_total_hamming=0,optimal_accuracy=1.0,codebook=list(range(n)))
    distances=np.array([[(x^y).bit_count() for y in range(n)] for x in range(n)],dtype=np.int16)
    best=m*n+1;best_code=None;count=0
    for code in combinations(range(n),k):
        total=int(distances[:,code].min(axis=1).sum());count+=1
        if total<best:best=total;best_code=code
    return dict(records=m,memory_bits=bits,codebooks_examined=count,min_total_hamming=best,
        optimal_accuracy=1-best/(m*n),codebook=list(best_code),
        naive_store_bits_accuracy=.5+.5*bits/m)


def binary_entropy(p):
    return 0.0 if p in (0,1) else -p*math.log2(p)-(1-p)*math.log2(1-p)


def fano_error_lower(m,bits):
    target=max(0,1-bits/m);lo=0.;hi=.5
    for _ in range(60):
        mid=(lo+hi)/2
        if binary_entropy(mid)<target:lo=mid
        else:hi=mid
    return (lo+hi)/2


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    scans=enumerate_scans()
    compression=[optimal_binary_compression(m,b) for m in range(1,5) for b in range(m+1)]
    for row in compression:
        row['fano_error_lower']=fano_error_lower(row['records'],row['memory_bits'])
        assert 1-row['optimal_accuracy']+1e-12>=row['fano_error_lower']
    result=dict(scope='exact restricted algorithms and finite rate-distortion enumeration; not trained-model evidence',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scan_assumptions=['unique chain edges','active-key online walker','no off-path cache except separately labeled control','one directional traversal per scan','no learned hidden representations'],
        compression_assumptions=['independent uniform binary records','uniform query revealed after encoding','all retained history information limited to B bits','no later history access','unlimited decoder computation'],
        scans=scans,compression=compression,checks=dict(permutations=sum(r['permutations'] for r in scans),
        forward_formula_mismatches=0,alternating_formula_mismatches=0,eulerian_histogram_mismatches=0),
        interpretation='Extra computation cannot beat the optimal decoder for a fixed information bottleneck. Rereading can help a restricted walker; direction and cache policy affect required scans. Neither fact is a universal diffusion advantage.')
    (out/'exact_results.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result,out)
    return result


def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'savefig.bbox':'tight'})
    fig,ax=plt.subplots(figsize=(6.7,4))
    for m in [2,3,4]:
        rows=[r for r in result['compression'] if r['records']==m]
        ax.plot([r['memory_bits']/m for r in rows],[r['optimal_accuracy'] for r in rows],marker='o',label=f'{m} binary records: exact optimum')
        ax.plot([r['memory_bits']/m for r in rows],[1-r['fano_error_lower'] for r in rows],ls='--',alpha=.5)
    ax.set(xlabel='Retained bits / history bits',ylabel='Optimal random-query accuracy',ylim=(.45,1.04),title='Information limit despite unlimited decoder computation')
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.savefig(out/'compression_limit.pdf');fig.savefig(out/'compression_limit.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.7,4));rows=result['scans'];x=[r['hops'] for r in rows]
    for field,label in [('forward_mean_scans','Forward scans: exact Eulerian law'),('alternating_mean_scans','Alternating directions: exact law'),('one_record_cache_mean_scans','Forward + one cached record (control)')]:
        ax.plot(x,[r[field] for r in rows],marker='o',label=label)
    ax.set(xlabel='Chain hops',ylabel='Mean directional scans',title='Restricted walker; uniformly permuted chain records');ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.savefig(out/'scan_cost.pdf');fig.savefig(out/'scan_cost.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.7,4));r=rows[-1];x=list(range(1,r['hops']+1))
    ax.plot(x,r['forward_cdf'],marker='o',label='Forward only');ax.plot(x,r['alternating_cdf'],marker='s',label='Alternating directions')
    ax.set(xlabel='Allowed directional scans',ylabel='Exact success probability',title=f"All {r['permutations']:,} permutations of {r['hops']}-hop chains",ylim=(-.03,1.03));ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'pass_accuracy.pdf');fig.savefig(out/'pass_accuracy.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/theory_mvp'));a=p.parse_args()
    r=run(a.out);print(json.dumps(r['checks'],indent=2))
