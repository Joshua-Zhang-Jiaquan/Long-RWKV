"""Exact GF(2) compression/parallel-reveal MVP, independent of pretrained models.

Bits are represented as Python integer masks. Only exact finite distributions
are used. The results illustrate known linear-algebra/information identities;
novelty must be assessed separately, not inferred from passing these checks.
"""
from itertools import combinations
from collections import Counter
from pathlib import Path
import json, math, hashlib


def span(rows):
    s={0}
    for row in rows:s |= {x^row for x in tuple(s)}
    return s


def rank(rows):return (len(span(rows))-1).bit_length()


def kernel(rows,n):return [x for x in range(1<<n) if all((x&r).bit_count()%2==0 for r in rows)]


def entropy(values):
    c=Counter(values);n=sum(c.values())
    return -sum((v/n)*math.log2(v/n) for v in c.values())


def projected(x,mask):return x&mask


def direct_tc(support,mask,n):
    return sum(entropy([(x>>j)&1 for x in support]) for j in range(n) if mask>>j&1)-entropy([x&mask for x in support])


def tc_formula(rows,unknown,batch,n):
    if batch & ~unknown:raise ValueError('batch outside unknown coordinates')
    restricted=[r&unknown for r in rows];basis=span(restricted)
    deterministic=sum((1<<j) in basis for j in range(n) if batch>>j&1)
    return rank(restricted)-rank([r&(unknown^batch) for r in rows])-deterministic


def information_set(rows,n):
    support=kernel(rows,n);k=n-rank(rows)
    for indices in combinations(range(n),k):
        mask=sum(1<<i for i in indices)
        if len({x&mask for x in support})==len(support):return mask
    raise AssertionError('linear support has no information set')


def enumerate_rowspaces(n,r):
    seen=set()
    for rows in combinations(range(1,1<<n),r):
        s=frozenset(span(rows))
        if len(s)!=(1<<r) or s in seen:continue
        seen.add(s);yield rows


def run(out=Path('results/theory_mvp')):
    out.mkdir(parents=True,exist_ok=True)
    rows_out=[];checks=0
    for rows in enumerate_rowspaces(4,2):
        n=4;full=(1<<n)-1;r=rank(rows);support=kernel(rows,n)
        # Every valid observed assignment has identical entropy/rank structure
        # by affine translation. Verify all cosets and observed values explicitly.
        by_c={}
        for x in range(1<<n):
            c=tuple((x&a).bit_count()%2 for a in rows);by_c.setdefault(c,[]).append(x)
        for coset in by_c.values():
            for unknown in range(1<<n):
                groups={}
                for x in coset:groups.setdefault(x&(full^unknown),[]).append(x)
                for group in groups.values():
                    batch=unknown
                    while True:
                        empirical=direct_tc(group,batch,n);pred=tc_formula(rows,unknown,batch,n)
                        assert abs(empirical-pred)<1e-12,(rows,unknown,batch,empirical,pred)
                        checks+=1
                        if batch==0:break
                        batch=(batch-1)&unknown
        d=sum((1<<j) in span(rows) for j in range(n));tc=r-d
        assert abs(direct_tc(support,full,n)-tc)<1e-12
        info=information_set(rows,n)
        assert direct_tc(support,info,n)==0
        # Independent first-round fair bits on an information set enumerate
        # every true support point exactly once; remaining coordinates then fixed.
        completions={}
        for x in support:completions.setdefault(x&info,[]).append(x)
        assert len(completions)==1<<(n-r) and all(len(v)==1 for v in completions.values())
        independent={}
        marginals=[Counter((x>>j)&1 for x in support) for j in range(n)]
        for x in range(1<<n):
            prob=math.prod(marginals[j][(x>>j)&1]/len(support) for j in range(n))
            independent[x]=prob
        kl=sum(math.log2((1/len(support))/independent[x])/len(support) for x in support)
        assert abs(kl-tc)<1e-12
        rows_out.append(dict(matrix_rows=list(rows),rank=r,memory_bits=r,conditional_entropy_bits=n-r,
            individually_determined_coordinates=d,one_round_forward_kl_bits=kl,
            one_round_constraint_satisfaction=sum(independent[x] for x in support),
            one_round_total_variation=1-2**(-tc),
            full_history_logloss_bits=n-d,
            two_round_forward_kl_bits=0,information_set_mask=info))
    # Higher-order dependence can be completely invisible to pairwise checks.
    blind_rows=[15,51];blind_support=kernel(blind_rows,6)
    pairs=[direct_tc(blind_support,(1<<a)|(1<<b),6) for a,b in combinations(range(6),2)]
    assert all(x==0 for x in pairs)
    blind_tc=direct_tc(blind_support,63,6)
    assert blind_tc==2
    blind=dict(n=6,rows=blind_rows,rank=2,pairwise_tc_bits=pairs,total_correlation_bits=blind_tc,
        independent_reveal_constraint_satisfaction=.25,independent_reveal_tv=.75,
        caveat='Counterexample to pairwise-only independence certification, not a test of any named sampler implementation')
    result=dict(higher_order_counterexample=blind,scope='Uniform binary source, exact linear compression, oracle conditional marginals; no neural claim',
        n=4,rank=2,rowspaces=len(rows_out),conditional_batch_checks=checks,max_formula_residual=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),encoders=rows_out,
        novelty_status='Linear algebra and information-theoretic construction; not established as a new theorem',
        theorem='Fixed rank fixes retained information, but not factorized posterior error. A coordinate information set permits exact two-round sampling.',
        caveat='All states, masks and matrix are available to the oracle. Matrix solving and coefficient storage are not free in a recurrent implementation; no F2 guarantee follows.')
    (out/'linear_compression_results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts=Counter(x['one_round_forward_kl_bits'] for x in rows_out)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    axs[0].bar(sorted(counts),[counts[k] for k in sorted(counts)],color='#326a92')
    axs[0].set(xlabel='One-round forward KL (bits)',ylabel='Number of distinct encoders',xticks=[0,1,2],title='Same two retained bits; different parallel error')
    xs=[0,1,2];axs[1].plot(xs,[2**(-x) for x in xs],'o-',label='Independent one-round reveal')
    axs[1].plot(xs,[1]*3,'s--',label='Information-set two-round reveal')
    axs[1].set(xlabel='Posterior total correlation (bits)',ylabel='Probability of satisfying stored constraints',xticks=xs,ylim=(0,1.05))
    axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'representation_parallelism.pdf');fig.savefig(out/'representation_parallelism.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['rowspaces','conditional_batch_checks','max_formula_residual']}))
    print('one-round KL histogram:',dict(counts))
    return result

if __name__=='__main__':run()
