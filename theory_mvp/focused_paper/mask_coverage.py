"""Exact query-law density ratios for the declared weighted absorbing objective."""
import itertools

def ordinary_query_law(n=8):
    # N uniform stages, masking probability t/N, per masked-coordinate weight 1/t.
    law={}
    for t in range(1,n+1):
        p=t/n
        for mask in range(1,1<<n):
            m=mask.bit_count();prob=p**m*(1-p)**(n-m)
            if not prob:continue
            for i in range(n):
                if mask>>i&1:law[t,mask,i]=prob/(n*t)
    return law

def policy_query_law(groups,n=8):
    if sorted(itertools.chain.from_iterable(groups))!=list(range(n)):raise ValueError('not a partition')
    law={};visible=0
    for group in groups:
        mask=((1<<n)-1)^visible;t=mask.bit_count()
        for i in group:law[t,mask,i]=1/n
        for i in group:visible|=1<<i
    return law

def density_ratio(target,training):
    return max(p/training[k] if training.get(k,0)>0 else float('inf') for k,p in target.items())

def results():
    n=8;ordinary=ordinary_query_law(n)
    info=policy_query_law([list(range(4)),list(range(4,8))]);seq=policy_query_law([[i] for i in range(n)])
    other_info=policy_query_law([list(range(4,8)),list(range(4))])
    other_mixture={k:.5*ordinary.get(k,0)+.5*other_info.get(k,0) for k in ordinary.keys()|other_info.keys()}
    pair=policy_query_law([[0,1,4,5],[2,3,6,7]])
    mixture={k:.5*ordinary.get(k,0)+.5*info.get(k,0) for k in ordinary.keys()|info.keys()}
    return dict(ordinary_total_mass=sum(ordinary.values()),information_set_under_ordinary=density_ratio(info,ordinary),sequential_under_ordinary=density_ratio(seq,ordinary),information_set_under_mixture=density_ratio(info,mixture),sequential_under_mixture=density_ratio(seq,mixture),pair_preserving_under_mixture=density_ratio(pair,mixture),sequential_under_permuted_information_set_mixture=density_ratio(seq,other_mixture),scope='First-four information set except explicitly permuted comparison; population query weights for same context law and target-distributed visible values; no generalization claim')

if __name__=='__main__':
    import json
    print(json.dumps(results(),indent=2))
