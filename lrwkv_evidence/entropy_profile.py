"""Independent finite-law checks of the absorbing entropy-profile mesh bound."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random


def subset_entropies(prob,n):
    h=[]
    for mask in range(1<<n):
        masses=defaultdict(float)
        for y,p in enumerate(prob):masses[y&mask]+=p
        h.append(-sum(p*math.log(p) for p in masses.values() if p))
    return h


def profile(h,n,s):
    value=0.
    for i in range(n):
        for mask in range(1<<n):
            if mask>>i&1:continue
            m=mask.bit_count()
            value+=s**m*(1-s)**(n-1-m)*(h[mask|1<<i]-h[mask])
    return value


def integrated_profile(h,n):
    return sum((h[mask|1<<i]-h[mask])/(n*math.comb(n-1,mask.bit_count()))
               for i in range(n) for mask in range(1<<n) if not mask>>i&1)


def direct_path_dependence(h,n,grid):
    full=(1<<n)-1;value=0.
    for left,right in zip(grid,grid[1:]):
        omega=(right-left)/(1-left)
        for visible in range(1<<n):
            count=visible.bit_count();pv=left**count*(1-left)**(n-count);unknown=full^visible
            batch=unknown
            while True:
                j=batch.bit_count();pj=omega**j*(1-omega)**(n-count-j)
                tc=sum(h[visible|1<<i]-h[visible] for i in range(n) if batch>>i&1)-(h[visible|batch]-h[visible])
                value+=pv*pj*tc
                if batch==0:break
                batch=(batch-1)&unknown
    return value


def run():
    laws=[]
    for n in (2,3,4,6):
        def uniform_on(name,select):
            support=[y for y in range(1<<n) if select(y)];laws.append((name,n,[float(y in support)/len(support) for y in range(1<<n)]))
        uniform_on('independent',lambda y:True)
        uniform_on('repetition',lambda y:y in (0,(1<<n)-1))
        uniform_on('single_parity',lambda y:y.bit_count()%2==0)
        rng=random.Random(300+n);p=[rng.random() for _ in range(1<<n)];total=sum(p);laws.append(('random_positive',n,[x/total for x in p]))
    grids=[[i/t for i in range(t+1)] for t in (1,2,4,8)]+[[0.,.05,.25,.7,1.]]
    rows=[]
    for name,n,p in laws:
        h=subset_entropies(p,n);integral=integrated_profile(h,n)
        assert abs(integral-h[-1])<1e-10
        f0,f1=profile(h,n,0),profile(h,n,1)
        for grid in grids:
            direct=direct_path_dependence(h,n,grid)
            riemann=sum((b-a)*profile(h,n,a) for a,b in zip(grid,grid[1:]))-h[-1]
            mesh=max(b-a for a,b in zip(grid,grid[1:]));bound=mesh*(f0-f1)
            tc=f0-h[-1];old=math.comb(n,2)*math.log(2)*sum((b-a)**2 for a,b in zip(grid,grid[1:]))
            assert abs(direct-riemann)<1e-10
            assert -1e-10<=direct<=min(bound,tc,old)+1e-10
            rows.append(dict(law=name,n=n,grid=grid,direct_path_dependence=direct,profile_riemann_gap=riemann,
                             profile_bound=bound,total_correlation_bound=tc,pair_collision_bound=old,
                             universal_linear_bound=n*mesh*math.log(2)))
    return dict(cases=len(rows),maximum_identity_residual=max(abs(r['direct_path_dependence']-r['profile_riemann_gap']) for r in rows),rows=rows,
                scope='Finite laws verify implementation and examples; proof is the entropy derivative plus monotone Riemann-sum argument. Not a novelty claim.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=run()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
