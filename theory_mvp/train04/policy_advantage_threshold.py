import itertools,json,math
from pathlib import Path
records=[]
for n in [2,4,6,8,10,12]:
 m=n//2;cost=[]
 for group in itertools.combinations(range(n),m):
  g=set(group);cost.append(sum((2*j in g)==(2*j+1 in g) for j in range(m))*math.log(2))
 measured=sum(cost)/len(cost);formula=m*(m-1)/(2*m-1)*math.log(2)
 assert abs(measured-formula)<1e-12
 records.append(dict(n=n,partitions=len(cost),mean_random_dependence_nats=measured,formula_nats=formula,parity_mixture_excess_sufficient_threshold=formula/(2*n),equal_family_mixture_excess_sufficient_threshold=formula/(4*n)))
Path('results/theory_mvp/policy_advantage_threshold.json').write_text(json.dumps(dict(scope='Exact combinatorial check; not measured neural loss or a finite-sample certificate',alpha=.5,records=records),indent=2)+'\n')
