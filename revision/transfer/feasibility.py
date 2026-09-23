"""CPU-only design exploration; no learned-model predictions or test selection."""
import itertools,json
from collections import Counter
from pathlib import Path

def rank(rows):
    pivots={}
    for x in rows:
        while x:
            i=x.bit_length()-1
            if i in pivots:x^=pivots[i]
            else:pivots[i]=x;break
    return len(pivots)

def inverse(rows):
    a=[v|(1<<(4+i)) for i,v in enumerate(rows)]
    for j in range(4):
        k=next(i for i in range(j,4) if a[i]>>j&1);a[j],a[k]=a[k],a[j]
        for i in range(4):
            if i!=j and a[i]>>j&1:a[i]^=a[j]
    assert [v&15 for v in a]==[1,2,4,8]
    return tuple(v>>4 for v in a)

def profile(rows):
    counts=[0]*9
    for u in range(16):
        p=sum(((u&r).bit_count()%2)<<j for j,r in enumerate(rows))
        counts[(u|(p<<4)).bit_count()]+=1
    return tuple(counts)

def main():
    groups={};total=0;asymmetric=0
    for rows in itertools.product(range(1,16),repeat=4):
        if rank(rows)!=4:continue
        total+=1;inv=inverse(rows)
        forward=tuple(sorted(r.bit_count() for r in rows));reverse=tuple(sorted(r.bit_count() for r in inv))
        different=forward!=reverse;asymmetric+=different
        key=profile(rows);g=groups.setdefault(key,dict(matrices=0,asymmetric=0,example=None,asymmetric_example=None))
        g['matrices']+=1;g['asymmetric']+=different
        ex=dict(B=rows,inverse=inv,forward_fan_in=forward,reverse_fan_in=reverse)
        if g['example'] is None:g['example']=ex
        if different and g['asymmetric_example'] is None:g['asymmetric_example']=ex
    result=dict(status='design exploration, not preregistration or neural evidence',invertible_matrices=total,
      distinct_weight_enumerators=len(groups),asymmetric_fan_in_matrices=asymmetric,
      classes=[dict(weight_enumerator=k,**v) for k,v in sorted(groups.items())],
      interpretation='Different homogeneous-code weight enumerators certify non-equivalence under arbitrary output-coordinate permutation. Equal enumerators do not prove equivalence. For invertible B, U-first and P-first each have two calls and zero oracle discarded dependence; their deterministic second-pass fan-in profiles can differ.')
    Path(__file__).with_name('feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='classes'},indent=2))
if __name__=='__main__':main()
