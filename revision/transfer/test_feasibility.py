import itertools,math
from revision.transfer.feasibility import rank,inverse,profile

def matvec(rows,u):return sum(((r&u).bit_count()%2)<<i for i,r in enumerate(rows))
def entropy(values):
    from collections import Counter
    counts=Counter(values);n=len(values)
    return -sum(v/n*math.log(v/n) for v in counts.values())

def test_all_inverses_and_oracle_schedule_errors():
    total=0;classes=set();different=0
    for rows in itertools.product(range(1,16),repeat=4):
        if rank(rows)!=4:continue
        total+=1;inv=inverse(rows);outputs=[matvec(rows,u) for u in range(16)]
        assert sorted(outputs)==list(range(16))
        assert all(matvec(inv,outputs[u])==u for u in range(16))
        # Either first-round vector is uniform on all 16 binary values; the
        # second-round vector is a deterministic function. Both D terms vanish.
        for values in (list(range(16)),outputs):
            d=sum(entropy([(v>>i)&1 for v in values]) for i in range(4))-entropy(values)
            assert abs(d)<1e-12
        classes.add(profile(rows))
        different+=sorted(v.bit_count() for v in rows)!=sorted(v.bit_count() for v in inv)
    assert total==15*14*12*8==20160
    assert len(classes)==25 and different==13440
