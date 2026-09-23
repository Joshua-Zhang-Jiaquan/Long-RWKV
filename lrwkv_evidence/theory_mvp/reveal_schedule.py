"""Exact absorbing-reveal DP on the linear-compression oracle experiment.

The reveal decisions are independent of sampled token values. Original tokens
are never revised. Once a partial assignment violates a projected constraint,
no later assignment can make a valid codeword; arbitrary completion then has
no effect on valid-support probability. Symmetry makes valid endpoint mass
uniform on the affine code, so endpoint KL(bits)=-log2(valid mass).
"""
from pathlib import Path
import json, math, hashlib
from .linear_compression import tc_formula, enumerate_rowspaces


def transitions(rows,n):
    result={}
    for unknown in range(1<<n):
        pairs=[];batch=unknown
        while True:
            pairs.append((batch,batch.bit_count(),tc_formula(rows,unknown,batch,n)))
            if not batch:break
            batch=(batch-1)&unknown
        result[unknown]=pairs
    return result


def exact_schedule(rows,n,steps):
    if steps<1:raise ValueError('positive steps')
    edges=transitions(rows,n);size=1<<n
    survival=[0.]*size;law=[0.]*size;survival[-1]=law[-1]=1.;path=0.
    for remaining in range(steps,0,-1):
        w=1/remaining;next_surv=[0.]*size;next_law=[0.]*size
        for unknown in range(size):
            m=unknown.bit_count()
            for batch,k,tc in edges[unknown]:
                prob=w**k*(1-w)**(m-k)
                target=unknown^batch
                next_law[target]+=law[unknown]*prob
                next_surv[target]+=survival[unknown]*prob*2**(-tc)
                path+=law[unknown]*prob*tc
        survival,law=next_surv,next_law
    valid=survival[0]
    assert abs(law[0]-1)<1e-10
    kl=-math.log2(valid)
    assert -1e-10<=kl<=path+1e-10
    return dict(steps=steps,valid_probability=valid,total_variation=1-valid,
        endpoint_forward_kl_bits=max(0.,kl),path_dependence_bits=path)


def run(out=Path('results/theory_mvp')):
    stages=[1,2,4,8,16,32,64,128];encoders=[]
    for rows in enumerate_rowspaces(4,2):
        values=[exact_schedule(rows,4,t) for t in stages]
        assert abs(values[0]['endpoint_forward_kl_bits']-tc_formula(rows,15,15,4))<1e-12
        encoders.append(dict(matrix_rows=list(rows),schedules=values))
    result=dict(scope='Exact oracle linear-code posteriors, independent absorbing reveals; no neural measurements',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n=4,rank=2,
        schedules=stages,encoders=encoders,information_set_two_round_kl_bits=0,
        off_support_policy='Any irreversible completion after first infeasible partial assignment; none can enter true support',
        units='bits; multiply KL by ln(2) for nats')
    (out/'reveal_schedule_results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(6.8,4.2))
    for rows,label in [([1,2],'Stored coordinates'),([1,6],'One fixed coordinate + parity'),([3,12],'Two parity constraints')]:
        vals=[exact_schedule(rows,4,t) for t in stages]
        ax.plot(stages,[v['endpoint_forward_kl_bits'] for v in vals],marker='o',label=label)
    ax.scatter([2],[0],marker='*',s=150,color='black',zorder=5,label='Information-set two-round oracle')
    ax.set(xscale='log',xlabel='Nominal absorbing-reveal stages',ylabel='Exact endpoint forward KL (bits)',title='Same retained information; schedule costs differ')
    ax.set_xticks(stages,labels=[str(t) for t in stages]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'reveal_schedule.pdf');fig.savefig(out/'reveal_schedule.png',dpi=160);plt.close(fig)
    print(json.dumps({'encoders':len(encoders),'schedules_per_encoder':len(stages),'two_parity_example':[exact_schedule([3,12],4,t) for t in [1,8,64]]},indent=2))
    return result

if __name__=='__main__':run()
