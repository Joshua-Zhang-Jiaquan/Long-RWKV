"""Report a completed matched development phase, never an interim winner."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(plans):
    runs=[]
    for path in plans:
        plan=json.loads(path.read_text());root=Path(plan['out'])
        completion=json.loads((root/'completion.json').read_text());provenance=json.loads((root/'provenance.json').read_text())
        train=[json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()]
        dev=[json.loads(x) for x in (root/'dev.jsonl').read_text().splitlines()]
        assert completion['execution_complete'] and completion['step']==plan['stop_step']
        assert [r['step'] for r in train]==list(range(provenance['start_step']+1,plan['stop_step']+1))
        assert dev[-1]['step']==plan['stop_step']
        runs.append(dict(objective=plan['objective'],source=str(root),completion=completion,provenance=provenance,
                         final100_mean_loss=statistics.mean(r['mean_loss'] for r in train[-100:]),development=dev,train=train))
    assert {r['objective'] for r in runs}=={'hard','rao_blackwell'}
    contracts=[]
    for r in runs:
        contract=dict(r['provenance']['contract']);contract.pop('objective')
        # Label arms have different trained parent weights; each parent is recorded.
        contract.pop('parent_checkpoint_sha256',None);contracts.append(contract)
    assert contracts[0]==contracts[1], 'arms must share exact training contract except labels'
    assert len({r['completion']['step'] for r in runs})==1
    return dict(scope='One matched development seed; neither locked test nor independent training replication',runs=runs)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plans',type=Path,nargs=2,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=summarize(a.plans);a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Matched curriculum development','',result['scope'],'','| Objective | Dimension | Deterministic accuracy | Marginal KL/bit | Fair-bit probability error |','|---|---:|---:|---:|---:|']
    for run in result['runs']:
        for row in run['development'][-1]['metrics']:
            lines.append(f"| {run['objective']} | {row['n']} | {100*row['deterministic_accuracy']:.2f}% | {row['marginal_kl_per_masked_bit']:.6f} | {row['fair_probability_absolute_error']:.6f} |")
    lines+=['','Final-task N8 outcomes remain the target. N2/N4 curriculum outcomes are not substitutes. Proper loss and endpoint distribution metrics are needed alongside thresholded accuracy.']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dimensions=sorted({m['n'] for run in result['runs'] for m in run['development'][-1]['metrics']})
    fig,grid=plt.subplots(1,len(dimensions),figsize=(4.5*len(dimensions),3.3),squeeze=False);axes=grid[0]
    for run in result['runs']:
        label=run['objective'].replace('_',' ')
        for n,axis in zip(dimensions,axes):
            points=[(r['step'],m['marginal_kl_per_masked_bit']) for r in run['development'] for m in r['metrics'] if m['n']==n]
            if points:axis.plot(*zip(*points),marker='o',label=label)
            axis.set(title=f'{n}-bit development',xlabel='Update',ylabel='Marginal KL / masked bit (nats)');axis.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(a.out/'development_kl.pdf');fig.savefig(a.out/'development_kl.png',dpi=160);plt.close(fig)
    print('\n'.join(lines))
if __name__=='__main__':main()
