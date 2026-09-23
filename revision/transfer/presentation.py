"""Render the completed frozen report; refuses to generate provisional numbers."""
import argparse,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,default=ROOT/'results/predictive_transfer/report.json')
    ap.add_argument('--out',type=Path,default=ROOT/'results/predictive_transfer/paper');args=ap.parse_args()
    report=json.loads(args.report.read_text());protocol=json.loads((ROOT/'revision/transfer/FROZEN.json').read_text())
    if report['status']!='complete held-out analysis; all fixed selectors retained' or len(report['cells'])!=1440:
        raise ValueError('complete fixed report required; no provisional or synthetic plotting')
    seeds=protocol['training_seeds'];cells=report['cells'];args.out.mkdir(parents=True,exist_ok=True)
    lookup={(r['seed'],r['cell']['class_id'],r['cell']['index'],r['cell']['position']):r for r in cells}
    if len(lookup)!=1440:raise ValueError('duplicate cells')
    rows=['\\begin{tabular}{@{}lrrrr@{}}','\\toprule','Lineage & Selected KL & KL reduction & Regret & Competent\\\\','\\midrule']
    for i,lineage in enumerate(report['lineages']):
        own=[r for r in cells if r['seed']==lineage['seed']]
        regret=sum(r['regret'] for r in own)/len(own)
        rows.append(f"{i+1} & {lineage['mean_selected_kl']:.4f} & {lineage['primary']:.4f} & {regret:.4f} & {'yes' if lineage['competent'] else 'no'}\\\\")
    rows.extend(['\\midrule',f"Mean & {report['predicted_policy_kl']:.4f} & {report['primary_mean_nats']:.4f} & {report['mean_selection_regret']:.4f} & {sum(r['competent'] for r in report['lineages'])}/6\\\\",'\\bottomrule','\\end{tabular}'])
    (args.out/'lineage_table.tex').write_text('\n'.join(rows)+'\n')
    names={'fixed_A':'Fixed basis A','fixed_B':'Fixed basis B','random':'Uniform random choice','unconditional':'Source-unconditional choice','fanin_heuristic':'Parity fan-in heuristic'}
    rows=['\\begin{tabular}{@{}lrr@{}}','\\toprule','Comparator & Mean KL & Reduction from selection\\\\','\\midrule',
          f"Dependence-only hash tie & {report['predicted_policy_kl']+report['primary_mean_nats']:.4f} & {report['primary_mean_nats']:.4f}\\\\"]
    for key,label in names.items():rows.append(f"{label} & {report['baseline_kl'][key]:.4f} & {report['improvement_over_baselines'][key]:.4f}\\\\")
    rows.extend(['\\bottomrule','\\end{tabular}']);(args.out/'baseline_table.tex').write_text('\n'.join(rows)+'\n')
    # Descriptive position summaries preserve the complete fixed primary panel.
    positions=('far','middle','near');by_position=[]
    for seed in seeds:
        for position in positions:
            own=[r for r in cells if r['seed']==seed and r['cell']['position']==position]
            by_position.append(dict(seed=seed,position=position,count=len(own),selected_kl=float(np.mean([r['selected_kl'] for r in own])),
                                    benefit=float(np.mean([r['benefit'] for r in own])),regret=float(np.mean([r['regret'] for r in own]))))
    predicted=np.array([r['prediction']['scores'][0]-r['prediction']['scores'][1] for r in cells]);realized=np.array([r['kl'][0]-r['kl'][1] for r in cells])
    diagnostics=dict(position_summaries=by_position,near_tied_predicted_contrasts=int((np.abs(predicted)<=1e-9).sum()),
                     near_tie_tolerance_nats=1e-9,interpretation='descriptive diagnostics; do not change sealed decisions or primary gates',
                     plotted_cells=len(cells),score_contrast_rmse=float(np.sqrt(np.mean((predicted-realized)**2))))
    (args.out/'descriptive_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1),constrained_layout=True)
    colors=('#356b91','#c06b3d','#68994c')
    ax=axes[0,0]
    for position,color in zip(positions,colors):
        mask=np.array([r['cell']['position']==position for r in cells]);ax.scatter(predicted[mask],realized[mask],s=7,alpha=.28,label=position,color=color,rasterized=True)
    lo=float(min(predicted.min(),realized.min(),0));hi=float(max(predicted.max(),realized.max(),0));pad=max((hi-lo)*.04,.01)
    ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='.5',lw=.8,ls='--');ax.set(xlabel='Predicted KL(A) − KL(B)',ylabel='Realized KL(A) − KL(B)',title='Prospective contrasts (all cells)')
    ax.legend(frameon=False,fontsize=6)
    ax=axes[0,1];ax.scatter(range(1,7),[r['primary'] for r in report['lineages']],color=colors[0]);ax.axhline(0,color='.5',lw=.8);ax.set(xlabel='Training lineage',ylabel='Mean KL reduction (nats)',title='All independent task-training lineages',xticks=range(1,7))
    ax=axes[1,0];ax.scatter(range(15,25),[r['primary'] for r in report['classes']],color=colors[0]);ax.axhline(0,color='.5',lw=.8);ax.set(xlabel='Held-out code class',ylabel='Mean KL reduction (nats)',title='All held-out dependency classes',xticks=range(15,25))
    ax=axes[1,1]
    for i,seed in enumerate(seeds):
        vals=[r['selected_kl'] for r in by_position if r['seed']==seed];ax.plot(range(3),vals,marker='o',ms=3,lw=.8,label=str(i+1))
    ax.axhline(4*math.log(2),color='.35',ls='--',lw=.8,label='Product floor');ax.set(xticks=range(3),xticklabels=positions,ylabel='Selected-policy KL (nats)',title='Absolute quality at each position')
    ax.legend(frameon=False,fontsize=6,ncol=4)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.13)
    fig.savefig(args.out/'prospective_results.pdf');fig.savefig(args.out/'prospective_results.png',dpi=180)
    print(json.dumps(dict(out=str(args.out),plotted_cells=len(cells),all_primary_gates_pass=report['all_primary_gates_pass'])))
if __name__=='__main__':main()
