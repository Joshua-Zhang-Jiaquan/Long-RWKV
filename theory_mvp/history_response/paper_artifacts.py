"""Presentation and a clearly labeled algebraic corollary; primary analysis is frozen."""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'results/history_response'


def main():
    report=json.loads((FOLDER/'report.json').read_text())
    if report['conditions']!=672 or report['paired_interventions']!=43008:
        raise ValueError('complete exhaustive audit required')
    roles=list(report['checkpoints'])
    names={'original':'Original'}
    for seed in (20271011,20271012,20271013):
        for a in ('near','balanced'):
            names[f'{a}_seed{seed}']=f'{a.capitalize()} {str(seed)[-2:]}'
    rows=[];corollary={}
    for role in roles:
        raw=json.loads((FOLDER/f'exact_{role}.json').read_text())
        corollary[role]={}
        for p in ('far','middle','near'):
            cells=[r for r in raw['conditions'] if r['position']==p]
            bounds=[r['first_error']+8*r['response']['mean_correct_conditional_ce'] for r in cells]
            exact=[r['information_set_error'] for r in cells]
            if any(b+1e-8<e for b,e in zip(bounds,exact)):
                raise ValueError('response sufficient bound violated')
            c=dict(mean_upper_bound=float(np.mean(bounds)),mean_exact_error=float(np.mean(exact)),
                   max_upper_bound=max(bounds),max_exact_error=max(exact),
                   mean_response_certificate=np.mean(bounds)<4*math.log(2),
                   all32_response_certificates=all(b<4*math.log(2) for b in bounds),
                   all32_exact_certificates=all(e<4*math.log(2) for e in exact),
                   global_worst_pair_ce=max(r['worst_pair_ce'] for r in cells),
                   global_max_unaffected_change=max(r['max_unaffected_change'] for r in cells))
            c['mean_response_certificate']=bool(c['mean_response_certificate'])
            corollary[role][p]=c
            r=report['checkpoints'][role][p]
            rows.append((role,p,r,c))
    primary=report['seed_averaged']['far_mean_correct_conditional_ce_reduction']
    lo,hi=primary['ci95']
    mantissa,exponent=f"{report['prior_max_discrepancy']:.1e}".split('e')
    lines=[r'\paragraph{Exhaustive results.}',
           f"The audit retains all 672 conditions and 43,008 paired interventions. The far near-only minus balanced paired-CE contrast is {primary['mean']:.4f} nats (conditional 95\\% interval [{lo:.4f}, {hi:.4f}]).",
           'The three individual seed contrasts are '+', '.join(f"{report['per_seed'][str(s)]['far_mean_correct_conditional_ce_reduction']['mean']:.4f}" for s in (20271011,20271012,20271013))+ ' nats.',
           f"Recomputed archived metrics agree to within ${mantissa}\\times10^{{{int(exponent)}}}$ in absolute value.",
           r'Table~\ref{tab:history-response} reports every checkpoint and placement. The response-based upper bound is an additional algebraic diagnostic; the frozen primary analysis uses the paired-CE contrast.',
           r'\begin{table}[t]',r'\centering\small',
           r"\caption{Exhaustive response audit. $\bar L$ averages all histories and constraints; ``both'' is the fraction for which both evidence variants predict the required target correctly. $U=E_{\mathrm{first}}+8\bar L$ upper-bounds the original information-set error $E$. Last column is the worst paired CE over all 32 problems, 16 histories and four constraints, not the mean of per-problem maxima. All errors are nats.}",
           r'\label{tab:history-response}',r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Model & Position & $\bar L$ & Both & $E$ & $U$ & Max CE\\',r'\midrule']
    for role,p,r,c in rows:
        vals=[r['mean_correct_conditional_ce']['mean'],r['both_variants_correct']['mean'],c['mean_exact_error'],c['mean_upper_bound'],c['global_worst_pair_ce']]
        lines.append(names[role]+' & '+p+' & '+' & '.join(f'{v:.4g}' for v in vals)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    balanced=[corollary[f'balanced_seed{s}'][p] for s in (20271011,20271012,20271013) for p in ('far','middle','near')]
    if all(c['all32_response_certificates'] for c in balanced):
        lines.append('Every balanced checkpoint satisfies the conservative response-based criterion for every one of the 32 problems at every measured position. This statement is limited to the enumerated problem panel.')
    else:
        lines.append('The conservative criterion does not pass every balanced problem/position cell; complete cell-level measurements remain in the archived raw audit.')
    lines.append(f"The largest unaffected-target probability change among all balanced interventions is {max(c['global_max_unaffected_change'] for c in balanced):.4g}. This worst-case value is reported separately from average selectivity.")
    diagnostic=json.loads((FOLDER/'collateral_diagnostic.json').read_text())
    collateral=[s for pp in diagnostic['checkpoints'].values() for s in pp.values()]
    failures=sum(s['correct_to_incorrect'] for s in collateral)
    comparisons=sum(s['unaffected_predictions'] for s in collateral)
    lines += [r'\paragraph{Rare collateral failures.}',
              f"A post hoc inspection of these large shifts finds {failures} correct-to-incorrect changes among {comparisons:,} unaffected-target comparisons across the three balanced models, 32 problems, all positions, histories and constraints. These are dependent comparisons within the enumerated panel, not independent test examples.",
              f"Averaging complete second-round errors on the flipped prompts, including unaffected targets, gives at most {max(s['mean_full_flipped_second_error'] for s in collateral):.4g} nats for any balanced model/position cell. The first-round flipped-prompt error was not measured, so this is not a full counterfactual endpoint KL.",
              'The findings support low average conditional error and mostly selective responses, while ruling out a claim of uniform selectivity. The conservative certificate in Eq.~\\eqref{eq:response-certificate} concerns the original target law and does not certify the complete counterfactual prompt.']
    resources=json.loads((FOLDER/'resource_audit.json').read_text())
    busy=[j['utilization']['entire_observed_job']['mean_gpu_busy_percent'] for j in resources['jobs']]
    memory=[j['utilization']['entire_observed_job']['mean_memory_used_percent'] for j in resources['jobs']]
    lines += [r'\paragraph{Additional resources.}',
              f"The seven evaluation jobs consume {resources['total_scheduler_running_gpu_hours']:.2f} additional scheduler H100-hours, with no retraining or failed launches. Peak queued-plus-running reservation is 48 GPUs, split 32 plus 16 across the authorized projects; all jobs are terminal.",
              f"Whole-job mean GPU busy ranges from {min(busy):.1f}\\% to {max(busy):.1f}\\%, including loading and numerical checks, while mean memory occupancy is {min(memory):.1f}--{max(memory):.1f}\\%. GPU busy is not achieved FLOP utilization; the memory target is not met under the stated fastest-completion priority."]
    (FOLDER/'paper_results.tex').write_text('\n'.join(lines)+'\n')
    (FOLDER/'response_corollary.json').write_text(json.dumps(dict(scope='Algebraic sufficient criterion added during follow-up preparation; not a new preregistered inferential endpoint.',checkpoints=corollary),indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained')
    colors={'near':'#b05236','balanced':'#217b7e'}
    for arm in ('near','balanced'):
        for j,seed in enumerate((20271011,20271012,20271013)):
            role=f'{arm}_seed{seed}';x=j+(-.12 if arm=='near' else .12)
            for ax,key in zip(axes,('mean_correct_conditional_ce','unaffected_mean_abs_probability_change')):
                r=report['checkpoints'][role]['far'][key];mean=r['mean'];lo,hi=r['ci95']
                ax.errorbar(x,mean,yerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',capsize=3,color=colors[arm],label=arm.capitalize() if j==0 else None)
    for ax,title in zip(axes,('Mean correct paired target CE','Mean unaffected-target change')):
        ax.set_yscale('log');ax.set_xticks(range(3),['11','12','13']);ax.set_xlabel('Adaptation seed suffix');ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('16K far; all histories and constraints');axes[0].legend(frameon=False)
    fig.supxlabel('32 reused problems; conditional problem-bootstrap 95% intervals.\nRare collateral failures are reported separately.',fontsize=8)
    fig.savefig(FOLDER/'exhaustive_response.pdf');fig.savefig(FOLDER/'exhaustive_response.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(primary=primary,all_balanced_cells_response_certificate=all(c['all32_response_certificates'] for c in balanced)),indent=2))


if __name__=='__main__':main()
