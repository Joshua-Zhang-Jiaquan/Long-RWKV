"""Readable complete-panel tables and a factorial figure, without new selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];FOLDER=ROOT/'results/header_distance'


def main():
    report=json.loads((FOLDER/'report.json').read_text())
    if report['conditions']!=1152:raise ValueError('full panel required')
    roles=list(report['checkpoints']);cells=('header_far_task_far','header_far_task_near','header_near_task_far','header_near_task_near','legacy_far','legacy_near')
    p=report['primary'];lo,hi=p['ci95']
    lines=[r'\paragraph{Results.}',
           f"All 1,152 conditions and 2,304 endpoint laws are validated. The primary error reduction is {p['mean']:.4f} nats, with conditional 95\\% interval [{lo:.4f}, {hi:.4f}].",
           'The individual adaptation-seed effects are '+', '.join(f"{report['per_seed'][str(s)]['header_near_task_far_error_reduction']['mean']:.4f}" for s in (20271011,20271012,20271013))+' nats.',
           ('The predeclared stronger-interpretation criteria are satisfied.' if report['stronger_interpretation_supported'] else 'The predeclared stronger-interpretation criteria are not all satisfied.'),
           r'Table~\ref{tab:header-distance} includes every layout and both legacy controls.',
           r'\begin{table}[t]',r'\centering\small',
           r'\caption{Information-set KL for independent header/task positions, averaged over 32 fresh problems. FF, FN, NF and NN denote header then task position (far or near). Legacy controls use the original whole-block layouts. Seed suffixes identify the same six fixed adaptations used throughout. All entries are nats.}',
           r'\label{tab:header-distance}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Model & FF & FN & NF (primary) & NN & Legacy far & Legacy near\\',r'\midrule']
    for role in roles:
        arm,seed=role.split('_seed');numbers=[report['checkpoints'][role][c]['info_kl']['mean'] for c in cells]
        lines.append(f'{arm.capitalize()} {seed[-2:]} & '+' & '.join(f'{v:.4g}' for v in numbers)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    lines += [r'\begin{table}[t]',r'\centering\small',
        r'\caption{Complete factorial contrasts within each checkpoint. $P_F,P_N$ are task-distance penalties with far/near header; $H_F,H_N$ are header-distance penalties with far/near task. $I=P_F-P_N$. All are paired mean KL differences in nats.}',
        r'\label{tab:header-factorial}',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'Model & $P_F$ & $P_N$ & $H_F$ & $H_N$ & $I$\\',r'\midrule']
    for role in roles:
        arm,seed=role.split('_seed')
        numbers=[report['within_checkpoint'][role][k]['mean'] for k in ('task_penalty_header_far','task_penalty_header_near','header_penalty_task_far','header_penalty_task_near','factorial_interaction')]
        lines.append(f'{arm.capitalize()} {seed[-2:]} & '+' & '.join(f'{v:.4g}' for v in numbers)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    effects=report['seed_averaged']
    for key,label in [('task_penalty_header_near_reduction','Reduction in task-distance penalty with nearby header'),('task_penalty_header_far_reduction','Reduction in task-distance penalty with distant header')]:
        e=effects[key];lines.append(f"{label}: {e['mean']:.4f} nats [{e['ci95'][0]:.4f}, {e['ci95'][1]:.4f}].")
    delta=max(abs(v[k]['mean']) for v in report['within_checkpoint'].values() for k in ('colocated_far_minus_legacy','colocated_near_minus_legacy'))
    lines.append(f"The largest absolute mean co-located-versus-legacy discrepancy is {delta:.4g} nats; these controls quantify the additional distractor regrouping.")
    resources=json.loads((FOLDER/'resource_audit.json').read_text())
    lines += [r'\paragraph{Scope and resources.}',
              'This test uses fresh prompts within the existing catalog and existing training lineage. The task-bearing block includes mapping and output-order information, so the contrast does not isolate factual retention from every other task requirement. No new latency certificate, independent training replication or natural-language result is claimed.',
              f"The six completed evaluations and one stopped startup attempt use {resources['total_scheduler_running_gpu_hours']:.2f} additional H100-hours. All are terminal; peak reservation is {resources['reservation_peak']} GPUs within the primary32 plus extra16 allocation. The stopped attempt produced no predictions and was manually replaced with the same frozen evaluation; its scheduler reports zero runtime despite startup logs."]
    observed=[r['utilization']['entire_observed_job'] for r in resources['jobs'] if not r['superseded_attempt']]
    busy=[r['mean_gpu_busy_percent'] for r in observed];memory=[r['mean_memory_used_percent'] for r in observed]
    lines.append(f'Across completed jobs, whole-job mean GPU busy is {min(busy):.1f}--{max(busy):.1f}\\%, and memory occupancy is {min(memory):.1f}--{max(memory):.1f}\\%. GPU busy is not achieved FLOP utilization. These measurements include startup, numerical screens and waiting for slower ranks; they do not satisfy an 80\\% utilization target.')
    (FOLDER/'paper_results.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.7),layout='constrained',sharey=True)
    for ax,arm in zip(axes,('near','balanced')):
        for j,seed in enumerate((20271011,20271012,20271013)):
            role=f'{arm}_seed{seed}';means=[report['checkpoints'][role][c]['info_kl']['mean'] for c in cells[:4]]
            ax.plot(range(4),means,'o-',label='Seed '+str(seed)[-2:])
        ax.axhline(4*np.log(2),color='gray',ls=':',label='One-call product floor')
        ax.set_yscale('log');ax.set_ylim(5e-7,30);ax.set_xticks(range(4),['Far/Far','Far/Near','Near/Far','Near/Near'],rotation=25)
        ax.set_title(arm.capitalize()+'-trained');ax.set_xlabel('Header / task position');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Information-set KL (nats)');axes[1].legend(frameon=False,fontsize=8)
    fig.supxlabel('32 fresh prompts; same six fixed checkpoints and selected starting lineage.',fontsize=8)
    fig.savefig(FOLDER/'header_task_distance.pdf');fig.savefig(FOLDER/'header_task_distance.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(primary=p,stronger_interpretation_supported=report['stronger_interpretation_supported']),indent=2))


if __name__=='__main__':main()
