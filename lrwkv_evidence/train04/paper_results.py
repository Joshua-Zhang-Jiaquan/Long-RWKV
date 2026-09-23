"""Render measured trained-study tables; requires all three qualified evaluations."""
import argparse
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);args=ap.parse_args()
    data=json.loads(args.summary.read_text())
    if not data['complete_three_seed_study']:raise ValueError('all three seeds required before paper tables')
    lines=[r'\subsection{Measured outcomes}',r'All three terminal checkpoints are included. Table~\ref{tab:train04competence} separates denoiser competence from the two-call policy comparison. Table~\ref{tab:train04samplers} reports equal-seed descriptive means; full seed-specific results and source hashes are retained in the accompanying receipts.',
           r'\begin{table}[htbp]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\caption{Held-out competence and sampling, by training seed. Accuracy and validity are percentages; marginal KL is in nats per masked bit. The denoiser calibration panel is separate from the sampling conditions.}',r'\label{tab:train04competence}',
           r'\begin{tabular}{@{}rlrrrrr@{}}',r'\toprule',r'Seed & Family & Det. acc. & Marg. KL & Info valid & Random valid & One valid\\',r'\midrule']
    for run in data['runs']:
        for family,label in [('systematic','Systematic'),('paired_parity','Pair parity')]:
            cal=next(c for c in run['calibration'] if c['family']==family)
            cells={c['method']:c for c in run['cells'] if c['family']==family}
            lines.append(f"{run['seed']} & {label} & {100*cal['deterministic_accuracy']:.2f} & {cal['conditional_marginal_kl_per_masked_bit']:.3f} & {100*cells['information_set']['valid_fraction']:.2f} & {100*cells['random_halves']['valid_fraction']:.2f} & {100*cells['one']['valid_fraction']:.2f}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    for c in data['paired_contrasts']:
        lo,hi=c['conditional_paired_bootstrap95'];name=c['family'].replace('_',' ')
        lines.append(f"For {name}, the equal-seed information-set minus random-halves validity difference is {100*c['info_minus_random_validity']:.2f} percentage points, with paired condition-bootstrap95\\% interval $[{100*lo:.2f},{100*hi:.2f}]$. The interval conditions on the three fitted checkpoints.")
    lines += [r'\begin{table}[htbp]',r'\centering\small',r'\caption{Equal-seed decoder means. Coverage is the fraction of the 16 valid vectors observed among 32 draws per condition. Collision is the fraction of distinct draw pairs with identical outputs. Joint KL uses only two audit conditions per family and is not estimated for Bernoulli policies.}',r'\label{tab:train04samplers}',r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Family & Policy & Valid (\%) & Coverage (\%) & Collision (\%) & NFE & Joint KL\\',r'\midrule']
    names={'one':'One round','information_set':'Info set (2)','random_halves':'Random halves (2)','bernoulli2':'Bernoulli (2)','bernoulli4':'Bernoulli (4)','bernoulli8':'Bernoulli (8)','sequential':'Sequential (8)'}
    for c in data['equal_seed_means']:
        kl='---' if c['small_panel_exact_joint_kl_nats'] is None else f"{c['small_panel_exact_joint_kl_nats']:.3f}"
        family='Systematic' if c['family']=='systematic' else 'Pair parity'
        lines.append(f"{family} & {names[c['method']]} & {100*c['valid_fraction']:.2f} & {100*c['valid_support_coverage']:.2f} & {100*c['collision_probability']:.2f} & {c['nfe_per_draw']:.2f} & {kl}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'\begin{figure}[htbp]',r'\centering',r'\includegraphics[width=\linewidth]{results/train04/final/figures/kl_decomposition.pdf}',r'\caption{Exact fixed-policy endpoint KL split into discarded dependence and denoiser estimation error on the small audit panel. The bars average the same two conditions per family across all three training seeds.}',r'\label{fig:train04decomposition}',r'\end{figure}']
    (args.summary.parent/'paper_results.tex').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
