"""Presentation only: transfer every frozen selector from final reports to TeX."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'results/distance_intervention'
SEEDS = (20271011, 20271012, 20271013)


def number(value):
    if value != 0 and abs(value) < .001:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    return f'{value:.4f}'


def interval(row):
    lo, hi = row['ci95']
    return f"{row['mean']:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    report = json.loads((FOLDER / 'final_report.json').read_text())
    cost = json.loads((ROOT / 'results/distance_intervention_cost/budget_certificates.json').read_text())
    roles = [f'{arm}_seed{seed}' for seed in SEEDS for arm in ('near', 'balanced')]
    certificates = {r['role']: r for r in cost['certificates']}
    if report['conditions'] != 1008 or set(certificates) != set(roles) or len(cost['certificates']) != 6:
        raise ValueError('All fixed selectors are required; partial tables are not emitted')
    lines = [r'\begin{tabular}{lrrl}', r'\toprule',
             r'Adaptation seed & Near-only far KL & Balanced far KL & Reduction [95\% CI] \\',
             r'\midrule']
    for seed in SEEDS:
        a = report['checkpoints'][f'near_seed{seed}']['primary']['far']['information_set_kl']['mean']
        b = report['checkpoints'][f'balanced_seed{seed}']['primary']['far']['information_set_kl']['mean']
        delta = report['intervention']['per_adaptation_seed'][str(seed)]['far_conditional_error_reduction']
        lines.append(f'{seed} & {number(a)} & {number(b)} & {interval(delta)}' + r' \\')
    delta = report['intervention']['seed_averaged_problem_effects']['far_conditional_error_reduction']
    lines += [r'\midrule', 'Seed mean & --- & --- & ' + interval(delta) + r' \\',
              r'\bottomrule', r'\end{tabular}']
    (FOLDER / 'paper_intervention_table.tex').write_text('\n'.join(lines) + '\n')
    lines = [r'\begin{tabular}{llrrrrl}', r'\toprule',
             r'Arm & Seed & 16K KL & Equal-call gain & Matched KL & Two-call (s) & Certificate \\',
             r'\midrule']
    for seed in SEEDS:
        for arm in ('near', 'balanced'):
            role = f'{arm}_seed{seed}'
            summary = report['checkpoints'][role]['primary']['mean_positions']
            cert = certificates[role]
            fields = ['Near-only' if arm == 'near' else 'Balanced', str(seed),
                      number(summary['information_set_kl']['mean']),
                      number(summary['same_call_kl_gain']['mean']),
                      number(cert['quality']['information_set']['mean']),
                      f"{cert['recurrent_timing']['information_set']['mean']:.3f}",
                      'Pass' if cert['empirical_sufficient_certificate'] else 'Fail']
            lines.append(' & '.join(fields) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (FOLDER / 'paper_quality_cost_table.tex').write_text('\n'.join(lines) + '\n')
    cells = ['None_evidence_only', '4096_far', '4096_middle', '4096_near', '32768_far', '32768_near']
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Model & Short & 4K far & 4K middle & 4K near & 32K far & 32K near \\',
             r'\midrule']
    for role in ['original'] + roles:
        label = 'Original' if role == 'original' else role.replace('_seed202710', ' ')
        row = report['checkpoints'][role]['secondary']
        lines.append(' & '.join([label] + [number(row[c]['information_set']['mean']) for c in cells]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (FOLDER / 'paper_secondary_table.tex').write_text('\n'.join(lines) + '\n')
    print('Wrote all-seed intervention and matched quality/cost tables.')


if __name__ == '__main__':
    main()
