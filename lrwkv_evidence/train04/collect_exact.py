"""Audit and summarize post-primary exact endpoint probabilities, retaining primary results."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from . import tasks
from .evaluate import CONTRACT,support


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    rows=[];provenance=[]
    for rank in range(8):
        p=args.directory/f'rank{rank}.jsonl';data=[json.loads(s) for s in p.read_text().splitlines()]
        if data[-1]['kind']!='complete' or data[-1]['rank']!=rank:raise ValueError('incomplete exact shard')
        if data[0]['kind']!='provenance' or data[0]['rank']!=rank:raise ValueError('wrong provenance')
        provenance.append(data[0]);rows.extend(r for r in data if r['kind']=='exact')
    expected={(s,i,m) for s in (17,29,43) for i in range(32) for m in ('information_set','random_halves')}
    if {(r['seed'],r['index'],r['method']) for r in rows}!=expected or len(rows)!=len(expected):raise ValueError('exact coverage mismatch')
    for r in rows:
        ex=tasks.make_example('test',CONTRACT['data_seed'],r['index'])
        if r['instance_id']!=ex['instance_id'] or r['valid_endpoints']!=[list(v) for v in support(ex)]:raise ValueError('wrong support')
        if len(r['endpoint_log_probabilities'])!=16:raise ValueError('wrong support size')
        mass=math.fsum(math.exp(v) for v in r['endpoint_log_probabilities']);kl=-math.log(16)-math.fsum(r['endpoint_log_probabilities'])/16
        if abs(mass-r['exact_valid_mass'])>1e-12 or abs(kl-r['joint_kl_nats'])>1e-12:raise ValueError('endpoint recomputation differs')
        if abs(kl+math.log(mass)-r['conditional_valid_kl_nats'])>1e-12:raise ValueError('validity/diversity identity differs')
    cells=[]
    fields=('exact_valid_mass','joint_kl_nats','invalid_mass_log_loss','conditional_valid_kl_nats','dependence_cost_nats','estimation_error_nats')
    for seed in (17,29,43):
        for family in tasks.FAMILIES:
            for method in ('information_set','random_halves'):
                cell=[r for r in rows if (r['seed'],r['family'],r['method'])==(seed,family,method)]
                if len(cell)!=16:raise ValueError('bad cell size')
                cells.append(dict(seed=seed,family=family,method=method,**{k:math.fsum(r[k] for r in cell)/16 for k in fields}))
    means=[]
    for family in tasks.FAMILIES:
        a=[r for r in cells if r['family']==family and r['method']=='information_set'];b=[r for r in cells if r['family']==family and r['method']=='random_halves']
        means.append(dict(family=family,information_set_valid_mass=math.fsum(r['exact_valid_mass'] for r in a)/3,
                          random_halves_valid_mass=math.fsum(r['exact_valid_mass'] for r in b)/3,
                          info_minus_random_validity_pp=100*math.fsum(x['exact_valid_mass']-y['exact_valid_mass'] for x,y in zip(a,b))/3,
                          information_set_joint_kl=math.fsum(r['joint_kl_nats'] for r in a)/3,
                          random_halves_joint_kl=math.fsum(r['joint_kl_nats'] for r in b)/3))
    result=dict(scope='Post-primary exact enumeration on all existing32testconditions per checkpoint; no checkpoint or hyperparameter selection',
                methods=['information_set','random_halves'],seeds=[17,29,43],exact_rows=len(rows),provenance=provenance,cells=cells,equal_seed_means=means,
                caveat='Exact finite-panel model expectations in floating-point arithmetic; not a confidence statement for unseen conditions. Audit memoization is not sampler timing evidence.')
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'exact_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Post-primary exact endpoint audit','','All16valid endpoint probabilities were enumerated for both two-call policies on each of the32primary test conditions, for all three locked checkpoints. The primary sampled results remain unchanged.','',
           '| Family | Exact info validity % | Exact random validity % | Difference pp | Info joint KL | Random joint KL |','|---|---:|---:|---:|---:|---:|']
    tex=[r'\paragraph{Post-primary exact-probability diagnostic.}',r'The sampled parity contrast motivated an additional diagnostic, without changing any checkpoint, hyperparameter or sampler. We enumerated all 16 valid endpoint probabilities for both two-round policies on all 32 existing test conditions per checkpoint. This removes Monte Carlo error in finite-panel validity, but does not remove uncertainty about unseen conditions. Audit memoization is not a decoding-speed measurement.',r'\begin{table}[htbp]',r'\centering\small',r'\caption{Post-primary exact model expectations, equal-seed means on the existing conditions. The sampled tables above are retained unchanged.}',r'\label{tab:train04exact}',r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Family & Info valid (\%) & Random valid (\%) & Difference (pp) & Info KL & Random KL\\',r'\midrule']
    for r in means:
        lines.append(f"| {r['family']} | {100*r['information_set_valid_mass']:.5f} | {100*r['random_halves_valid_mass']:.5f} | {r['info_minus_random_validity_pp']:.5f} | {r['information_set_joint_kl']:.5f} | {r['random_halves_joint_kl']:.5f} |")
        label='Systematic' if r['family']=='systematic' else 'Pair parity'
        tex.append(f"{label} & {100*r['information_set_valid_mass']:.4f} & {100*r['random_halves_valid_mass']:.4f} & {r['info_minus_random_validity_pp']:.4f} & {r['information_set_joint_kl']:.4f} & {r['random_halves_joint_kl']:.4f}"+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (args.out/'EXACT_AUDIT.md').write_text('\n'.join(lines)+'\n');(args.out/'exact_paper_results.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(means,indent=2))
if __name__=='__main__':main()
