"""Strict receipts aggregation; preserves every seed and labels small-panel joint KL."""
import argparse
from collections import defaultdict
import json
import math
import random
from pathlib import Path
import statistics
from .evaluate import CONTRACT, METHODS, fixed_groups, valid
from . import tasks


def average(values): return sum(values)/len(values)

def aggregate(directory):
    rows=[]; hashes=set(); seeds=set(); ranks=set(); complete=set()
    for path in sorted(directory.glob('rank*.jsonl')):
        shard=[json.loads(line) for line in path.read_text().splitlines()]
        if not shard or shard[-1].get('kind')!='complete': raise ValueError(f'incomplete shard{path}')
        p=shard[0]
        if p['kind']!='provenance' or p['contract']!=CONTRACT or p['world']!=8: raise ValueError('wrong eval contract')
        hashes.add(p['checkpoint_sha256']); seeds.add(p['training_provenance']['seed'])
        ranks.add(p['rank']); complete.add(shard[-1]['rank']); rows+=shard
    if len(hashes)!=1 or len(seeds)!=1 or ranks!=set(range(8)) or complete!=ranks: raise ValueError('missing or mixed ranks')
    samples=[r for r in rows if r['kind']=='sampling']; calibration=[r for r in rows if r['kind']=='calibration']
    keys={(r['index'],r['method']) for r in samples}
    expected={(i,m) for i in range(32) for m in METHODS}
    if keys!=expected or len(samples)!=len(expected): raise ValueError('sampling coverage differs')
    if {r['index'] for r in calibration}!=set(range(128)) or len(calibration)!=128: raise ValueError('calibration coverage differs')
    conditions={r['index']:r['example'] for r in rows if r['kind']=='condition'}
    if set(conditions)!=set(range(32)): raise ValueError('condition coverage differs')
    for i,ex in conditions.items():
        if ex != tasks.make_example('test',CONTRACT['data_seed'],i): raise ValueError('held-out manifest differs')
    for r in samples:
        ex=conditions[r['index']]
        if r['family']!=ex['family'] or len(r['samples'])!=32: raise ValueError('sample metadata differs')
        if any(len(b)!=8 or any(type(v) is not int or v not in (0,1) for v in b) for b in r['samples']): raise ValueError('invalid binary receipt')
        good=[tuple(b) for b in r['samples'] if valid(ex,b)]
        if len(good)!=r['valid'] or len(set(good))/16 != r['valid_support_coverage']: raise ValueError('recomputed validity/coverage differs')
        if r['method'] in ('information_set','random_halves') and r['actual_sampling_nfe']!=64: raise ValueError('two-call comparison violated')
    cells=[]
    for family in tasks.FAMILIES:
        for method in METHODS:
            group=[r for r in samples if r['family']==family and r['method']==method]
            if len(group)!=16 or any(r['draws']!=32 for r in group): raise ValueError('cell coverage differs')
            values=[r['valid']/r['draws'] for r in group]
            kls=[r['exact_joint_kl_nats'] for r in group if r['exact_joint_kl_nats'] is not None]
            audited=[r for r in group if r['exact_joint_kl_nats'] is not None]
            dependence=[]
            for r in audited:
                ex=conditions[r['index']];groups=fixed_groups(ex,method);assignment={i:k for k,g in enumerate(groups) for i in g}
                cost=0.
                if family=='paired_parity':
                    cost=sum(len({assignment[i] for i in range(8) if row>>i&1})==1 for row in ex['matrix_rows'])*math.log(2)
                if r['exact_joint_kl_nats'] < cost-1e-8: raise ValueError('KL below oracle dependence cost')
                dependence.append(cost)
            cells.append(dict(family=family,method=method,valid_fraction=average(values),
                              condition_standard_error=statistics.stdev(values)/math.sqrt(len(values)),
                              valid_support_coverage=average([r['valid_support_coverage'] for r in group]),
                              collision_probability=average([r['collision_probability'] for r in group]),
                              max_output_fraction=average([r['max_output_fraction'] for r in group]),
                              nfe_per_draw=sum(r['actual_sampling_nfe'] for r in group)/512,
                              sampling_seconds=sum(r['sampling_seconds'] for r in group),
                              small_panel_exact_joint_kl_nats=average(kls) if kls else None,
                              joint_audit_conditions=len(kls),
                              small_panel_dependence_cost_nats=average(dependence) if dependence else None,
                              small_panel_estimation_error_nats=average(kls)-average(dependence) if dependence else None))
    cal=[]
    for family in tasks.FAMILIES:
        values=[v for r in calibration if r['family']==family for v in r['rows']]
        den=sum(v['deterministic'] for v in values); masked=sum(v['masked'] for v in values); fair=sum(v['fair_count'] for v in values)
        cal.append(dict(family=family,conditional_marginal_kl_per_masked_bit=sum(v['marginal_kl_sum_nats'] for v in values)/masked,
                        elbo_excess_per_target_token=average([v['elbo_excess_per_token'] for v in values]),
                        deterministic_accuracy=sum(v['deterministic_correct'] for v in values)/den,
                        deterministic_count=den, fair_probability_abs_error=sum(v['fair_abs_deviation_sum'] for v in values)/fair))
    contrasts=[]
    for family in tasks.FAMILIES:
        by={(r['index'],r['method']):r for r in samples if r['family']==family}
        deltas=[(by[i,'information_set']['valid']-by[i,'random_halves']['valid'])/32 for i,m in by if m=='information_set']
        contrasts.append(dict(family=family,info_minus_random_validity=average(deltas),paired_condition_standard_error=statistics.stdev(deltas)/math.sqrt(len(deltas))))
    return dict(seed=next(iter(seeds)),checkpoint_sha256=next(iter(hashes)),source=str(directory),cells=cells,calibration=cal,contrasts=contrasts,
                elapsed_gpu_rank_seconds=sum(r['elapsed_seconds'] for r in rows if r['kind']=='complete'))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directories',type=Path,nargs='+'); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    results=sorted([aggregate(d) for d in args.directories],key=lambda r:r['seed'])
    if len({r['seed'] for r in results})!=len(results): raise ValueError('duplicate training seed')
    args.out.mkdir(parents=True,exist_ok=True)
    means=[]
    for family in tasks.FAMILIES:
        for method in METHODS:
            cells=[c for r in results for c in r['cells'] if c['family']==family and c['method']==method]
            numeric=('valid_fraction','valid_support_coverage','collision_probability','nfe_per_draw','small_panel_exact_joint_kl_nats','small_panel_dependence_cost_nats','small_panel_estimation_error_nats')
            means.append(dict(family=family,method=method,**{k:average([c[k] for c in cells]) if cells[0][k] is not None else None for k in numeric}))
    paired=[]
    for family in tasks.FAMILIES:
        per_condition=defaultdict(list)
        for run in results:
            raw=[json.loads(line) for p in Path(run['source']).glob('rank*.jsonl') for line in p.read_text().splitlines()]
            readings={(r['index'],r['method']):r for r in raw if r['kind']=='sampling' and r['family']==family}
            for i,m in readings:
                if m=='information_set': per_condition[i].append((readings[i,'information_set']['valid']-readings[i,'random_halves']['valid'])/32)
        deltas=[average(v) for _,v in sorted(per_condition.items())];rng=random.Random(20260922)
        boot=sorted(average([rng.choice(deltas) for _ in deltas]) for _ in range(20000))
        paired.append(dict(family=family,info_minus_random_validity=average(deltas),conditional_paired_bootstrap95=[boot[499],boot[19499]],conditions=len(deltas),meaning='Condition resampling with allseed/sampler pairs kept together; conditional on fittedcheckpoints, not trainingpopulation uncertainty'))
    result=dict(contract=CONTRACT,runs=results,equal_seed_means=means,paired_contrasts=paired,complete_three_seed_study={r['seed'] for r in results}=={17,29,43},
                perfect_uniform_reference=dict(validity=1,expected_coverage_32_draws=1-(15/16)**32,collision_probability=1/16),
                caveats=['Restricted binary synthetic task, not full-vocabulary language capability','Information-set schedule uses known A; no learned scheduling claim','Marginal KL and small-panel fixed-policy joint KL are distinct','Only3trainedreplicates; condition SE is conditional on checkpoint','FLA backend official-RWKV kernel equivalence not established'])
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Trained 0.4B controlled posterior study','',f"Completed training seeds: {', '.join(str(r['seed']) for r in results)}.",'',
           'All decoders use the same terminal 500-step checkpoint within each seed. Each family has16 held-out conditions ×32 samples; calibration uses64 additional conditions across8stages. The two-round information-set schedule receives the known constraint matrix, not the gold target.','',
           '| Seed | Family | Decoder | Valid % | Valid support coverage % | NFE/draw | Small-panel joint KL (nats) |',
           '|---|---|---|---:|---:|---:|---:|']
    for run in results:
        for c in run['cells']:
            kl='—' if c['small_panel_exact_joint_kl_nats'] is None else f"{c['small_panel_exact_joint_kl_nats']:.4f}"
            lines.append(f"| {run['seed']} | {c['family']} | {c['method']} | {100*c['valid_fraction']:.2f} | {100*c['valid_support_coverage']:.2f} | {c['nfe_per_draw']:.2f} | {kl} |")
    lines+=['','## Matched two-round contrast','']
    for c in paired:
        lo,hi=c['conditional_paired_bootstrap95']
        lines.append(f"{c['family']}: information set minus random halves = {100*c['info_minus_random_validity']:.2f} percentage points;95%paired condition-bootstrap interval[{100*lo:.2f},{100*hi:.2f}]pp, conditional on the fitted checkpoints.")
    lines+=['','## Conditional denoiser competence','','| Seed | Family | Deterministic-bit accuracy % | Marginal KL/bit (nats) | ELBO excess/token (nats) |','|---|---|---:|---:|---:|']
    for run in results:
        for c in run['calibration']: lines.append(f"| {run['seed']} | {c['family']} | {100*c['deterministic_accuracy']:.2f} | {c['conditional_marginal_kl_per_masked_bit']:.4f} | {c['elbo_excess_per_target_token']:.4f} |")
    lines+=['','The perfect uniform posterior has validity100%, expected valid-support coverage87.32% with32draws, and collision probability6.25%. Exact fixed-policy joint KL is measured on only2conditions per family. Empirical coverage is sample-limited; no plug-in joint KL is reported.','',
            'The oracle paired-parity one-round validity is6.25%, while the oracle information-set and sequential schedules are exact. Learned failures must be separated from this oracle dependence penalty. This study cannot establish language-model superiority, long-context capability, or mathematical novelty.','']
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(args.out),seeds=[r['seed'] for r in results]),indent=2))
if __name__=='__main__':main()
