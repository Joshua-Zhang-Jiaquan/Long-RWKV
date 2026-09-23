"""Paired problem-level analysis; training-seed effects remain explicit."""
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'results/distance_intervention'
SEEDS=(20271011,20271012,20271013)
POSITIONS=('far','middle','near')


def estimate(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('finite paired problem means required')
    rng=np.random.default_rng(20271022);indices=rng.integers(0,len(a),size=(10000,len(a)))
    means=a[indices].mean(axis=1)
    return dict(mean=float(a.mean()),ci95=[float(v) for v in np.quantile(means,[.025,.975])],problems=len(a))


def primary(data):
    rows={(r['index'],r['serial']['position']):r for r in data['conditions'] if r['primary']}
    if len(rows)!=96:raise ValueError('missing primary panel')
    return rows


def metric(rows,key,position=None,method='information_set'):
    result=[]
    for i in range(32):
        values=[]
        for p in POSITIONS if position is None else (position,):
            r=rows[i,p]
            if key.startswith('response.'):
                value=r['counterfactual']['response'][key.split('.',1)[1]]
            elif key.startswith('stages.'):
                value=r['stages'][key.split('.',1)[1]]
            else:value=next(a for a in r['exact']['results'] if a['method']==method)[key]
            values.append(value)
        result.append(np.mean(values))
    return np.asarray(result)


def summarize(data):
    rows=primary(data);out={}
    for position in (*POSITIONS,'mean_positions'):
        p=None if position=='mean_positions' else position
        info=metric(rows,'joint_kl_nats',p);one=metric(rows,'joint_kl_nats',p,'one');pair=metric(rows,'joint_kl_nats',p,'pair_preserving_halves')
        out[position]=dict(information_set_kl=estimate(info),one_kl=estimate(one),intact_pair_kl=estimate(pair),
                           kl_gain_over_one=estimate(one-info),same_call_kl_gain=estimate(pair-info),
                           same_call_estimation_difference=estimate(pair-info-4*math.log(2)),
                           valid_mass=estimate(metric(rows,'exact_valid_mass',p)),
                           first_error=estimate(metric(rows,'stages.first_round_estimation_nats',p)),
                           second_error=estimate(metric(rows,'stages.second_round_estimation_nats',p)),
                           response_contrast=estimate(metric(rows,'response.signed_logit_contrast',p)),
                           response_ce=estimate(metric(rows,'response.mean_correct_conditional_ce',p)),
                           response_floor=estimate(metric(rows,'response.response_ce_lower_bound',p)),
                           centering_penalty=estimate(metric(rows,'response.centering_penalty',p)),
                           both_variants_correct=estimate(metric(rows,'response.both_variants_correct',p)),
                           signed_probability_shift=estimate(metric(rows,'response.signed_probability_shift',p)),
                           unaffected_change=estimate(metric(rows,'response.unaffected_mean_abs_probability_change',p)))
    timings={}
    for method in ('one','information_set','pair_preserving_halves'):
        rr=[r for r in data['timings'] if r['method']==method]
        vals=[np.mean([v for r in rr if r['index']==i for v in r['seconds']]) for i in range(8)]
        timings[method]=dict(seconds=estimate(vals),requests=sum(len(r['seconds']) for r in rr),peak_allocated_gib=max(max(r['peak_allocated_bytes']) for r in rr)/2**30)
    secondary={}
    for length,pos in [(None,'evidence_only'),(4096,'far'),(4096,'middle'),(4096,'near'),(32768,'far'),(32768,'near')]:
        rr=[r for r in data['conditions'] if r['serial']['requested_context_tokens']==length and r['serial']['position']==pos]
        if len(rr)!=8:raise ValueError('secondary panel missing')
        secondary[f'{length}_{pos}']={m:estimate([next(a['joint_kl_nats'] for a in r['exact']['results'] if a['method']==m) for r in rr]) for m in ('one','information_set','pair_preserving_halves')}
    info=out['mean_positions']['information_set_kl'];gain=out['mean_positions']['kl_gain_over_one'];latency=timings['information_set']['seconds']
    return dict(primary=out,timing=timings,secondary=secondary,
                quality_cost_checks=dict(mean_kl_below_product_floor=info['mean']<4*math.log(2),
                                         upper_kl_ci_below_product_floor=info['ci95'][1]<4*math.log(2),
                                         mean_gain_over_one_positive=gain['mean']>0,
                                         mean_two_call_latency_within_budget=latency['mean']<=1.5,
                                         upper_mean_latency_ci_within_budget=latency['ci95'][1]<=1.5))


def contrasts(data):
    outputs={};pooled={}
    for seed in SEEDS:
        near=primary(data[f'near_seed{seed}']);balanced=primary(data[f'balanced_seed{seed}'])
        effects={}
        for position in POSITIONS:
            for name,key,sign in [('conditional_error_reduction','estimation_nats',1),
                                   ('second_error_reduction','stages.second_round_estimation_nats',1),
                                   ('response_ce_reduction','response.mean_correct_conditional_ce',1),
                                   ('response_contrast_increase','response.signed_logit_contrast',-1),
                                   ('centering_penalty_reduction','response.centering_penalty',1)]:
                values=sign*(metric(near,key,position)-metric(balanced,key,position))
                label=position+'_'+name;effects[label]=estimate(values);pooled.setdefault(label,[]).append(values)
        interaction=(metric(near,'estimation_nats','far')-metric(balanced,'estimation_nats','far'))-(metric(near,'estimation_nats','near')-metric(balanced,'estimation_nats','near'))
        effects['far_minus_near_error_reduction']=estimate(interaction)
        pooled.setdefault('far_minus_near_error_reduction',[]).append(interaction)
        outputs[str(seed)]=effects
    averaged={key:estimate(np.asarray(values).mean(axis=0)) for key,values in pooled.items()}
    return dict(per_adaptation_seed=outputs,seed_averaged_problem_effects=averaged,
                primary_far_error_reduction_ci_positive=averaged['far_conditional_error_reduction']['ci95'][0]>0,
                primary_far_direction_positive_all_seeds=all(v['far_conditional_error_reduction']['mean']>0 for v in outputs.values()),
                interval_scope='Paired problem resampling, conditional on these three adaptation seeds and selected starting lineage; not population training-seed uncertainty')


def main():
    roles=['original']+[f'{a}_seed{s}' for s in SEEDS for a in ('near','balanced')]
    data={r:json.loads((FOLDER/f'exact_{r}.json').read_text()) for r in roles}
    if not all(d['validated'] and d['condition_count']==144 for d in data.values()):raise ValueError('unvalidated panel')
    summary={role:summarize(d) for role,d in data.items()};effects=contrasts(data)
    report=dict(checkpoints=summary,intervention=effects,conditions=sum(d['condition_count'] for d in data.values()),
                timed_requests=sum(d['timed_requests'] for d in data.values()),
                caveats=['All adaptation runs share a selected starting checkpoint.',
                         'Fresh conditions use a previously observed held-out structure catalog.',
                         'Distance manipulation moves instructions together with equations.',
                         'Counterfactual responses use one fixed visible history; endpoint KL integrates all histories.',
                         'Same-call dependence differences are not isolated unless measured estimation errors are controlled.',
                         'A restricted task and implementation comparison does not establish general language-model superiority.'])
    (FOLDER/'final_report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Controlled distance-intervention results','',f"Validated {report['conditions']} exact conditions and {report['timed_requests']} timed requests.",'',
           '| Checkpoint | 16K info KL | Gain over one | Far evidence CE | Two-call seconds |','|---|---:|---:|---:|---:|']
    for role,r in summary.items():
        p=r['primary'];lines.append(f"| {role} | {p['mean_positions']['information_set_kl']['mean']:.4f} | {p['mean_positions']['kl_gain_over_one']['mean']:.4f} | {p['far']['response_ce']['mean']:.4f} | {r['timing']['information_set']['seconds']['mean']:.4f} |")
    lines+=['','Primary effect: near-only minus distance-balanced information-set conditional error at16K far.']
    for seed,r in effects['per_adaptation_seed'].items():
        s=r['far_conditional_error_reduction'];lines.append(f"- Seed {seed}: {s['mean']:.4f}, paired problem95%CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}].")
    s=effects['seed_averaged_problem_effects']['far_conditional_error_reduction']
    lines+=['',f"Seed-averaged effect: {s['mean']:.4f}, conditional problem95%CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}].",'',effects['interval_scope'],'',*['- '+c for c in report['caveats']]]
    (FOLDER/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(conditions=report['conditions'],timed_requests=report['timed_requests'],primary_effect=s),indent=2))


if __name__=='__main__':main()
