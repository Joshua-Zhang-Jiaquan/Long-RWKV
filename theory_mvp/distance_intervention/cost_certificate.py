"""Same-eight-problem empirical quality/cost check for all fixed selectors."""
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
from report import estimate


def certificate(attention, recurrent, budget=1.5):
    a={r['method']:r for r in attention['summary']}
    if {r['index'] for r in attention['conditions']}!=set(range(8)):raise ValueError('wrong matched attention cases')
    primary=[r for r in recurrent['conditions'] if r['primary'] and r['index']<8]
    if len(primary)!=24:raise ValueError('missing matched quality cases')
    if {(r['index'],r['serial']['position']) for r in primary}!={(i,p) for i in range(8) for p in ('far','middle','near')}:raise ValueError('mismatched positions')
    quality={};timing={}
    for method in ('one','information_set','pair_preserving_halves'):
        q=[np.mean([next(m['joint_kl_nats'] for m in r['exact']['results'] if m['method']==method) for r in primary if r['index']==i]) for i in range(8)]
        quality[method]=estimate(q)
        tr=[r for r in recurrent['timings'] if r['method']==method]
        if len(tr)!=24 or {r['index'] for r in tr}!=set(range(8)):raise ValueError('wrong recurrent timing cases')
        t=[np.mean([v for r in tr if r['index']==i for v in r['seconds']]) for i in range(8)]
        timing[method]=estimate(t)
    checks=dict(recurrent_two_call_fits=timing['information_set']['mean']<=budget,
                attention_one_call_fits=a['one']['mean_request_seconds']<=budget,
                attention_information_set_excluded=a['information_set']['mean_request_seconds']>budget,
                attention_intact_pair_excluded=a['pair_preserving_halves']['mean_request_seconds']>budget,
                recurrent_kl_below_product_floor=quality['information_set']['mean']<4*math.log(2))
    return dict(role=recurrent['role'],checkpoint_sha256=recurrent['checkpoint_sha256'],matched_problem_indices=list(range(8)),
                budget_seconds=budget,quality=quality,recurrent_timing=timing,attention_timing=a,
                conditions=checks,empirical_sufficient_certificate=all(checks.values()),
                scope='Eight-problem matched empirical law, three positions; plugin mean certificate, not population or per-request guarantee. All six selectors retained. Distinct from the32-problem primary mechanism analysis.')


def main():
    folder=ROOT/'results/distance_intervention_cost';attention=json.loads((folder/'exact_attention.json').read_text())
    results=[]
    for seed in (20271011,20271012,20271013):
        for arm in ('near','balanced'):
            role=f'{arm}_seed{seed}'
            data=json.loads((ROOT/'results/distance_intervention'/f'exact_{role}.json').read_text())
            results.append(certificate(attention,data))
    report=dict(certificates=results,attention_timed_requests=360,recurrent_timed_requests=2160,
                scope='Frozen matched-cost supplement; no additional attention quality experiment.')
    (folder/'budget_certificates.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Matched fresh-condition quality/cost certificates','',
           'Same eight paired problems and three16K positions for quality and timing. Mean budget1.5s. All fixed adaptation checkpoints retained.','',
           '| Checkpoint | Matched KL | Two-call seconds | Sufficient certificate |','|---|---:|---:|---|']
    for r in results:lines.append(f"| {r['role']} | {r['quality']['information_set']['mean']:.4f} | {r['recurrent_timing']['information_set']['mean']:.4f} | {r['empirical_sufficient_certificate']} |")
    lines+=['',results[0]['scope']]
    (folder/'BUDGET_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([dict(role=r['role'],certificate=r['empirical_sufficient_certificate']) for r in results],indent=2))


if __name__=='__main__':main()
