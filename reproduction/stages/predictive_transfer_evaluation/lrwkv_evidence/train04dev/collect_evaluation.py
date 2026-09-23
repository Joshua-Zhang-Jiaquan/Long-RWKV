"""Collect all development shards and retain per-condition paired contrasts."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics



def validate_endpoint(row):
    """Independently recompute probability identities from the stored endpoint law."""
    n=row['n'];logs=row['endpoint_log_probabilities'];points=row['valid_endpoints']
    if n not in (2,4,8) or len(logs)!=2**(n//2) or len(points)!=len(logs):
        raise ValueError('wrong endpoint support size')
    if len({tuple(p) for p in points})!=len(points) or any(len(p)!=n or any(b not in (0,1) for b in p) for p in points):
        raise ValueError('invalid or duplicate endpoint vectors')
    fields=('joint_kl_nats','exact_valid_mass','dependence_cost_nats','estimation_error_nats','conditional_valid_kl_nats','decomposition_residual')
    if not all(math.isfinite(x) for x in logs) or not all(math.isfinite(row[k]) for k in fields):
        raise ValueError('nonfinite endpoint evidence')
    maximum=max(logs);logmass=maximum+math.log(sum(math.exp(x-maximum) for x in logs))
    kl=-math.log(len(logs))-statistics.mean(logs)
    checks=((kl,row['joint_kl_nats']),(math.exp(logmass),row['exact_valid_mass']),
            (kl+logmass,row['conditional_valid_kl_nats']),
            (kl-row['dependence_cost_nats']-row['estimation_error_nats'],row['decomposition_residual']))
    if any(abs(a-b)>1e-8 for a,b in checks):raise ValueError('stored endpoint metric mismatch')
    if logmass>1e-8 or kl < -1e-8 or kl+logmass < -1e-8 or abs(row['decomposition_residual'])>1e-8:
        raise ValueError('invalid probability or KL decomposition')
    if row['dependence_cost_nats'] < -1e-8 or row['estimation_error_nats'] < -1e-8:
        raise ValueError('negative KL component')


def collect(root):
    rows=[];provenance=[];responses=[]
    for rank in range(8):
        shard=[json.loads(x) for x in (root/f'rank{rank}.jsonl').read_text().splitlines()]
        assert shard[0]['kind']=='provenance' and shard[0]['rank']==rank and shard[0]['world']==8
        assert shard[-1]['kind']=='complete'
        provenance.append(shard[0]);rows.extend(r for r in shard if r['kind']=='endpoint')
        responses.extend(r for r in shard if r['kind']=='history_response')
    common=('checkpoint_sha256','step','objective','split','seed','conditions_per_dimension','dimensions')
    if any(any(p[k]!=provenance[0][k] for k in common) for p in provenance):raise ValueError('shard provenance mismatch')
    # The phase300 schema predates mask_law; require consistent presence/value.
    if any(p.get('mask_law')!=provenance[0].get('mask_law') for p in provenance):raise ValueError('shard mask law mismatch')
    for key in ('triton_f32_default','torch_tf32','serialization','history_coupling'):
        if any(p.get(key)!=provenance[0].get(key) for p in provenance):raise ValueError('shard precision mismatch')
    expected={(n,i,m) for n in (2,4,8) for i in range(16) for m in ('one','information_set','random_halves')}
    by={(r['n'],r['index'],r['method']):r for r in rows};assert len(rows)==len(by) and set(by)==expected
    grouped=defaultdict(list)
    for r in rows:
        validate_endpoint(r)
        grouped[(r['n'],r['family'],r['method'])].append(r)
    fields=('joint_kl_nats','exact_valid_mass','dependence_cost_nats','estimation_error_nats','conditional_valid_kl_nats')
    summary=[dict(n=n,family=f,method=m,conditions=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in fields}) for (n,f,m),rr in sorted(grouped.items())]
    contrasts=[]
    for n in (2,4,8):
        for i in range(16):
            a,b=by[n,i,'information_set'],by[n,i,'random_halves']
            contrasts.append(dict(n=n,index=i,family=a['family'],information_set_minus_random_valid_mass=a['exact_valid_mass']-b['exact_valid_mass'],
                                  random_minus_information_set_kl=b['joint_kl_nats']-a['joint_kl_nats']))
    response_summary=[]
    if responses:
        keys={(r['n'],r['index']) for r in responses}
        if len(responses)!=48 or keys!={(n,i) for n in (2,4,8) for i in range(16)}:raise ValueError('incomplete history response panel')
        response_groups=defaultdict(list)
        metrics=('paired_signed_response','off_pair_absolute_response','systematic_absolute_response')
        for r in responses:
            if r['additional_model_calls']!=0:raise ValueError('unexpected diagnostic model calls')
            for key in metrics:
                value=r[key]
                if value is not None and (not math.isfinite(value) or not (-1.00000001<=value<=1.00000001)):
                    raise ValueError('invalid history response')
                if key!='paired_signed_response' and value is not None and value < -1e-8:raise ValueError('negative absolute response')
            response_groups[r['n'],r['family']].append(r)
        for (n,family),rr in sorted(response_groups.items()):
            values={key:[r[key] for r in rr if r[key] is not None] for key in metrics}
            response_summary.append(dict(n=n,family=family,conditions=len(rr),**{key:statistics.mean(v) if v else None for key,v in values.items()}))
    return dict(provenance=provenance[0],scope='Development panel on one checkpoint; no confirmatory significance claim',summary=summary,paired_condition_contrasts=contrasts,history_response_summary=response_summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=collect(a.root);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
