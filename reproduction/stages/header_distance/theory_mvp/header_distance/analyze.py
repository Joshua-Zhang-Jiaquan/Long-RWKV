"""Frozen complete-panel validation and paired factorial analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.header_distance import tasks as T
from lrwkv_evidence.long_context_eval.numerical import validate

FOLDER=ROOT/'results/header_distance'
MANIFEST=ROOT/'theory_mvp/header_distance/FROZEN.json'
SEEDS=(20271011,20271012,20271013)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def collect(role):
    manifest=json.loads(MANIFEST.read_text());plan=json.loads((FOLDER/f'plan_{role}.json').read_text())
    if plan['manifest_sha256']!=sha(MANIFEST):raise ValueError('manifest changed')
    for rel,digest in plan['sources'].items():
        if sha(Path(plan['stage'])/rel)!=digest:raise ValueError('staged source changed: '+rel)
    if sha(plan['checkpoint'])!=manifest['checkpoints'][role]['sha256']:raise ValueError('checkpoint changed')
    panelpath=ROOT/'theory_mvp/header_distance/INPUTS.json'
    if sha(panelpath)!=manifest['sources']['theory_mvp/header_distance/INPUTS.json']:raise ValueError('input panel changed')
    panel=json.loads(panelpath.read_text())['conditions'];inputs={(r['index'],r['cell']):r['serial'] for r in panel}
    conditions=[];files={};provenances=[]
    for rank in range(8):
        path=Path(plan['out'])/f'rank{rank}.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        files[str(path)]=sha(path)
        if rows[-1]!=dict(kind='complete',conditions=24,all_declared_cells_retained=True):raise ValueError('rank incomplete')
        provenance=[r for r in rows if r['kind']=='provenance']
        if len(provenance)!=1:raise ValueError('invalid provenance count')
        pr=provenance[0]
        for k,v in dict(rank=rank,world=8,role=role,seed=T.SEED,followup='header_distance_v1',checkpoint_sha256=plan['checkpoint_sha256'],manifest_sha256=sha(MANIFEST)).items():
            if pr[k]!=v:raise ValueError('bad provenance: '+k)
        provenances.append(pr)
        screens=[r for r in rows if r['kind']=='numerical_screen']
        if len(screens)!=1:raise ValueError('missing numerical screen')
        validate(screens[0]['rows'])
        factorial=[r for r in rows if r['kind']=='numerical_factorial']
        if len(factorial)!=1:raise ValueError('missing split-layout numerical screen')
        expected=[(c,h) for c in T.CELLS[:4] for h in ('all_masked','information_set_zero')]
        if [(r['cell'],r['history']) for r in factorial[0]['rows']]!=expected:raise ValueError('incomplete factorial screen')
        for r in factorial[0]['rows']:
            if any(not math.isfinite(r[k]) or r[k]<0 or r[k]>tol for k,tol in [('within',1e-5),('cross',1e-4)]):raise ValueError('factorial numerical failure')
        rc=[r for r in rows if r['kind']=='condition'];keys=[(r['index'],r['cell']) for r in rc]
        if len(keys)!=24 or set(keys)!={(i,c) for i in range(rank,32,8) for c in T.CELLS}:raise ValueError('missing or duplicate conditions')
        for r in rc:
            if r['serial']!=inputs[r['index'],r['cell']]:raise ValueError('input mismatch')
            actual=T.endpoint(T.example(r['index']),r['initial'],r['conditional'])
            for key,value in actual.items():
                if np.max(np.abs(np.asarray(value)-np.asarray(r['metrics'][key])))>1e-9:raise ValueError('endpoint mismatch: '+key)
            conditions.append(dict(index=r['index'],cell=r['cell'],**actual))
    return dict(role=role,validated=True,manifest_sha256=sha(MANIFEST),checkpoint_sha256=plan['checkpoint_sha256'],
                conditions=conditions,raw_file_sha256=files,provenance=provenances)


def estimate(values):
    values=np.asarray(values,dtype=float)
    if values.shape!=(32,) or not np.isfinite(values).all():raise ValueError('32 finite paired problem values required')
    samples=np.random.default_rng(2026092302).integers(0,32,size=(10000,32))
    ci=np.quantile(values[samples].mean(axis=1),[.025,.975])
    return dict(mean=float(values.mean()),ci95=ci.tolist(),problems=32)


def metric(data,cell,key='info_kl'):
    rows={r['index']:r for r in data['conditions'] if r['cell']==cell}
    if set(rows)!=set(range(32)):raise ValueError('incomplete cell')
    return np.array([rows[i][key] for i in range(32)])


def factorial(data):
    ff=metric(data,'header_far_task_far');fn=metric(data,'header_far_task_near')
    nf=metric(data,'header_near_task_far');nn=metric(data,'header_near_task_near')
    return dict(task_penalty_header_far=ff-fn,task_penalty_header_near=nf-nn,
                header_penalty_task_far=ff-nf,header_penalty_task_near=fn-nn,
                factorial_interaction=(ff-fn)-(nf-nn),
                colocated_far_minus_legacy=ff-metric(data,'legacy_far'),
                colocated_near_minus_legacy=nn-metric(data,'legacy_near'))


def report():
    manifest=json.loads(MANIFEST.read_text());data={role:json.loads((FOLDER/f'exact_{role}.json').read_text()) for role in manifest['roles']}
    if any(not d['validated'] or d['manifest_sha256']!=sha(MANIFEST) or len(d['conditions'])!=192 for d in data.values()):raise ValueError('unvalidated panel')
    keys=('info_kl','one_kl','gain_over_one','first_error','second_error','valid_mass_info')
    summaries={role:{c:{k:estimate(metric(d,c,k)) for k in keys} for c in T.CELLS} for role,d in data.items()}
    within={r:factorial(d) for r,d in data.items()};effects={};pooled={}
    for seed in SEEDS:
        nr=f'near_seed{seed}';br=f'balanced_seed{seed}';n,b=data[nr],data[br]
        values={c+'_error_reduction':metric(n,c)-metric(b,c) for c in T.CELLS}
        values.update({k+'_reduction':within[nr][k]-within[br][k] for k in within[nr]})
        values['primary_second_error_reduction']=metric(n,'header_near_task_far','second_error')-metric(b,'header_near_task_far','second_error')
        effects[str(seed)]={k:estimate(v) for k,v in values.items()}
        for k,v in values.items():pooled.setdefault(k,[]).append(v)
    pooled={k:estimate(np.mean(v,axis=0)) for k,v in pooled.items()}
    primary=pooled['header_near_task_far_error_reduction']
    passes={str(s):all(summaries[f'balanced_seed{s}'][c]['info_kl']['mean']<4*math.log(2) for c in ('header_near_task_far','header_near_task_near')) for s in SEEDS}
    result=dict(manifest_sha256=sha(MANIFEST),conditions=1152,endpoint_laws=2304,
                checkpoints=summaries,within_checkpoint={r:{k:estimate(v) for k,v in f.items()} for r,f in within.items()},
                per_seed=effects,seed_averaged=pooled,primary=primary,
                primary_positive_interval=primary['ci95'][0]>0,
                primary_direction_positive_all_seeds=all(effects[str(s)]['header_near_task_far_error_reduction']['mean']>0 for s in SEEDS),
                balanced_primary_and_near_control_below_product_floor=passes,
                scope='Fresh prompts from a previously observed structure catalog, conditional on six fixed adaptation checkpoints and one selected starting lineage; inference layout intervention does not separate components of the earlier training treatment.')
    result['stronger_interpretation_supported']=result['primary_positive_interval'] and result['primary_direction_positive_all_seeds'] and all(passes.values())
    (FOLDER/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Independent header/task distance results','',result['scope'],'',
           '| Checkpoint | Header far/task far | Header far/task near | Header near/task far | Both near | Legacy far | Legacy near |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for role in manifest['roles']:lines.append('| '+role+' | '+' | '.join(f"{summaries[role][c]['info_kl']['mean']:.7g}" for c in T.CELLS)+' |')
    lines+=['',f"Primary error reduction: {primary['mean']:.7g} nats, conditional95%CI {primary['ci95']}.",
            f"All three primary directions positive: {result['primary_direction_positive_all_seeds']}.",
            f"Balanced primary and nearby-control quality conditions: {passes}.",
            f"Predeclared stronger interpretation supported: {result['stronger_interpretation_supported']}.",
            '', 'Every layout and legacy control is retained; full factorial contrasts, first/second errors and one-call comparisons are in report.json.']
    (FOLDER/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoints','within_checkpoint','per_seed','seed_averaged')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role');args=parser.parse_args()
    if args.role:
        result=collect(args.role);(FOLDER/f'exact_{args.role}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(role=args.role,validated=True,conditions=len(result['conditions']))))
    else:report()
