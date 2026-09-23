"""Revalidate a complete raw development probe and summarize paired endpoints."""
import argparse,json,math,statistics
from pathlib import Path
from lrwkv_evidence.posterior_context_final import tasks as T
from lrwkv_evidence.posterior_context_probe.exact import audit,logsumexp
from lrwkv_evidence.train04.worker import file_sha,load_tokenizer
from lrwkv_evidence.long_context_eval.numerical import validate as validate_numerical

def collect(planpath):
    plan=json.loads(planpath.read_text());stage=Path(plan['stage']);out=Path(plan['out'])
    for rel,digest in plan['sources'].items():
        if file_sha(stage/rel)!=digest:raise ValueError('changed staged source')
    training=json.loads(Path(plan['training_plan']).read_text());checkpoint=Path(training['out'])/'resume.pt'
    if file_sha(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('changed checkpoint')
    tok,_,_=load_tokenizer(Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B'))
    design=json.loads((stage/'final_design.json').read_text())
    if file_sha(stage/'final_design.json')!=plan['design_sha256']:raise ValueError('design mismatch')
    conditions=[];rawhash={}
    for rank in range(8):
        path=out/f'rank{rank}.jsonl';rawhash[str(path)]=file_sha(path)
        rows=[json.loads(line) for line in path.read_text().splitlines()];index=plan['shard']*8+rank;ex=T.example(index)
        if not rows or rows[0]['kind']!='provenance' or rows[-1]['kind']!='complete':raise ValueError('incomplete rank')
        p=rows[0]
        if p['rank']!=rank or p['world']!=8 or p['seed']!=T.SEED or p['split']!='test' or p['checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('wrong provenance')
        if p['shard']!=plan['shard'] or p['role']!=plan['role'] or p['design_sha256']!=plan['design_sha256']:raise ValueError('wrong shard identity')
        frozen=design['panel'][index]
        if ex['instance_id']!=frozen['instance_id'] or ex['support']!=frozen['support']:raise ValueError('wrong frozen condition')
        expected=[(None,'evidence_only')]+[(n,pos) for n in (1024,4096,16384) for pos in ('far','middle','near')]
        observed=[r for r in rows if r['kind']=='condition']
        if [(r['serial']['requested_context_tokens'],r['serial']['position']) for r in observed]!=expected:raise ValueError('incomplete conditions')
        screens=[r for r in rows if r['kind']=='numerical_screen']
        if len(screens)!=1:raise ValueError('bad numerical screen count')
        if screens:validate_numerical(screens[0]['rows'])
        if len(rows)!=len(observed)+2+len(screens) or rows[-1]['long_sweep_complete']!=True:raise ValueError('unexpected raw rows')
        # A uniform model has KL=4log2; the auditor computes policy-specific D independently.
        reference={r['method']:r for r in audit(ex,lambda _: [[-math.log(2)]*2 for i in range(8)])['results']}
        for row in observed:
            ser=row['serial'];fresh=T.serialize(ex,tok,ser['requested_context_tokens'],ser['position'])
            if ser!={k:v for k,v in fresh.items() if k!='ids'} or row['index']!=index or row['family']!=ex['family']:raise ValueError('wrong serialized condition')
            exact=row['exact']
            if exact['support']!=ex['support'] or exact['instance_id']!=ex['instance_id'] or exact['family']!=ex['family']:raise ValueError('wrong target law')
            if [r['method'] for r in exact['results']]!=list(T.partitions(ex)):raise ValueError('wrong policy list')
            for result in exact['results']:
                lp=result['support_log_probabilities']
                if len(lp)!=16 or not all(math.isfinite(x) and x<=0 for x in lp):raise ValueError('bad probabilities')
                kl=-math.log(16)-statistics.mean(lp);lv=logsumexp(lp);d=reference[result['method']]['dependence_nats']
                expected_values=dict(joint_kl_nats=kl,log_valid_mass=lv,exact_valid_mass=math.exp(lv),within_valid_kl_nats=kl+lv,dependence_nats=d,estimation_nats=kl-d)
                if lv>1e-8 or kl-d < -1e-8 or kl+lv < -1e-8:raise ValueError('invalid endpoint law')
                for key,value in expected_values.items():
                    if not math.isfinite(result[key]) or abs(result[key]-value)>1e-8:raise ValueError('inconsistent endpoint '+key)
            conditions.append(row)
    passed=True
    summary=[]
    for family in ('systematic','paired_parity'):
        for length,pos in [(None,'evidence_only')]+([(n,p) for n in (1024,4096,16384) for p in ('far','middle','near')] if passed else []):
            selected=[r for r in conditions if r['family']==family and r['serial']['requested_context_tokens']==length and r['serial']['position']==pos]
            if len(selected)!=4:raise ValueError('bad family condition count')
            for method in T.partitions(T.example(0)):
                endpoints=[next(e for e in r['exact']['results'] if e['method']==method) for r in selected]
                summary.append(dict(family=family,context_tokens=length,position=pos,method=method,conditions=4,mean_kl=statistics.mean(e['joint_kl_nats'] for e in endpoints),mean_valid=statistics.mean(e['exact_valid_mass'] for e in endpoints)))
    return dict(execution_complete=True,long_sweep_complete=passed,design_sha256=plan['design_sha256'],role=plan['role'],shard=plan['shard'],checkpoint_sha256=plan['checkpoint_sha256'],plan=str(planpath.resolve()),raw_sha256=rawhash,summary=summary,conditions=conditions,scope=design['selection_disclosure'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result=collect(args.plan);args.out.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Frozen fresh-condition shard','',result['scope'],'',f"Frozen shard complete: {result['long_sweep_complete']}",'','| Family | Context | Position | Policy | Mean KL | Mean valid mass |','|---|---:|---|---|---:|---:|']
    lines += [f"| {r['family']} | {r['context_tokens']} | {r['position']} | {r['method']} | {r['mean_kl']:.8g} | {r['mean_valid']:.8g} |" for r in result['summary']]
    args.out.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('conditions','raw_sha256')},indent=2))
if __name__=='__main__':main()
