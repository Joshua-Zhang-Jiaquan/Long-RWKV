"""Read raw complete panels, seal prospective choices, then analyze held-out outcomes."""
import argparse,json,math
from pathlib import Path
import numpy as np
from lrwkv_evidence.predictive_transfer import prediction as P,training as R
from lrwkv_evidence.train04.worker import file_sha,canonical_sha
ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'results/predictive_transfer'


def read_panel(seed,split):
    plan=json.loads((FOLDER/f'plan_{split}_seed{seed}.json').read_text())
    rows=[];hashes={}
    expected=P.panel(split)
    for rank in range(8):
        path=Path(plan['out'])/f'rank{rank}.jsonl'
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if records[0]['kind']!='provenance' or records[-1]['kind']!='complete':raise ValueError('incomplete raw rank')
        prov=records[0]
        if prov['rank']!=rank or prov['seed']!=seed or prov['split']!=split or prov['checkpoint_sha256']!=plan['checkpoint_sha256']:
            raise ValueError('wrong raw selector')
        conditions=[r for r in records if r['kind']=='condition']
        if [r['cell'] for r in conditions]!=expected[rank::8] or records[-1]['conditions']!=len(conditions):raise ValueError('missing/duplicate cells')
        for row in conditions:
            metrics=P.metrics(P.example(row['cell']),row['initial'],row['conditional'])
            for p in (0,1):
                if abs(metrics[p]['kl']-row['metrics'][p]['kl'])>1e-10:raise ValueError('raw arithmetic mismatch')
            row['metrics']=metrics
        rows.extend(conditions);hashes[str(path)]=file_sha(path)
    order={(c['index'],c['position']):i for i,c in enumerate(expected)}
    return sorted(rows,key=lambda r:order[(r['cell']['index'],r['cell']['position'])]),hashes,plan


def seal():
    dest=FOLDER/'PREDICTION_SEAL.json'
    if dest.exists():raise ValueError('already sealed')
    if list(FOLDER.glob('plan_heldout_seed*.json')):raise ValueError('target evaluation planned before prediction seal')
    protocol=ROOT/'revision/transfer/FROZEN.json';predictions={}
    for seed in R.SEEDS:
        rows,hashes,plan=read_panel(seed,'calibration')
        fitted=P.fit(rows)
        value=dict(seed=seed,protocol_sha256=file_sha(protocol),checkpoint_sha256=plan['checkpoint_sha256'],
                   calibration_raw_sha256=hashes,fitted=fitted,predictions=[P.predict(fitted,c) for c in P.panel('heldout')])
        path=FOLDER/f'predictions_seed{seed}.json'
        if path.exists():raise ValueError('prediction file already exists')
        path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
        predictions[str(seed)]=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))
    value=dict(protocol_sha256=file_sha(protocol),predictions=predictions,
               commitment='all six predictions recorded before any heldout job plan or model evaluation')
    dest.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(seal_sha256=file_sha(dest),models=len(predictions))))


def report():
    sealpath=FOLDER/'PREDICTION_SEAL.json';commit=json.loads(sealpath.read_text())
    protocol=json.loads((ROOT/'revision/transfer/FROZEN.json').read_text())
    shape=(6,10,8,3);primary=np.zeros(shape);selected=np.zeros(shape);regret=np.zeros(shape)
    supported=np.zeros(shape,dtype=bool);baselines={k:np.zeros(shape) for k in ('random','fixed_A','fixed_B','unconditional','fanin_heuristic')}
    squared=[];all_raw={};per_cell=[]
    for si,seed in enumerate(R.SEEDS):
        artifact=commit['predictions'][str(seed)];path=ROOT/artifact['path']
        if file_sha(path)!=artifact['sha256']:raise ValueError('prediction seal broken')
        pred=json.loads(path.read_text());rows,hashes,_=read_panel(seed,'heldout');all_raw.update(hashes)
        for row,prediction in zip(rows,pred['predictions']):
            cell=row['cell']
            if prediction['cell']!=cell:raise ValueError('prediction alignment')
            c=cell['class_id']-15;j=cell['index']%100;p=P.POSITIONS.index(cell['position']);idx=(si,c,j,p)
            kl=[m['kl'] for m in row['metrics']];choice=prediction['chosen'];value=kl[choice]
            selected[idx]=value;primary[idx]=kl[prediction['dependence_only']]-value;regret[idx]=value-min(kl)
            supported[idx]=prediction['supported']
            for key in ('unconditional','fanin_heuristic'):baselines[key][idx]=kl[prediction[key]]
            baselines['random'][idx]=sum(kl)/2;baselines['fixed_A'][idx]=kl[0];baselines['fixed_B'][idx]=kl[1]
            squared.extend((a-b)**2 for a,b in zip(prediction['scores'],kl))
            per_cell.append(dict(seed=seed,cell=cell,kl=kl,prediction=prediction,selected_kl=value,benefit=float(primary[idx]),regret=float(regret[idx])))
    random=np.random.default_rng(protocol['estimands']['bootstrap_seed']);boot=[]
    for _ in range(10000):
        lineages=random.integers(0,6,6);classes=random.integers(0,10,10);values=[]
        for c in classes:
            prompts=random.integers(0,8,8)
            values.append(primary[lineages][:,c][:,prompts].mean())
        boot.append(float(np.mean(values)))
    interval=np.quantile(boot,[.025,.975]).tolist();lineage_kl=selected.mean(axis=(1,2,3))
    gates=dict(effect_size=float(primary.mean())>=.1,positive_interval=interval[0]>0,
               support=float(supported.mean())>=.8,all_six_complete=True,
               competent_lineages=int((lineage_kl<4*math.log(2)).sum())>=5)
    result=dict(status='complete held-out analysis; all fixed selectors retained',prediction_seal_sha256=file_sha(sealpath),raw_sha256=all_raw,
                primary_mean_nats=float(primary.mean()),primary_interval=interval,predicted_policy_kl=float(selected.mean()),
                mean_selection_regret=float(regret.mean()),supported_fraction=float(supported.mean()),score_rmse=math.sqrt(sum(squared)/len(squared)),
                baseline_kl={k:float(v.mean()) for k,v in baselines.items()},
                improvement_over_baselines={k:float((v-selected).mean()) for k,v in baselines.items()},
                lineages=[dict(seed=seed,mean_selected_kl=float(lineage_kl[i]),primary=float(primary[i].mean()),competent=bool(lineage_kl[i]<4*math.log(2))) for i,seed in enumerate(R.SEEDS)],
                classes=[dict(class_id=15+c,primary=float(primary[:,c].mean()),selected_kl=float(selected[:,c].mean())) for c in range(10)],
                gates=gates,all_primary_gates_pass=all(gates.values()),cells=per_cell)
    (FOLDER/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','raw_sha256')},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('seal','report'));args=ap.parse_args()
    (seal if args.action=='seal' else report)()
