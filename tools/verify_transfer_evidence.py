"""Independent CPU audit of prospective raw outputs and selection arithmetic.

Parses the public equations, enumerates all256 generated endpoints, and rebuilds
feature bins without importing the experiment's metric/fit/predict functions.
"""
import argparse,gzip,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.predictive_transfer import tasks as generator


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def close(a,b,label,tolerance=1e-9):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance,label+f': {a} != {b}')


def parse_public(prompt):
    names=prompt.split('\nOutput order: ')[1].split('\n')[0].split(', ')
    require(len(names)==len(set(names))==8,'public output coordinates')
    equations=prompt.split('\nConstraints:\n')[1].split('\nBasis A: ')[0].splitlines()
    rows=[];syndrome=[]
    for equation in equations:
        left,right=equation.split(' = ');terms=left.split(' XOR ')
        require(len(set(terms))==len(terms) and right in ('0','1'),'public equation syntax')
        rows.append(sum(1<<names.index(term) for term in terms));syndrome.append(int(right))
    require(len(rows)==4,'four public constraints')
    bases=[[names.index(name) for name in prompt.split('\nBasis '+label+': ')[1].split('\n')[0].split(', ')] for label in ('A','B')]
    require(all(len(b)==len(set(b))==4 for b in bases) and set(bases[0]).isdisjoint(bases[1]),'public bases')
    ys=[y for y in range(256) if all((y&r).bit_count()%2==s for r,s in zip(rows,syndrome))]
    require(len(ys)==16,'rank-four posterior')
    for basis in bases:require(len({tuple((y>>i)&1 for i in basis) for y in ys})==16,'basis must index all histories')
    enumerator=[0]*9
    for y in ys:enumerator[(y^ys[0]).bit_count()]+=1
    return rows,bases,ys,enumerator


def features(rows,bases,policy,position):
    first=bases[policy];second=sorted(set(range(8))-set(first));out=[]
    for i in first:out.append((i,('fair',0,sum((r>>i)&1 for r in rows),position)))
    for i in second:
        matches=[]
        for mask in range(1,16):
            relation=0
            for j,row in enumerate(rows):
                if mask>>j&1:relation^=row
            known=sum(1<<j for j in first)
            if relation&(1<<i) and relation&~(known|(1<<i))==0:
                matches.append((relation.bit_count()-1,mask.bit_count()))
        require(len(matches)==1,'unique public parity expression')
        out.append((i,('deterministic',*matches[0],position)))
    return out


def endpoint(prompt,initial,conditional):
    rows,bases,ys,enumerator=parse_public(prompt)
    initial=np.asarray(initial,dtype=np.float64);conditional=np.asarray(conditional,dtype=np.float64)
    require(initial.shape==(8,2) and conditional.shape==(2,16,8,2),'raw probability shapes')
    require(np.isfinite(initial).all() and np.isfinite(conditional).all(),'finite raw log probabilities')
    require(float(np.abs(np.exp(initial).sum(-1)-1).max())<1e-10,'initial marginal normalization')
    require(float(np.abs(np.exp(conditional).sum(-1)-1).max())<1e-10,'conditional marginal normalization')
    outcomes=[]
    for policy,first in enumerate(bases):
        remaining=sorted(set(range(8))-set(first))
        lookup={tuple((y>>i)&1 for i in first):h for h,y in enumerate(ys)}
        loglaw=[]
        for y in range(256):
            h=lookup[tuple((y>>i)&1 for i in first)]
            loglaw.append(sum(initial[i,(y>>i)&1] for i in first)+sum(conditional[policy,h,i,(y>>i)&1] for i in remaining))
        require(abs(math.fsum(math.exp(x) for x in loglaw)-1)<1e-9,'full256endpoint normalization')
        kl=math.fsum(-math.log(16)-loglaw[y] for y in ys)/16
        mass=math.fsum(math.exp(loglaw[y]) for y in ys)
        errors={str(i):(-math.log(2)-float(initial[i].mean())) for i in first}
        for i in remaining:errors[str(i)]=math.fsum(-conditional[policy,h,i,(y>>i)&1] for h,y in enumerate(ys))/16
        close(kl,math.fsum(errors.values()),'independent KL/error decomposition')
        outcomes.append(dict(kl=kl,valid_mass=mass,errors=errors))
    return outcomes,(rows,bases,ys,enumerator)


def load_panel(folder,seed,split,protocol,index=None):
    plan=json.loads((folder/f'plan_{split}_seed{seed}.json').read_text());cells=protocol[split+'_panel'];observed=[];hashes={}
    for rank in range(8):
        original=str(Path(plan['out'])/f'rank{rank}.jsonl')
        if index is None:
            path=Path(original);raw=path.read_bytes()
        else:
            record=index[original];path=ROOT/record['path'];require(sha(path)==record['gzip_sha256'],'compressed raw checksum')
            raw=gzip.decompress(path.read_bytes());require(hashlib.sha256(raw).hexdigest()==record['sha256'],'decompressed raw checksum')
        hashes[original]=hashlib.sha256(raw).hexdigest();records=[json.loads(line) for line in raw.splitlines()]
        require(records[0]['kind']=='provenance' and records[-1]['kind']=='complete','complete rank output')
        provenance=records[0]
        require(provenance['rank']==rank and provenance['seed']==seed and provenance['split']==split,'rank selector')
        require(provenance['checkpoint_sha256']==plan['checkpoint_sha256'],'checkpoint selector')
        require(provenance['protocol_sha256']==sha(ROOT/'revision/transfer/FROZEN.json'),'protocol selector')
        screens=[r for r in records if r['kind']=='numerical_screen']
        require(len(screens)==6 and all(r['within']<=1e-5 and r['cross']<=1e-4 for r in screens),'numerical screens')
        conditions=[r for r in records if r['kind']=='condition']
        require([r['cell'] for r in conditions]==cells[rank::8] and records[-1]['conditions']==len(conditions),'complete fixed panel')
        for row in conditions:
            cell=row['cell'];ex=generator.make_example(split,cell['seed'],cell['index'],cell['class_id'])
            require(row['instance_id']==ex['instance_id'],'public instance identity')
            computed,public=endpoint(ex['prompt'],row['initial'],row['conditional'])
            require(public[3]==protocol['classes'][str(cell['class_id'])]['weight_enumerator'],'public code class')
            for p in (0,1):
                for metric in ('kl','valid_mass'):close(computed[p][metric],row['metrics'][p][metric],metric)
                for i,e in computed[p]['errors'].items():close(e,row['metrics'][p]['errors'][i],'coordinate error')
            observed.append(dict(cell=cell,prompt=ex['prompt'],instance_id=ex['instance_id'],public=public,metrics=computed,stored_metrics=row['metrics']))
    order={(c['index'],c['position']):i for i,c in enumerate(cells)}
    return sorted(observed,key=lambda r:order[(r['cell']['index'],r['cell']['position'])]),hashes


def fit_source(records):
    values=defaultdict(list);instances=defaultdict(set);parent=defaultdict(list);totals=[[],[]]
    for rec in records:
        rows,bases,_,_=rec['public'];position=rec['cell']['position']
        for p in (0,1):
            stored=rec.get('stored_metrics',rec['metrics'])
            totals[p].append(stored[p]['kl'])
            for i,feature in features(rows,bases,p,position):
                error=stored[p]['errors'][str(i)]
                values[feature].append(error);instances[feature].add(rec['instance_id']);parent[(feature[0],position)].append(error)
    means={k:sum(v)/len(v) for k,v in parent.items()}
    bins={k:dict(mean=(sum(v)+8*means[(k[0],k[3])])/(len(v)+8),prompts=len(instances[k]),outputs=len(v)) for k,v in values.items()}
    return bins,means,int(sum(totals[1])<sum(totals[0]))


def choice_from_source(fitted,rec):
    bins,pooled,unconditional=fitted;rows,bases,_,_=rec['public'];position=rec['cell']['position']
    tie=int(hashlib.sha256(str(('dependence_tie_v1',rec['prompt'],position)).encode()).hexdigest(),16)%2
    scores=[];supported=True;fan=[];missing=[]
    for p in (0,1):
        score=0.;fan.append(0)
        for _,feature in features(rows,bases,p,position):
            entry=bins.get(feature)
            if entry is None or entry['prompts']<8:supported=False;missing.append('|'.join(map(str,feature)))
            score+=entry['mean'] if entry else pooled[(feature[0],position)]
            if feature[0]=='deterministic':fan[-1]+=feature[1]
        scores.append(score)
    chosen=int(scores[1]<scores[0]) if supported and scores[0]!=scores[1] else tie
    return dict(scores=scores,supported=supported,chosen=chosen,dependence_only=tie,unconditional=unconditional,
                unsupported_bins=sorted(set(missing)),fanin_heuristic=int(fan[1]<fan[0]) if fan[0]!=fan[1] else tie)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,default=ROOT/'results/predictive_transfer')
    ap.add_argument('--raw-index',type=Path);ap.add_argument('--out',type=Path);args=ap.parse_args()
    protocol=json.loads((ROOT/'revision/transfer/FROZEN.json').read_text());folder=args.folder
    report=json.loads((folder/'report.json').read_text());sealpath=folder/'PREDICTION_SEAL.json';seal=json.loads(sealpath.read_text())
    require(report['prediction_seal_sha256']==sha(sealpath),'report prediction seal')
    require(seal['protocol_sha256']==sha(ROOT/'revision/transfer/FROZEN.json'),'sealed protocol')
    index=None if args.raw_index is None else {r['original_path']:r for r in json.loads(args.raw_index.read_text())['files']}
    primary=np.zeros((6,10,8,3));selected=np.zeros_like(primary);regret=np.zeros_like(primary);supported=np.zeros_like(primary)
    baseline={k:np.zeros_like(primary) for k in ('random','fixed_A','fixed_B','unconditional','fanin_heuristic')}
    all_hashes={};squared=[];positions=('far','middle','near');full_laws=0
    for si,seed in enumerate(protocol['training_seeds']):
        source,source_hashes=load_panel(folder,seed,'calibration',protocol,index)
        target,target_hashes=load_panel(folder,seed,'heldout',protocol,index);all_hashes.update(source_hashes);all_hashes.update(target_hashes)
        predrec=seal['predictions'][str(seed)];predpath=ROOT/predrec['path'];require(sha(predpath)==predrec['sha256'],'prediction checksum')
        pred=json.loads(predpath.read_text());require(pred['calibration_raw_sha256']==source_hashes,'source-only prediction inputs')
        require(pred['seed']==seed and pred['protocol_sha256']==sha(ROOT/'revision/transfer/FROZEN.json'),'prediction identity')
        plans=[json.loads((folder/f'plan_{split}_seed{seed}.json').read_text()) for split in ('calibration','heldout')]
        require(all(plan['checkpoint_sha256']==pred['checkpoint_sha256'] for plan in plans),'same calibrated/evaluated checkpoint')
        require(all(report['raw_sha256'][p]==h for p,h in target_hashes.items()),'reported target raw hashes')
        require(len(pred['predictions'])==len(target),'all predictions present')
        fitted=fit_source(source)
        for record,stored in zip(target,pred['predictions']):
            require(record['cell']==stored['cell'],'prediction pairing')
            expected=choice_from_source(fitted,record)
            for k in ('supported','chosen','dependence_only','unconditional','fanin_heuristic','unsupported_bins'):require(expected[k]==stored[k],'independent choice mismatch '+k)
            for a,b in zip(expected['scores'],stored['scores']):close(a,b,'independent score')
            cell=record['cell'];idx=(si,cell['class_id']-15,cell['index']%100,positions.index(cell['position']))
            kl=[m['kl'] for m in record['metrics']];chosen=kl[stored['chosen']]
            selected[idx]=chosen;primary[idx]=kl[stored['dependence_only']]-chosen;regret[idx]=chosen-min(kl);supported[idx]=stored['supported']
            baseline['random'][idx]=sum(kl)/2;baseline['fixed_A'][idx]=kl[0];baseline['fixed_B'][idx]=kl[1]
            for k in ('unconditional','fanin_heuristic'):baseline[k][idx]=kl[stored[k]]
            squared.extend((a-b)**2 for a,b in zip(expected['scores'],kl))
        full_laws+=2*(len(source)+len(target))
    for key,value in dict(primary_mean_nats=primary.mean(),predicted_policy_kl=selected.mean(),mean_selection_regret=regret.mean(),
                          supported_fraction=supported.mean(),score_rmse=math.sqrt(math.fsum(squared)/len(squared))).items():close(float(value),report[key],key)
    for k,v in baseline.items():
        close(float(v.mean()),report['baseline_kl'][k],k)
        close(float((v-selected).mean()),report['improvement_over_baselines'][k],k+' gain')
    require([r['seed'] for r in report['lineages']]==protocol['training_seeds'],'reported lineages')
    for i,row in enumerate(report['lineages']):
        close(float(selected[i].mean()),row['mean_selected_kl'],'lineage KL')
        close(float(primary[i].mean()),row['primary'],'lineage benefit')
        require(row['competent']==bool(selected[i].mean()<4*math.log(2)),'lineage competence')
    require([r['class_id'] for r in report['classes']]==list(range(15,25)),'reported classes')
    for c,row in enumerate(report['classes']):
        close(float(selected[:,c].mean()),row['selected_kl'],'class KL')
        close(float(primary[:,c].mean()),row['primary'],'class benefit')
    require(len(report['cells'])==1440,'reported cell census')
    seen=set()
    for row in report['cells']:
        cell=row['cell'];idx=(protocol['training_seeds'].index(row['seed']),cell['class_id']-15,cell['index']%100,positions.index(cell['position']))
        require(idx not in seen,'duplicate reported cell');seen.add(idx)
        close(row['selected_kl'],float(selected[idx]),'reported selected cell')
        close(row['benefit'],float(primary[idx]),'reported cell benefit')
        close(row['regret'],float(regret[idx]),'reported cell regret')
    random=np.random.default_rng(protocol['estimands']['bootstrap_seed']);boot=[]
    for _ in range(10000):
        seeds=random.integers(0,6,6);classes=random.integers(0,10,10);total=0.
        for c in classes:
            prompts=random.integers(0,8,8)
            total+=float(primary[np.ix_(seeds,[c],prompts,range(3))].mean())
        boot.append(total/10)
    interval=np.quantile(boot,[.025,.975]).tolist()
    for a,b in zip(interval,report['primary_interval']):close(a,b,'hierarchical interval')
    competent=int((selected.mean((1,2,3))<4*math.log(2)).sum())
    gates=dict(effect_size=float(primary.mean())>=.1,positive_interval=interval[0]>0,support=float(supported.mean())>=.8,all_six_complete=True,competent_lineages=competent>=5)
    require(gates==report['gates'] and all(gates.values())==report['all_primary_gates_pass'],'success gates')
    result=dict(complete=True,raw_rank_files=len(all_hashes),full256endpoint_laws=full_laws,primary_mean_nats=float(primary.mean()),primary_interval=interval,
                gates=gates,scope='Independent public-equation parsing, complete joint laws, calibration/selection reconstruction and paired hierarchical inference; no new neural inference.')
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
