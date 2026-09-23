"""Offline audit from preserved raw predictions. Requires only NumPy; never uses GPUs."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lrwkv_evidence.train04 import tasks as BASE
SEEDS=(20271011,20271012,20271013)
CELLS=('header_far_task_far','header_far_task_near','header_near_task_far','header_near_task_near','legacy_far','legacy_near')
POSITIONS=('far','middle','near')

def read(path):return json.loads((ROOT/path).read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def close(a,b):require(np.allclose(a,b,rtol=0,atol=1e-9),'numerical disagreement')
def example(index,seed):
    ex=BASE.make_example('test',seed,index,family='paired_parity')
    # Enumerate independently rather than using the archived elimination helper.
    ex['support']=[y for y in range(256) if all((y&r).bit_count()%2==v for r,v in zip(ex['matrix_rows'],ex['syndrome']))]
    require(len(ex['support'])==16,'invalid target support')
    return ex

def validate_lp(lp):
    a=np.asarray(lp)
    require(a.shape==(8,2) and np.isfinite(a).all() and (a<=1e-8).all(),'invalid log probabilities')
    require(np.max(abs(np.exp(a).sum(axis=1)-1))<1e-6,'unnormalized binary head')

def header_metrics(ex,initial,conditional):
    require(len(conditional)==16,'all histories required')
    for lp in [initial,*conditional]:validate_lp(lp)
    first=ex['information_set'];hidden=[i for i in range(8) if i not in first]
    one=[];info=[];second=[]
    for y in ex['support']:
        h=sum(((y>>i)&1)<<(3-j) for j,i in enumerate(first))
        a=sum(initial[i][(y>>i)&1] for i in first)
        b=sum(conditional[h][i][(y>>i)&1] for i in hidden)
        one.append(sum(initial[i][(y>>i)&1] for i in range(8)))
        info.append(a+b);second.append(-b)
    result=dict(one_kl=-math.log(16)-np.mean(one),info_kl=-math.log(16)-np.mean(info),
                first_error=sum(-math.log(2)-sum(initial[i])/2 for i in first),second_error=np.mean(second),
                valid_mass_one=sum(math.exp(v) for v in one),valid_mass_info=sum(math.exp(v) for v in info))
    result['gain_over_one']=result['one_kl']-result['info_kl']
    close(result['info_kl'],result['first_error']+result['second_error'])
    require(result['one_kl']>=4*math.log(2)-1e-8,'product floor violated')
    return result

def history_metrics(ex,row):
    require(len(row['original'])==16 and len(row['flipped'])==4 and all(len(v)==16 for v in row['flipped']),'incomplete interventions')
    for lp in [row['initial'],*row['original'],*(p for v in row['flipped'] for p in v)]:validate_lp(lp)
    first=ex['information_set'];hidden=[i for i in range(8) if i not in first]
    second=0.;flipped=0.;pairs=[];failures=0;unaffected=0
    for h,bits in enumerate(itertools.product((0,1),repeat=4)):
        feasible=[y for y in ex['support'] if all((y>>i)&1==b for i,b in zip(first,bits))]
        require(len(feasible)==1,'history does not identify target')
        gold=[(feasible[0]>>i)&1 for i in range(8)];a=row['original'][h]
        second-=sum(a[i][gold[i]] for i in hidden)/16
        for r,mask in enumerate(ex['matrix_rows']):
            changed=[i for i in hidden if mask>>i&1];require(len(changed)==1,'ambiguous flipped coordinate')
            i=changed[0];b=row['flipped'][r][h];target=1-gold[i]
            flipped-=b[i][target]/16
            ce=-(a[i][gold[i]]+b[i][target])/2
            delta=(1 if target else -1)*((b[i][1]-b[i][0])-(a[i][1]-a[i][0]))
            z=-delta/2;floor=max(z,0)+math.log1p(math.exp(-abs(z)))
            require(ce>=floor-1e-9,'response inequality violated')
            pairs.append(ce)
            for j in hidden:
                if j==i:continue
                unaffected+=1
                failures+=int(a[j][gold[j]]>a[j][1-gold[j]] and b[j][gold[j]]<=b[j][1-gold[j]])
    first_error=sum(-math.log(2)-sum(row['initial'][i])/2 for i in first)
    mean=float(np.mean(pairs));close(4*mean,(second+flipped)/2)
    return dict(first_error=first_error,second_error=second,information_set_error=first_error+second,
                mean_correct_conditional_ce=mean,flipped_target_error=flipped,
                collateral_errors=failures,unaffected_comparisons=unaffected)

def estimate(values,seed):
    values=np.asarray(values);require(values.shape==(32,) and np.isfinite(values).all(),'incomplete problem vector')
    samples=np.random.default_rng(seed).integers(0,32,size=(10000,32))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[samples].mean(axis=1),[.025,.975]).tolist())

def raw_records(index):
    seen=set();bygroup={}
    for item in index['files']:
        key=(item['study'],item['role'],Path(item['original_path']).name)
        require(key not in seen,'duplicate raw rank');seen.add(key)
        p=ROOT/item['path'];require(p.resolve().is_relative_to(ROOT),'path escape')
        blob=p.read_bytes();require(sha(blob)==item['gzip_sha256'],'compressed data changed')
        data=gzip.decompress(blob);require(sha(data)==item['sha256'],'raw data changed')
        rows=[json.loads(line) for line in data.splitlines()]
        require(rows[-1]['kind']=='complete','incomplete rank')
        provenance=[r for r in rows if r['kind']=='provenance'];require(len(provenance)==1,'invalid provenance')
        require(provenance[0]['rank']==int(Path(item['original_path']).stem[4:]),'wrong rank')
        for r in rows:
            if r['kind']=='numerical_screen':
                require(len(r['rows'])==12,'incomplete numerical screen')
                for v in r['rows']:
                    for k,tol in [('within_repeat_max_logprob_diff',1e-5),('cross_rank_max_logprob_diff',1e-4)]:
                        require(math.isfinite(v[k]) and 0<=v[k]<=tol,'numerical screen failed')
            if r['kind']=='numerical_factorial':
                require(len(r['rows'])==8,'incomplete factorial screen')
                for v in r['rows']:
                    require(0<=v['within']<=1e-5 and 0<=v['cross']<=1e-4,'factorial screen failed')
        bygroup.setdefault((item['study'],item['role']),[]).extend(rows)
    expected={(s,f'{a}_seed{seed}') for s in ('distance_intervention','history_response','header_distance') for seed in SEEDS for a in ('near','balanced')}
    expected|={('distance_intervention','original'),('history_response','original'),('distance_intervention_cost','attention')}
    require(set(bygroup)==expected and len(seen)==168,'incorrect model/rank census')
    for key,rows in bygroup.items():
        require(sorted(r['rank'] for r in rows if r['kind']=='provenance')==list(range(8)),'missing rank')
    return bygroup

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    # The original manifests remain unchanged; relocation is recorded separately.
    for rel in ('theory_mvp/header_distance/FROZEN.json','theory_mvp/history_response/FROZEN.json','theory_mvp/distance_intervention/FROZEN_EVALUATION.json'):
        for path,digest in read(rel)['sources'].items():require(sha((ROOT/path).read_bytes())==digest,'frozen source changed: '+path)
    groups=raw_records(read('evidence/raw/INDEX.json'));data={};counts={};collateral=0;unaffected=0
    for (study,role),rows in groups.items():
        saved=read(f'results/{study}/exact_{role}.json')
        expected_hashes=saved.get('raw_file_sha256',saved.get('raw_sha256'))
        exported={i['original_path']:i['sha256'] for i in read('evidence/raw/INDEX.json')['files'] if (i['study'],i['role'])==(study,role)}
        require(exported==expected_hashes,'raw export is not bound to original collector')
        conditions=[r for r in rows if r['kind']=='condition'];counts[study]=counts.get(study,0)+len(conditions)
        values={}
        if study=='header_distance':
            previous={(r['index'],r['cell']):r for r in saved['conditions']}
            for r in conditions:
                key=(r['index'],r['cell']);require(key not in values,'duplicate condition')
                v=header_metrics(example(r['index'],2026092301),r['initial'],r['conditional'])
                for k,x in v.items():close(x,previous[key][k]);close(x,r['metrics'][k])
                values[key]=v
            require(set(values)=={(i,c) for i in range(32) for c in CELLS},'missing layout')
        elif study=='history_response':
            previous={(r['index'],r['position']):r for r in saved['conditions']}
            for r in conditions:
                key=(r['index'],r['position']);require(key not in values,'duplicate condition')
                v=history_metrics(example(r['index'],20271021),r)
                for k in ('first_error','second_error','information_set_error','flipped_target_error'):close(v[k],previous[key][k])
                close(v['mean_correct_conditional_ce'],previous[key]['response']['mean_correct_conditional_ce'])
                if role.startswith('balanced'):
                    collateral+=v['collateral_errors'];unaffected+=v['unaffected_comparisons']
                    require(v['first_error']+8*v['mean_correct_conditional_ce']<4*math.log(2),'response certificate failed')
                values[key]=v
            require(set(values)=={(i,p) for i in range(32) for p in POSITIONS},'missing history condition')
        elif study=='distance_intervention':
            require(len(conditions)==144,'missing main-study condition')
            require(sorted(conditions,key=lambda r:(r['index'],str(r['serial'])))==sorted(saved['conditions'],key=lambda r:(r['index'],str(r['serial']))),'collected main predictions differ')
            for r in conditions:
                key=(r['index'],r['serial']['requested_context_tokens'],r['serial']['position']);require(key not in values,'duplicate condition')
                methods={}
                require(len(r['exact']['support'])==16,'wrong support')
                for m in r['exact']['results']:
                    logq=m['support_log_probabilities'];require(len(logq)==16 and all(math.isfinite(x) for x in logq),'invalid endpoint')
                    kl=-math.log(16)-sum(logq)/16;close(kl,m['joint_kl_nats']);close(sum(math.exp(x) for x in logq),m['exact_valid_mass'])
                    d=0 if m['method']=='information_set' else 4*math.log(2)
                    close(kl,d+m['estimation_nats']);methods[m['method']]=kl
                require(set(methods)=={'one','information_set','pair_preserving_halves'},'missing policy')
                close(methods['information_set'],r['stages']['first_round_estimation_nats']+r['stages']['second_round_estimation_nats'])
                values[key]=methods
            require({(i,p) for i,n,p in values if n==16384}=={(i,p) for i in range(32) for p in POSITIONS},'missing primary condition')
            timing=[r for r in rows if r['kind']=='timing'];require(timing==saved['timings'],'timing export changed')
        else:
            require(len(conditions)==24,'incomplete attention timing')
            for method in ('one','information_set','pair_preserving_halves'):
                samples=[v for r in conditions for m in r['costs'] if m['method']==method for v in m['seconds']]
                require(len(samples)==120,'incomplete timing repeats')
                close(np.mean(samples),next(m['mean_request_seconds'] for m in saved['summary'] if m['method']==method))
        data[study,role]=values
    require(counts==dict(distance_intervention=1008,history_response=672,header_distance=1152,distance_intervention_cost=24),'incorrect condition census')
    specifications=[('distance_intervention',20271022,(16384,'far'),'information_set',read('results/distance_intervention/final_report.json')['intervention']['seed_averaged_problem_effects']['far_conditional_error_reduction']),
                    ('history_response',20260923,('far',),'mean_correct_conditional_ce',read('results/history_response/report.json')['seed_averaged']['far_mean_correct_conditional_ce_reduction']),
                    ('header_distance',2026092302,('header_near_task_far',),'info_kl',read('results/header_distance/report.json')['primary'])]
    primary={}
    for study,seed,suffix,metric,saved in specifications:
        effects=np.array([[data[study,f'near_seed{s}'][(i,*suffix)][metric]-data[study,f'balanced_seed{s}'][(i,*suffix)][metric] for i in range(32)] for s in SEEDS])
        result=estimate(effects.mean(axis=0),seed);close(result['mean'],saved['mean']);close(result['ci95'],saved['ci95'])
        require(all(v.mean()>0 for v in effects),'seed direction mismatch');primary[study]=result
    require((collateral,unaffected)==(3,55296),'collateral-error count mismatch')
    for seed in SEEDS:
        for cell in ('header_near_task_far','header_near_task_near'):
            require(np.mean([data['header_distance',f'balanced_seed{seed}'][i,cell]['info_kl'] for i in range(32)])<4*math.log(2),'header quality criterion failed')
    # Recompute all six budget certificates using the supplied exact timing records.
    sys.path.insert(0,str(ROOT/'theory_mvp/distance_intervention'))
    from cost_certificate import certificate
    attention=read('results/distance_intervention_cost/exact_attention.json')
    for old in read('results/distance_intervention_cost/budget_certificates.json')['certificates']:
        new=certificate(attention,read(f"results/distance_intervention/exact_{old['role']}.json"))
        require(new==old,'budget certificate mismatch')
    result=dict(complete=True,raw_rank_files=168,conditions=counts,primary=primary,collateral_errors=collateral,unaffected_comparisons=unaffected,
                scope='Offline verification of raw hashes, complete selectors, endpoint/response arithmetic, three primary paired intervals, response inequalities, and all six empirical cost certificates. No model inference, new timing, checkpoint availability, or official-kernel parity is claimed.')
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
