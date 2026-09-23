"""Independent endpoint arithmetic, fresh-input geometry, provenance and delivery audit."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.header_distance import tasks as T
from lrwkv_evidence.train04.worker import load_tokenizer


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    folder=ROOT/'results/header_distance';manifestpath=ROOT/'theory_mvp/header_distance/FROZEN.json'
    manifest=json.loads(manifestpath.read_text());report=json.loads((folder/'report.json').read_text())
    if report['manifest_sha256']!=sha(manifestpath):raise ValueError('report manifest mismatch')
    for rel,digest in manifest['sources'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('frozen source changed: '+rel)
    archive=json.loads((ROOT/'results/paper_audit/completed_history_response_archive.json').read_text())
    if sha(ROOT/archive['archive'])!=archive['sha256']:raise ValueError('preceding delivery archive changed')
    inputs=json.loads((ROOT/'theory_mvp/header_distance/INPUTS.json').read_text())
    tok,_,_=load_tokenizer(Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B'))
    for row in inputs['conditions']:
        encoded=T.serialize(T.example(row['index']),tok,row['cell'])
        if row['serial']!={k:v for k,v in encoded.items() if k!='ids'}:raise ValueError('frozen inputs not reproduced')
        if not row['cell'].startswith('legacy_'):
            expected_header=[0,24] if encoded['header_position']=='far' else [16268,16292]
            expected_task=[24,112] if encoded['task_position']=='far' else [16292,16380]
            if encoded['header_span']!=expected_header or encoded['task_span']!=expected_task:raise ValueError('factor slots not fixed')
    metrics={};all_metrics={};count=0;max_error=0.
    for role in manifest['roles']:
        validated=json.loads((folder/f'exact_{role}.json').read_text())
        if not validated['validated'] or validated['manifest_sha256']!=sha(manifestpath):raise ValueError('bad collection')
        if validated['checkpoint_sha256']!=manifest['checkpoints'][role]['sha256']:raise ValueError('wrong checkpoint')
        saved={(r['index'],r['cell']):r for r in validated['conditions']};values={};all_metrics[role]={}
        for path,digest in validated['raw_file_sha256'].items():
            if sha(path)!=digest:raise ValueError('raw data changed')
            for row in map(json.loads,Path(path).read_text().splitlines()):
                if row['kind']!='condition':continue
                ex=T.example(row['index']);first=ex['information_set'];hidden=[i for i in range(8) if i not in first]
                for lp in [row['initial'],*row['conditional']]:
                    if len(lp)!=8:raise ValueError('bad output dimension')
                    for pair in lp:
                        if len(pair)!=2 or not all(math.isfinite(v) for v in pair) or abs(sum(math.exp(v) for v in pair)-1)>1e-6:raise ValueError('unnormalized probabilities')
                one=[];info=[];second=[]
                for y in ex['support']:
                    h=sum(((y>>i)&1)<<(3-j) for j,i in enumerate(first))
                    logfirst=sum(row['initial'][i][(y>>i)&1] for i in first)
                    logsecond=sum(row['conditional'][h][i][(y>>i)&1] for i in hidden)
                    one.append(sum(row['initial'][i][(y>>i)&1] for i in range(8)))
                    info.append(logfirst+logsecond);second.append(-logsecond)
                e1=sum(-math.log(2)-sum(row['initial'][i])/2 for i in first)
                actual=dict(one_kl=-math.log(16)-float(np.mean(one)),info_kl=-math.log(16)-float(np.mean(info)),
                            first_error=e1,second_error=float(np.mean(second)),
                            valid_mass_one=sum(math.exp(v) for v in one),valid_mass_info=sum(math.exp(v) for v in info))
                actual['gain_over_one']=actual['one_kl']-actual['info_kl']
                all_metrics[role][row['index'],row['cell']]=actual
                old=saved[row['index'],row['cell']]
                discrepancies=[abs(v-old[k]) for k,v in actual.items()]
                discrepancies += [max(abs(a-b) for a,b in zip(one,old['support_logq_one'])),max(abs(a-b) for a,b in zip(info,old['support_logq_info']))]
                if max(discrepancies)>1e-9:raise ValueError('independent endpoint disagreement')
                if abs(actual['info_kl']-e1-actual['second_error'])>1e-9:raise ValueError('chain decomposition failed')
                max_error=max(max_error,*discrepancies);values[row['index'],row['cell']]=actual['info_kl'];count+=1
        if len(values)!=192:raise ValueError('missing conditions')
        metrics[role]=values
    if count!=1152:raise ValueError('wrong count')
    samples=np.random.default_rng(2026092302).integers(0,32,size=(10000,32))
    summaries_checked=0
    def check_summary(values,saved):
        nonlocal summaries_checked
        values=np.asarray(values)
        ci=np.quantile(values[samples].mean(axis=1),[.025,.975])
        if abs(values.mean()-saved['mean'])>1e-9 or max(abs(ci-np.array(saved['ci95'])))>1e-9:
            raise ValueError('independent summary mismatch')
        summaries_checked+=1
    def vector(role,cell,key='info_kl'):
        return np.array([all_metrics[role][i,cell][key] for i in range(32)])
    within={}
    for role,cells in report['checkpoints'].items():
        for cell,summary in cells.items():
            for key,saved in summary.items():check_summary(vector(role,cell,key),saved)
        ff,fn,nf,nn=[vector(role,c) for c in T.CELLS[:4]]
        within[role]=dict(task_penalty_header_far=ff-fn,task_penalty_header_near=nf-nn,
             header_penalty_task_far=ff-nf,header_penalty_task_near=fn-nn,
             factorial_interaction=ff-fn-nf+nn,colocated_far_minus_legacy=ff-vector(role,'legacy_far'),
             colocated_near_minus_legacy=nn-vector(role,'legacy_near'))
        for key,v in within[role].items():check_summary(v,report['within_checkpoint'][role][key])
    pooled={}
    for seed in (20271011,20271012,20271013):
        near=f'near_seed{seed}';balanced=f'balanced_seed{seed}'
        effects={c+'_error_reduction':vector(near,c)-vector(balanced,c) for c in T.CELLS}
        effects.update({k+'_reduction':v-within[balanced][k] for k,v in within[near].items()})
        effects['primary_second_error_reduction']=vector(near,'header_near_task_far','second_error')-vector(balanced,'header_near_task_far','second_error')
        for key,v in effects.items():
            check_summary(v,report['per_seed'][str(seed)][key]);pooled.setdefault(key,[]).append(v)
    for key,v in pooled.items():check_summary(np.mean(v,axis=0),report['seed_averaged'][key])
    seeds=(20271011,20271012,20271013);cell='header_near_task_far'
    effects=np.array([[metrics[f'near_seed{s}'][i,cell]-metrics[f'balanced_seed{s}'][i,cell] for i in range(32)] for s in seeds])
    values=effects.mean(axis=0);samples=np.random.default_rng(2026092302).integers(0,32,size=(10000,32))
    ci=np.quantile(values[samples].mean(axis=1),[.025,.975])
    if abs(values.mean()-report['primary']['mean'])>1e-9 or max(abs(ci-np.array(report['primary']['ci95'])))>1e-9:raise ValueError('primary inference mismatch')
    passes=all(np.mean([metrics[f'balanced_seed{s}'][i,c] for i in range(32)])<4*math.log(2) for s in seeds for c in (cell,'header_near_task_near'))
    strong=bool(ci[0]>0 and all(e.mean()>0 for e in effects) and passes)
    if strong!=report['stronger_interpretation_supported']:raise ValueError('decision rule mismatch')
    recovery=json.loads((folder/'replacement_balanced_seed20271013.json').read_text())
    for file in recovery['old_files']:
        if sha(file['path'])!=file['sha256']:raise ValueError('stopped-attempt artifact changed')
    oldplan=json.loads((folder/'queued_attempts/plan_balanced_seed20271013_extra.json').read_text())
    newplan=json.loads((folder/'plan_balanced_seed20271013.json').read_text())
    if list(Path(oldplan['out']).glob('rank*.jsonl')):raise ValueError('stopped attempt contains predictions')
    for key in ('stage','sources','checkpoint','checkpoint_sha256','manifest_sha256'):
        if oldplan[key]!=newplan[key]:raise ValueError('operational replacement changed science')
    if oldplan['body']['command'].replace(oldplan['out'],newplan['out'])!=newplan['body']['command']:
        raise ValueError('replacement command changed beyond output path')
    resource=json.loads((folder/'resource_audit.json').read_text())
    if not resource['all_jobs_terminal'] or resource['live_campaign']['live_reserved_gpus']!=0:raise ValueError('live resources remain')
    pdf=ROOT/'build/Long_RWKV_ICLR2027.pdf';review=json.loads((ROOT/'results/paper_audit/header_distance_visual_review.json').read_text())
    if review['pdf_sha256']!=sha(pdf) or not review['all_pages_reviewed']:raise ValueError('stale PDF review')
    log=(ROOT/'build/focused-pass-3.log').read_text()
    if any(s in log for s in ('undefined','multiply defined','Overfull')):raise ValueError('unclean PDF')
    result=dict(complete=True,independently_checked_summaries=summaries_checked,conditions=count,endpoint_laws=2*count,independent_max_arithmetic_discrepancy=max_error,
                primary_mean=float(values.mean()),primary_ci95=ci.tolist(),stronger_interpretation_supported=strong,
                manifest_sha256=sha(manifestpath),pdf_sha256=sha(pdf),archive_sha256=archive['sha256'],all_jobs_terminal=True,live_reserved_gpus=0,
                scope='Frozen inputs and exact slot geometry, source/raw hashes, independent endpoint laws and bootstrap inference, predeclared interpretation rule, resources and reviewed PDF.')
    (folder/'delivery_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
