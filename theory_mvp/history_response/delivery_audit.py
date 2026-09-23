"""Independent arithmetic and provenance checks on the exhaustive response audit."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from lrwkv_evidence.distance_intervention.tasks import example


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    folder=ROOT/'results/history_response';manifestpath=ROOT/'theory_mvp/history_response/FROZEN.json'
    manifest=json.loads(manifestpath.read_text());report=json.loads((folder/'report.json').read_text())
    if report['manifest_sha256']!=sha(manifestpath):raise ValueError('report manifest mismatch')
    for rel,digest in manifest['sources'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('frozen source changed: '+rel)
    archive=json.loads((ROOT/'results/paper_audit/completed_distance_study_archive.json').read_text())
    if sha(ROOT/archive['archive'])!=archive['sha256']:raise ValueError('completed study archive changed')
    cells=0;pairs=0;max_error=0.;metrics={};collateral_counts={}
    for role in manifest['roles']:
        plan=json.loads((folder/f'plan_{role}.json').read_text())
        validated=json.loads((folder/f'exact_{role}.json').read_text())
        if not validated['validated'] or validated['manifest_sha256']!=sha(manifestpath):raise ValueError('invalid collection')
        if validated['checkpoint_sha256']!=manifest['checkpoints'][role]['sha256']:raise ValueError('wrong checkpoint')
        baseline=manifest['prior_outputs'][role]
        if sha(ROOT/baseline['path'])!=baseline['sha256']:raise ValueError('archived report changed')
        stored={(r['index'],r['position']):r for r in validated['conditions']}
        if len(stored)!=96:raise ValueError('incomplete collected panel')
        values={}
        for path,digest in validated['raw_file_sha256'].items():
            if sha(path)!=digest:raise ValueError('raw file changed')
            for row in [json.loads(x) for x in Path(path).read_text().splitlines()]:
                if row['kind']!='condition':continue
                ex=example(row['index']);first=ex['information_set'];hidden=[i for i in range(8) if i not in first]
                ce=[];floors=[];gaps=[];unaffected=[];correct=[];second=[]
                for h,bits in enumerate(itertools.product((0,1),repeat=4)):
                    ys=[y for y in ex['support'] if all((y>>i)&1==b for i,b in zip(first,bits))]
                    if len(ys)!=1:raise ValueError('invalid target history')
                    y=ys[0]
                    second.append(-sum(row['original'][h][i][(y>>i)&1] for i in hidden))
                    for r,mask in enumerate(ex['matrix_rows']):
                        target=[i for i in hidden if (mask>>i)&1]
                        if len(target)!=1:raise ValueError('constraint does not identify one hidden coordinate')
                        i=target[0];gold=(y>>i)&1
                        original=row['original'][h];flipped=row['flipped'][r][h]
                        loss=-(original[i][gold]+flipped[i][1-gold])/2
                        sign=1-2*gold
                        l0=sign*(original[i][1]-original[i][0]);l1=sign*(flipped[i][1]-flipped[i][0]);v=-(l1-l0)/2
                        floor=max(v,0)+math.log1p(math.exp(-abs(v)))
                        if loss+1e-9<floor:raise ValueError('convexity bound failed')
                        ce.append(loss);floors.append(floor);gaps.append(max(0,loss-floor));correct.append(l0<0<l1)
                        unaffected.append(sum(abs(math.exp(flipped[j][1])-math.exp(original[j][1])) for j in hidden if j!=i)/3)
                        if role.startswith('balanced_'):
                            key=(role,row['position'])
                            counts=collateral_counts.setdefault(key,[0,0])
                            for j in hidden:
                                if j==i:continue
                                g=(y>>j)&1;counts[1]+=1
                                counts[0]+=int(original[j][g]>original[j][1-g] and flipped[j][g]<=flipped[j][1-g])
                initial=sum(-math.log(2)-sum(row['initial'][i])/2 for i in first)
                m=dict(mean_correct_conditional_ce=float(np.mean(ce)),response_ce_lower_bound=float(np.mean(floors)),
                       centering_penalty=float(np.mean(gaps)),both_variants_correct=float(np.mean(correct)),
                       unaffected_mean_abs_probability_change=float(np.mean(unaffected)))
                saved=stored[row['index'],row['position']]
                errors=[abs(saved['response'][k]-v) for k,v in m.items()]
                errors += [abs(saved['second_error']-np.mean(second)),abs(saved['first_error']-initial)]
                max_error=max(max_error,*errors)
                if max(errors)>1e-9:raise ValueError('independent metric mismatch')
                if initial+8*m['mean_correct_conditional_ce']+1e-9<initial+np.mean(second):raise ValueError('response upper bound failed')
                values[row['index'],row['position']]=m['mean_correct_conditional_ce'];cells+=1;pairs+=len(ce)
        if len(values)!=96:raise ValueError('raw panel incomplete')
        metrics[role]=values
    if cells!=672 or pairs!=43008:raise ValueError('wrong complete counts')
    collateral=json.loads((folder/'collateral_diagnostic.json').read_text())['checkpoints']
    for (role,position),(failures,total) in collateral_counts.items():
        if failures!=collateral[role][position]['correct_to_incorrect'] or total!=collateral[role][position]['unaffected_predictions']:
            raise ValueError('collateral-error count mismatch')
    effects=np.array([[metrics[f'near_seed{s}'][i,'far']-metrics[f'balanced_seed{s}'][i,'far'] for i in range(32)] for s in (20271011,20271012,20271013)])
    avg=effects.mean(axis=0);indices=np.random.default_rng(20260923).integers(0,32,size=(10000,32))
    ci=np.quantile(avg[indices].mean(axis=1),[.025,.975]);primary=report['seed_averaged']['far_mean_correct_conditional_ce_reduction']
    if abs(avg.mean()-primary['mean'])>1e-9 or max(abs(ci-np.array(primary['ci95'])))>1e-9:raise ValueError('primary paired inference mismatch')
    resources=json.loads((folder/'resource_audit.json').read_text())
    if not resources['all_jobs_terminal'] or resources['live_campaign']['live_reserved_gpus']!=0:raise ValueError('resources remain live')
    pdf=ROOT/'build/Long_RWKV_ICLR2027.pdf';review=json.loads((ROOT/'results/paper_audit/history_response_visual_review.json').read_text())
    if review['pdf_sha256']!=sha(pdf) or not review['all_pages_reviewed']:raise ValueError('PDF review missing/stale')
    log=(ROOT/'build/focused-pass-3.log').read_text()
    if any(word in log for word in ('undefined','multiply defined','Overfull')):raise ValueError('unclean PDF build')
    result=dict(complete=True,conditions=cells,paired_interventions=pairs,independent_max_arithmetic_discrepancy=max_error,
                primary_mean=float(avg.mean()),primary_ci95=ci.tolist(),manifest_sha256=sha(manifestpath),pdf_sha256=sha(pdf),
                balanced_collateral_errors=sum(v[0] for v in collateral_counts.values()),
                balanced_unaffected_comparisons=sum(v[1] for v in collateral_counts.values()),
                archive_sha256=archive['sha256'],all_jobs_terminal=True,live_reserved_gpus=0,
                scope='Frozen source/raw hashes, archived prior outputs, independently recomputed target-law metrics and paired inference, algebraic bound, scheduler accounting and reviewed clean PDF.')
    (folder/'delivery_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
