"""Independently check the complete frozen panel before reporting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lrwkv_evidence.distance_intervention import tasks as T
from lrwkv_evidence.train04.worker import load_tokenizer
from lrwkv_evidence.long_context_eval.numerical import validate as validate_numerical


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(planpath):
    plan=json.loads(planpath.read_text());stage=Path(plan['stage']);out=Path(plan['out'])
    manifestpath=ROOT/'theory_mvp/distance_intervention/FROZEN_EVALUATION.json'
    if sha(manifestpath)!=plan['manifest_sha256']:raise ValueError('manifest changed')
    for rel,digest in plan['sources'].items():
        if sha(stage/rel)!=digest:raise ValueError('staged source changed: '+rel)
    checkpoint=Path(json.loads(Path(plan['training_plan']).read_text())['out'])/'resume.pt'
    if sha(checkpoint)!=plan['checkpoint_sha256']:raise ValueError('checkpoint changed')
    tok,_,_=load_tokenizer(Path('/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B'))
    conditions=[];timings=[];files={};provenances=[]
    for rank in range(8):
        path=out/f'rank{rank}.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        files[str(path)]=sha(path)
        if rows[-1].get('kind')!='complete' or rows[-1].get('conditions')!=18:raise ValueError('rank incomplete')
        provenance=[r for r in rows if r['kind']=='provenance']
        if len(provenance)!=1:raise ValueError('bad provenance count')
        pr=provenance[0]
        for key,value in dict(rank=rank,world=8,role=plan['role'],seed=T.SEED,
                              checkpoint_sha256=plan['checkpoint_sha256'],manifest_sha256=plan['manifest_sha256']).items():
            if pr[key]!=value:raise ValueError('provenance mismatch: '+key)
        provenances.append(pr)
        checks=[r for r in rows if r['kind']=='numerical_screen']
        if len(checks)!=1:raise ValueError('missing numerical screen')
        validate_numerical(checks[0]['rows'])
        checks32=[r for r in rows if r['kind']=='numerical_32k']
        if [r['position'] for r in checks32]!=['far','near']:raise ValueError('32K screen incomplete')
        if any(not math.isfinite(r[k]) or r[k]<0 or r[k]>(1e-5 if k=='within' else 1e-4) for r in checks32 for k in ('within','cross')):
            raise ValueError('32K numerical failure')
        rc=[r for r in rows if r['kind']=='condition'];rt=[r for r in rows if r['kind']=='timing']
        if len(rc)!=18 or len(rt)!=9:raise ValueError('missing rank cells')
        expected={(i,16384,p) for i in range(rank,32,8) for p in ('far','middle','near')}
        expected|={(rank,None,'evidence_only'),*( (rank,4096,p) for p in ('far','middle','near')),*( (rank,32768,p) for p in ('far','near'))}
        actual=[(r['index'],r['serial']['requested_context_tokens'],r['serial']['position']) for r in rc]
        if len(set(actual))!=18 or set(actual)!=expected:raise ValueError('wrong condition selectors')
        for row in rc:
            ex=T.example(row['index']);encoded=T.serialize(ex,tok,row['serial']['requested_context_tokens'],row['serial']['position'])
            if row['serial']!={k:v for k,v in encoded.items() if k!='ids'}:raise ValueError('native input mismatch')
            if row['primary']!=(row['serial']['requested_context_tokens']==16384):raise ValueError('wrong primary indicator')
            exact=row['exact']
            if exact['instance_id']!=ex['instance_id'] or exact['support']!=ex['support']:raise ValueError('wrong target law')
            methods={r['method']:r for r in exact['results']}
            if len(exact['results'])!=3 or set(methods)!={'one','information_set','pair_preserving_halves'}:raise ValueError('wrong policy set')
            for name,r in methods.items():
                logq=r['support_log_probabilities']
                if len(logq)!=16 or not all(math.isfinite(v) and v<=1e-8 for v in logq):raise ValueError('invalid endpoint probabilities')
                kl=-math.log(16)-sum(logq)/16
                maximum=max(logq);logmass=maximum+math.log(sum(math.exp(v-maximum) for v in logq));mass=math.exp(logmass)
                dependence=0 if name=='information_set' else 4*math.log(2)
                if abs(kl-r['joint_kl_nats'])>1e-8 or abs(mass-r['exact_valid_mass'])>1e-8 or mass>1+1e-8:raise ValueError('endpoint metric mismatch')
                if abs(r['dependence_nats']-dependence)>1e-8 or abs(kl-dependence-r['estimation_nats'])>1e-8 or r['estimation_nats'] < -1e-8:raise ValueError('decomposition mismatch')
                if abs(logmass-r['log_valid_mass'])>1e-8 or abs(kl+logmass-r['within_valid_kl_nats'])>1e-8:raise ValueError('within-support mismatch')
            st=row['stages']
            if abs(st['first_round_estimation_nats']+st['second_round_estimation_nats']-methods['information_set']['estimation_nats'])>1e-8:raise ValueError('stage sum mismatch')
            cf,meta=T.counterfactual(ex);changed=T.serialize(cf,tok,row['serial']['requested_context_tokens'],row['serial']['position'])
            actualcf=row['counterfactual']
            if actualcf['changed_native_token_index']!=T.validate_layout(encoded,changed) or actualcf['flipped_token_sha256']!=changed['token_sha256']:raise ValueError('counterfactual layout mismatch')
            # JSON converts visible-coordinate keys to strings.
            if actualcf['meta']!=json.loads(json.dumps(meta)):raise ValueError('counterfactual target mismatch')
            r=actualcf['response'];i=meta['target_coordinate'];a=meta['original_target'];b=meta['flipped_target'];sign=1 if b else -1
            lp0=r['original_target_logprobs'];lp1=r['flipped_target_logprobs']
            if any(abs(sum(math.exp(v) for v in lp)-1)>1e-6 for lp in (lp0,lp1)):raise ValueError('unnormalized response')
            delta=sign*((lp1[1]-lp1[0])-(lp0[1]-lp0[0]));ce=-(lp0[a]+lp1[b])/2
            floor=max(-delta/2,0)+math.log1p(math.exp(-abs(delta/2)))
            if abs(delta-r['signed_logit_contrast'])>1e-8 or abs(ce-r['mean_correct_conditional_ce'])>1e-8 or abs(floor-r['response_ce_lower_bound'])>1e-8 or abs(ce-floor-r['centering_penalty'])>1e-8:raise ValueError('response metric mismatch')
        expectedtiming={(rank,p,m) for p in ('far','middle','near') for m in ('one','information_set','pair_preserving_halves')}
        if {(r['index'],r['position'],r['method']) for r in rt}!=expectedtiming:raise ValueError('timing selectors mismatch')
        for r in rt:
            if r['warmups']!=1 or len(r['seconds'])!=5 or len(r['peak_allocated_bytes'])!=5 or not all(math.isfinite(v) and v>0 for v in r['seconds']):raise ValueError('timing repeats incomplete')
        conditions.extend(rc);timings.extend(rt)
    return dict(role=plan['role'],checkpoint_sha256=plan['checkpoint_sha256'],manifest_sha256=plan['manifest_sha256'],
                validated=True,conditions=conditions,timings=timings,provenance=provenances,raw_file_sha256=files,
                condition_count=len(conditions),timed_requests=sum(len(r['seconds']) for r in timings))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--role',required=True);args=ap.parse_args()
    folder=ROOT/'results/distance_intervention';result=collect(folder/f'plan_eval_{args.role}.json')
    path=folder/f'exact_{args.role}.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(role=args.role,validated=True,conditions=result['condition_count'],timed_requests=result['timed_requests'])))


if __name__=='__main__':main()
