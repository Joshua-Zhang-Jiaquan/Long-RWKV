"""Completion requires full fixed evidence and paper integration, not passing tests alone."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'qz')]
FOLDER=ROOT/'results/distance_intervention'
SEEDS=(20271011,20271012,20271013)
ROLES=[f'{a}_seed{s}' for s in SEEDS for a in ('near','balanced')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a,b):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>1e-8:raise ValueError(f'numeric mismatch: {a} != {b}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--live',action='store_true');args=ap.parse_args()
    checks=[]
    def check(name,fn):
        try:
            details=fn();checks.append(dict(requirement=name,passed=True,evidence=details))
        except Exception as error:checks.append(dict(requirement=name,passed=False,error=repr(error)))
    def frozen():
        manifests={
            'FROZEN_DESIGN.json':['training_sources'],
            'FROZEN_EVALUATION.json':['sources'],
            'ANALYSIS_MANIFEST.json':['files'],
            'OBJECTIVE_NORMALIZATION_MANIFEST.json':['files'],
            'FROZEN_COST_SUPPLEMENT.json':['sources','analysis_sources']}
        count=0;hashes={}
        for name,keys in manifests.items():
            path=ROOT/'theory_mvp/distance_intervention'/name;d=json.loads(path.read_text());hashes[str(path)]=sha(path)
            for key in keys:
                for rel,digest in d[key].items():
                    if sha(ROOT/rel)!=digest:raise ValueError('frozen source changed: '+rel)
                    count+=1
        design=json.loads((ROOT/'theory_mvp/distance_intervention/FROZEN_DESIGN.json').read_text())
        if sha(ROOT/'theory_mvp/distance_intervention/DESIGN.md')!=design['design_markdown_sha256']:raise ValueError('protocol prose changed')
        qualification=json.loads((FOLDER/'plan_qualification.json').read_text())
        if sha(ROOT/'theory_mvp/distance_intervention/FROZEN_DESIGN.json')!=qualification['design_sha256']:raise ValueError('training protocol differs from submitted qualification')
        original=json.loads((FOLDER/'plan_eval_original.json').read_text())
        if sha(ROOT/'theory_mvp/distance_intervention/FROZEN_EVALUATION.json')!=original['manifest_sha256']:raise ValueError('evaluation protocol differs from submitted original panel')
        return dict(verified_file_entries=count,manifest_sha256=hashes)
    check('Frozen training, evaluation, analysis and cost protocols intact',frozen)
    def training():
        from continue_distance_intervention import terminal_training
        q=json.loads((FOLDER/'plan_qualification.json').read_text())
        qualification=terminal_training(q)
        if not qualification or not qualification['completion']['qualification'] or qualification['steps']!=6:raise ValueError('qualification not complete')
        execution=json.loads((FOLDER/'FROZEN_EXECUTION.json').read_text())
        if execution['terminal_steps']!=600:raise ValueError('different frozen terminal budget')
        completed={}
        for role in ROLES:
            plan=json.loads((FOLDER/f'plan_{role}.json').read_text());v=terminal_training(plan)
            if not v or v['steps']!=600 or v['completion']['qualification']:raise ValueError('fixed terminal missing: '+role)
            if plan['source_sha256']!=q['source_sha256'] or plan['sources']!=q['sources']:raise ValueError('training source differs from qualification')
            if role!=f"{v['completion']['arm']}_seed{v['completion']['seed']}":raise ValueError('wrong matched selector')
            completed[role]=dict(checkpoint_sha256=v['completion']['checkpoint_sha256'],input_tokens=v['input_tokens'])
        for seed in SEEDS:
            if completed[f'near_seed{seed}']['input_tokens']!=completed[f'balanced_seed{seed}']['input_tokens']:raise ValueError('unmatched training token budget')
        return dict(qualification=qualification,terminal_checkpoints=completed)
    check('Qualification and all six matched600-update checkpoints complete',training)
    def exact():
        counts={};total_conditions=total_timings=0
        for role in ['original']+ROLES:
            data=json.loads((FOLDER/f'exact_{role}.json').read_text())
            plan=json.loads((FOLDER/f'plan_eval_{role}.json').read_text())
            if not data['validated'] or data['checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('wrong validated checkpoint')
            if data['condition_count']!=144 or data['timed_requests']!=360:raise ValueError('missing exact/timing cells')
            if len(data['conditions'])!=144 or len(data['timings'])!=72:raise ValueError('counts disagree with arrays')
            for path,digest in data['raw_file_sha256'].items():
                if sha(path)!=digest:raise ValueError('raw endpoint data changed')
            if any('H100' not in p['runtime']['gpu_name'] or p['runtime']['compute_capability']!=[9,0] for p in data['provenance']):raise ValueError('unexpected evaluation GPU')
            total_conditions+=144;total_timings+=360;counts[role]=sha(FOLDER/f'exact_{role}.json')
        if total_conditions!=1008 or total_timings!=2520:raise ValueError('incomplete fixed study')
        return dict(conditions=total_conditions,timed_requests=total_timings,artifact_sha256=counts)
    check('All1008 accuracy conditions, counterfactuals and2520 RWKV timings retained',exact)
    def numerical():
        report=json.loads((FOLDER/'final_report.json').read_text())
        values={}
        for role in ['original']+ROLES:
            data=json.loads((FOLDER/f'exact_{role}.json').read_text());values[role]={}
            for pos in ('far','middle','near'):
                rr=sorted([r for r in data['conditions'] if r['primary'] and r['serial']['position']==pos],key=lambda r:r['index'])
                if [r['index'] for r in rr]!=list(range(32)):raise ValueError('wrong primary indices')
                kl=[]
                for r in rr:
                    row=next(m for m in r['exact']['results'] if m['method']=='information_set')
                    value=-math.log(16)-sum(row['support_log_probabilities'])/16
                    close(value,row['joint_kl_nats']);kl.append(value)
                values[role][pos]=kl
                close(sum(kl)/32,report['checkpoints'][role]['primary'][pos]['information_set_kl']['mean'])
            close(sum(sum(v) for v in values[role].values())/96,
                  report['checkpoints'][role]['primary']['mean_positions']['information_set_kl']['mean'])
        effects=[]
        for seed in SEEDS:
            a=values[f'near_seed{seed}']['far'];b=values[f'balanced_seed{seed}']['far']
            effect=sum(x-y for x,y in zip(a,b))/32;effects.append(effect)
            close(effect,report['intervention']['per_adaptation_seed'][str(seed)]['far_conditional_error_reduction']['mean'])
        close(sum(effects)/3,report['intervention']['seed_averaged_problem_effects']['far_conditional_error_reduction']['mean'])
        return dict(independent_primary_effects=effects,source='Recomputed KL directly from16 stored endpoint log probabilities; no reported KL used for means')
    check('Primary findings independently recomputed from endpoint probabilities',numerical)
    def cost():
        folder=ROOT/'results/distance_intervention_cost'
        attention=json.loads((folder/'exact_attention.json').read_text());cert=json.loads((folder/'budget_certificates.json').read_text())
        if not attention['execution_complete'] or len(attention['conditions'])!=24:raise ValueError('attention panel incomplete')
        count=sum(len(c['seconds']) for r in attention['conditions'] for c in r['costs'])
        if count!=360:raise ValueError('missing attention timings')
        attention_means={method:sum(v for row in attention['conditions'] for c in row['costs'] if c['method']==method for v in c['seconds'])/120
                         for method in ('one','information_set','pair_preserving_halves')}
        for path,digest in attention['raw_sha256'].items():
            if sha(path)!=digest:raise ValueError('attention raw timings changed')
        if {r['role'] for r in cert['certificates']}!=set(ROLES) or len(cert['certificates'])!=6:raise ValueError('missing checkpoint certificate')
        for r in cert['certificates']:
            data=json.loads((FOLDER/f"exact_{r['role']}.json").read_text())
            selected=[x for x in data['conditions'] if x['primary'] and x['index']<8]
            means=[-math.log(16)-sum(next(m['support_log_probabilities'] for m in x['exact']['results'] if m['method']=='information_set'))/16 for x in selected]
            quality=sum(means)/24;close(quality,r['quality']['information_set']['mean'])
            times=[v for row in data['timings'] if row['method']=='information_set' for v in row['seconds']]
            if len(times)!=120:raise ValueError('wrong matched recurrent timing count')
            recurrent_mean=sum(times)/120;close(recurrent_mean,r['recurrent_timing']['information_set']['mean'])
            expected=dict(recurrent_two_call_fits=recurrent_mean<=1.5,
                          attention_one_call_fits=attention_means['one']<=1.5,
                          attention_information_set_excluded=attention_means['information_set']>1.5,
                          attention_intact_pair_excluded=attention_means['pair_preserving_halves']>1.5,
                          recurrent_kl_below_product_floor=quality<4*math.log(2))
            if r['conditions']!=expected:raise ValueError('cost conditions disagree with raw timings and matched endpoint probabilities')
            if r['empirical_sufficient_certificate']!=all(r['conditions'].values()):raise ValueError('certificate truth mismatch')
        return dict(attention_requests=count,all_checkpoint_certificates_retained=True,artifact_sha256=sha(folder/'budget_certificates.json'))
    check('Same-case cost supplement complete, with all six certificates',cost)
    def paper():
        required=['paper/distance_intervention_methods.tex','paper/distance_intervention_results.tex',
                  'results/distance_intervention/FINAL_REPORT.md','results/distance_intervention/mechanism_distance.pdf',
                  'results/distance_intervention_cost/BUDGET_REPORT.md','results/paper_audit/distance_intervention_claim_audit.md',
                  'results/paper_audit/distance_intervention_visual_review.json','build/Long_RWKV_ICLR2027.pdf',
                  'results/distance_intervention/mechanism_distance_paper.pdf',
                  'results/distance_intervention/publication_figure_receipt.json',
                  'results/distance_intervention/paper_intervention_table.tex',
                  'results/distance_intervention/paper_quality_cost_table.tex',
                  'results/distance_intervention/paper_secondary_table.tex']
        hashes={}
        for rel in required:
            p=ROOT/rel
            if not p.exists() or not p.stat().st_size:raise ValueError('missing delivery artifact: '+rel)
            hashes[rel]=sha(p)
        source=(ROOT/'Long_RWKV_ICLR2027.tex').read_text()
        for rel in required[:2]:
            if '\\input{'+rel+'}' not in source:raise ValueError('new study not integrated into paper: '+rel)
        manuscript=source+'\n'+(ROOT/'paper/distance_intervention_results.tex').read_text()
        for marker in ('working draft','results are pending','results are still pending','results are not yet','has not yet incorporated'):
            if marker in manuscript.lower():raise ValueError('unfinished manuscript text remains: '+marker)
        claim_audit=(ROOT/'results/paper_audit/distance_intervention_claim_audit.md').read_text()
        if 'Status: COMPLETE.' not in claim_audit or 'Status: INCOMPLETE.' in claim_audit:
            raise ValueError('claim audit has not been finalized against terminal evidence')
        log=(ROOT/'build/focused-pass-3.log').read_text()
        if any(s in log for s in ('undefined','multiply defined','Overfull')):raise ValueError('unclean final PDF build')
        review=json.loads((ROOT/'results/paper_audit/distance_intervention_visual_review.json').read_text())
        if review.get('pdf_sha256')!=hashes['build/Long_RWKV_ICLR2027.pdf'] or not review.get('reviewed'):raise ValueError('final PDF not visually reviewed')
        figure=json.loads((FOLDER/'publication_figure_receipt.json').read_text())
        if not figure['plotted_coordinates_unchanged'] or figure['frozen_renderer_sha256']!=sha(ROOT/'theory_mvp/distance_intervention/figures.py'):
            raise ValueError('publication figure differs from frozen plotted data')
        delivery=(ROOT/'PAPER_DELIVERY.md').read_text()
        if 'Follow-up in progress' in delivery or 'distance_intervention/FINAL_REPORT.md' not in delivery:raise ValueError('delivery note not finalized')
        return dict(artifact_sha256=hashes,source_sha256=sha(ROOT/'Long_RWKV_ICLR2027.tex'))
    check('Paper integrates the complete study and rendered artifacts are reviewed',paper)
    def resources():
        r=json.loads((FOLDER/'resource_audit.json').read_text())
        if not r['all_jobs_terminal']:raise ValueError('resource audit still has live jobs')
        if not args.live:raise ValueError('live scheduler verification required for completion')
        import campaign as C
        from confirm_capacity import live_usage,PROJECT_CAPS,CAP
        jobs=C.list_all_jobs();byid={j['job_id']:j for j in jobs};census=live_usage(jobs)
        if census['live_reserved_gpus']:raise ValueError('campaign still reserves GPUs')
        for row in r['jobs']:
            job=byid[row['job_id']]
            if job['status'] not in C.TERMINAL_STATUS:raise ValueError('job still live')
            plan=json.loads(Path(row['plan']).read_text());before=plan['capacity'];project=plan['budget_project_id']
            if before['live_reserved_gpus']+8>CAP or before['project_reserved_gpus'][project]+8>PROJECT_CAPS[project]:raise ValueError('admission budget violation')
        failures=[row for row in r['jobs'] if row['initial_launch_failed_before_model_load']]
        if len(failures)!=3 or any(row['logged_updates'] for row in failures):raise ValueError('launch-failure accounting missing')
        timeline=json.loads((FOLDER/'capacity_timeline_audit.json').read_text())
        if not timeline['complete'] or {v['job_id'] for v in timeline['jobs']}!={v['job_id'] for v in r['jobs']}:
            raise ValueError('capacity timeline does not cover all jobs')
        usage={p:0 for p in PROJECT_CAPS};peaks=usage.copy();peak=0;events=[]
        for row in timeline['jobs']:
            if row['gpus']!=8 or row['finished_ms']<row['created_ms']:raise ValueError('invalid reservation interval')
            events.extend([(row['created_ms'],1,row['project_id'],8),(row['finished_ms'],-1,row['project_id'],8)])
        for stamp,sign,project,gpus in sorted(events):
            usage[project]+=sign*gpus;peaks[project]=max(peaks[project],usage[project]);peak=max(peak,sum(usage.values()))
            if usage[project]<0 or usage[project]>PROJECT_CAPS[project] or sum(usage.values())>CAP:raise ValueError('timeline capacity violation')
        if any(usage.values()) or peaks!=timeline['project_peaks'] or peak!=timeline['peak_queued_or_running_gpus']:
            raise ValueError('reservation timeline summary mismatch')
        return dict(live_census=census,jobs_accounted=len(r['jobs']),failed_launches_accounted=len(failures),running_gpu_hours=r['total_scheduler_running_gpu_hours'],
                    peak_queued_or_running_gpus=peak,project_peaks=peaks)
    check('All jobs terminal, launch failures accounted,32+16cap respected',resources)
    result=dict(at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),complete=all(c['passed'] for c in checks),checks=checks)
    (FOLDER/'delivery_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['complete'] else 2)


if __name__=='__main__':main()
