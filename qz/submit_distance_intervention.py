"""Immutable, capacity-checked 16K distance study admission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E
from confirm_capacity import live_usage, route_body, PRIMARY
from submission_lock import campaign_submission_lock
from distance_shared import stage_execution
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from lrwkv_evidence.train04confirm.contract import sha
from lrwkv_evidence.train04.worker import canonical_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qualification', action='store_true')
    ap.add_argument('--qualification-plan', type=Path)
    ap.add_argument('--arm', choices=('near', 'balanced'), default='balanced')
    ap.add_argument('--seed', type=int, default=20271011)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--repair-execution-path', action='store_true')
    args = ap.parse_args()
    folder = ROOT/'results/distance_intervention'; folder.mkdir(parents=True, exist_ok=True)
    repair=None
    role=f'{args.arm}_seed{args.seed}'
    existing=folder/f'plan_{role}.json'
    if args.repair_execution_path:
        if args.qualification or not args.submit:raise ValueError('explicit main-job repair submission only')
        old=json.loads(existing.read_text())
        jobs={j['job_id']:j for j in C.list_all_jobs()}
        job=jobs.get(old['submission']['job_id'])
        if job is None or job['status']!='job_failed':raise ValueError('repair requires verified failed scheduler handle')
        oldout=Path(old['out'])
        if any((oldout/n).exists() for n in ('train.jsonl','provenance.json','resume.pt','completion.json')):raise ValueError('repair only before training/model load')
        log=(oldout/'torchrun.log').read_text()
        if 'FileNotFoundError' not in log or 'FROZEN_EXECUTION.json' not in log:raise ValueError('not the diagnosed path failure')
        repair=dict(original_plan=old,reason='project-local execution manifest not mounted on primary-project GPU nodes; unchanged bytes moved to global shared storage',verified_status=job['status'])
    designpath = ROOT/'theory_mvp/distance_intervention/FROZEN_DESIGN.json'
    design = json.loads(designpath.read_text())
    if sha(ROOT/'theory_mvp/distance_intervention/DESIGN.md') != design['design_markdown_sha256']:
        raise ValueError('changed frozen protocol')
    original = json.loads((ROOT/'results/long_context_mvp/plan_dacbbe1355018e7f_qualification.json').read_text())
    initialplan = ROOT/'results/train04confirm/plan_8cd809d08a96ea13_confirm_independent_seed71.json'
    initial = json.loads(initialplan.read_text()); checkpoint = Path(initial['out'])/'resume.pt'
    if sha(checkpoint) != design['initial_checkpoint_sha256']: raise ValueError('starting checkpoint changed')
    if args.qualification:
        if args.arm != 'balanced' or args.seed != 20271011: raise ValueError('qualification selector')
        sources = {rel:Path(original['stage'])/rel for rel in original['sources']}
        if any(sha(p) != original['sources'][rel] for rel,p in sources.items()): raise ValueError('inherited source changed')
        for rel, digest in design['training_sources'].items():
            if sha(ROOT/rel) != digest: raise ValueError('frozen runtime changed')
            sources[rel] = ROOT/rel
        sources['distance_design.json'] = designpath
        steps = 6
    else:
        if not args.qualification_plan: raise ValueError('qualification plan required')
        q = json.loads(args.qualification_plan.read_text())
        sources = {rel:Path(q['stage'])/rel for rel in q['sources']}
        if any(sha(p) != q['sources'][rel] for rel,p in sources.items()): raise ValueError('qualified source changed')
        completion = json.loads((Path(q['out'])/'completion.json').read_text())
        if not completion.get('qualification') or not completion.get('execution_complete') or completion['step'] != 6:
            raise ValueError('qualification incomplete')
        execution = json.loads((folder/'FROZEN_EXECUTION.json').read_text())
        if execution['source_sha256'] != canonical_sha(q['sources']): raise ValueError('unqualified execution source')
        if args.seed not in design['adaptation_seeds']: raise ValueError('undeclared adaptation seed')
        steps = execution['terminal_steps']
    hashes = {rel:sha(p) for rel,p in sources.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    stage = Path(E.G)/'long_rwkv_distance_intervention_stages'/digest
    phase = 'qualification' if args.qualification else f'{args.arm}_seed{args.seed}'
    attempt_suffix='_pathfix1' if repair else ''
    out = Path(E.OUTPUTS)/f'distance_intervention_{digest}_{phase}{attempt_suffix}'
    env = dict(DISTANCE_ROOT=str(stage), DISTANCE_OUT=str(out), DISTANCE_ARM=args.arm,
               DISTANCE_SEED=str(args.seed), DISTANCE_STEPS=str(steps), DISTANCE_QUALIFICATION=str(int(args.qualification)),
               DISTANCE_INITIAL=str(checkpoint), DISTANCE_INITIAL_SHA256=design['initial_checkpoint_sha256'],
               TRITON_F32_DEFAULT='ieee', TRITON_CACHE_DIR=str(out/'triton_ieee_cache'))
    if not args.qualification:
        env['DISTANCE_QUALIFICATION_OUT'] = q['out']
        execution_local=folder/'FROZEN_EXECUTION.json'
        execution_shared=stage_execution(execution_local,E.G)
        env['DISTANCE_EXECUTION'] = str(execution_shared)
    body = E.make_body(f'lrwkv-dist-{digest[:8]}-{phase}{attempt_suffix}', E.wrapped(env, str(stage/'qz/launch_distance_intervention.sh')),
                       'Matched 16K distance intervention; balanced history coverage; 8H100 under32+16cap', E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False, fault_tolerance_max_retry=0,
                max_running_time_ms=str((3 if args.qualification else 13)*3600000))
    record = dict(stage=str(stage), out=str(out), sources=hashes, source_sha256=canonical_sha(hashes),
                  body=body, phase=phase, arm=args.arm, seed=args.seed, steps=steps,
                  initial_checkpoint_sha256=design['initial_checkpoint_sha256'], design_sha256=sha(designpath))
    dest = folder/f'plan_{phase}.json'
    if dest.exists() and json.loads(dest.read_text()).get('submission') and not repair: raise ValueError('refuse duplicate submission')
    if repair:record['operational_repair']=repair
    if args.submit:
        record['capacity'] = live_usage(C.list_all_jobs())
        record['budget_project_id'] = route_body(body, record['capacity'], preferred_project=PRIMARY)
        if C.already_submitted(body['name']): raise ValueError('duplicate scheduler name')
        for rel,p in sources.items():
            if sha(p) != hashes[rel]: raise ValueError('source changed during staging')
            target = stage/rel; target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != hashes[rel]: raise ValueError('immutable stage conflict')
            if not target.exists(): shutil.copyfile(p, target)
        (stage/'distance_sources.json').write_text(json.dumps(hashes, indent=2)+'\n')
        if repair:
            archived=folder/'failed_attempts'/dest.name;archived.parent.mkdir(parents=True,exist_ok=True)
            if archived.exists():raise ValueError('failed receipt already archived; inspect previous repair before proceeding')
            shutil.copyfile(dest,archived)
        record['submission'] = C.submit_one(body, dry_run=False)
    dest.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','body','operational_repair')}, indent=2))


if __name__ == '__main__':
    with campaign_submission_lock(ROOT): main()
