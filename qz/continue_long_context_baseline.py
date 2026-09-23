"""Qualify comparators during main training; train only after main competence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from confirm_capacity import live_usage, CAP

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'job_failed', 'job_stopped', 'job_cancelled', 'job_canceled', 'job_succeeded'}


def read(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qualification-plan', type=Path, required=True)
    ap.add_argument('--main-training-plan', type=Path, required=True)
    ap.add_argument('--hours', type=float, default=6)
    args = ap.parse_args()
    initial = read(args.qualification_plan)
    kind = initial['kind']; digest = Path(initial['stage']).name
    folder = ROOT / 'results/long_context_mvp'
    state_path = folder / f'baseline_{kind}_watcher.json'
    state = dict(status='waiting_for_main_training_submission', kind=kind,
                 qualification_plan=str(args.qualification_plan.resolve()), started_unix=time.time())
    deadline = time.monotonic() + args.hours*3600
    def save():
        state['updated_unix'] = time.time()
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(state_path)
    def submit(script, *arguments):
        state['capacity'] = live_usage(C.list_all_jobs())
        if state['capacity']['live_reserved_gpus'] + 8 > CAP:
            return False
        result = subprocess.run([sys.executable, str(ROOT / 'qz' / script), *map(str, arguments), '--submit'],
                                cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            if f'{CAP} GPU cap' in result.stdout + result.stderr:
                return False
            raise RuntimeError((result.stdout + result.stderr)[-2000:])
        print(result.stdout, flush=True)
        return True
    def phase_ready(phase):
        path = folder / f'plan_baseline_{kind}_{digest}_{phase}.json'
        record = read(path) if path.exists() else None
        if record and record['sources'] != initial['sources']:
            raise ValueError('changed prepared comparator source set')
        if not record or not record.get('submission'):
            options = (['--qualification'] + (['--ieee'] if initial.get('ieee') else [])) if phase == 'qualification' else ['--qualification-plan', args.qualification_plan]
            submit('submit_long_context_baseline.py', '--kind', kind, '--expected-source-digest', digest, *options)
            return None
        out = Path(record['out'])
        if (out / 'completion.json').exists():
            return path
        status, _ = C.job_status(record['submission']['job_id'])
        state[phase] = dict(job_id=record['submission']['job_id'], scheduler_status=status)
        if status in TERMINAL:
            raise RuntimeError(f'{phase}: {status} without complete evidence')
        return None
    try:
        while time.monotonic() < deadline:
            if args.main_training_plan.exists() and read(args.main_training_plan).get('submission'):
                state['status'] = 'qualifying_comparator'
                qualification = phase_ready('qualification')
                if qualification:
                    main_out = Path(read(args.main_training_plan)['out'])
                    reports = []
                    if (main_out / 'completion.json').exists():
                        checkpoint = read(main_out / 'completion.json')['checkpoint_sha256']
                        reports = [read(p) for p in folder.glob('exact_development_*.json')
                                   if read(p)['checkpoint_sha256'] == checkpoint and read(p).get('comparator', 'rwkv') == 'rwkv']
                        if not reports:
                            reports = [read(p) for p in folder.glob('gate_development_*.json')
                                       if read(p)['checkpoint_sha256'] == checkpoint and read(p).get('short_gate_complete')]
                    if not reports:
                        state['status'] = 'qualified_waiting_for_main_competence'
                    elif len(reports) != 1:
                        raise ValueError('ambiguous main development result')
                    elif not (reports[0]['execution_complete'] or reports[0].get('short_gate_complete')):
                        raise ValueError('incomplete prerequisite competence report')
                    elif not reports[0]['competence_gate_passed']:
                        state['status'] = 'qualified_main_competence_failed_no_full_training'
                        save(); return
                    else:
                        state['status'] = 'training_comparator'
                        training_path = phase_ready('development600')
                        if training_path:
                            plans = [(p, read(p)) for p in folder.glob(f'plan_baseline_eval_{kind}_*.json')]
                            plans = [(p, r) for p, r in plans if Path(r['training_plan']).resolve() == training_path.resolve() and r.get('submission')]
                            if len(plans) > 1:
                                raise ValueError('duplicate submitted comparator evaluations')
                            if not plans:
                                submit('submit_long_context_baseline_eval.py', '--training-plan', training_path)
                                state['status'] = 'waiting_for_comparator_evaluation_submission'
                            else:
                                path, record = plans[0]
                                status, _ = C.job_status(record['submission']['job_id'])
                                state.update(status='evaluating_comparator', evaluation_plan=str(path), scheduler_status=status)
                                complete = True
                                for rank in range(8):
                                    file = Path(record['out']) / f'rank{rank}.jsonl'
                                    lines = file.read_text().splitlines() if file.exists() else []
                                    try:
                                        complete = complete and bool(lines) and json.loads(lines[-1]).get('kind') == 'complete'
                                    except json.JSONDecodeError:
                                        complete = False
                                if complete:
                                    destination = folder / f'exact_baseline_{kind}_{Path(record["stage"]).name}.json'
                                    result = subprocess.run([sys.executable, '-m', 'lrwkv_evidence.long_context_eval.collect',
                                                             '--plan', str(path), '--out', str(destination)],
                                                            cwd=ROOT, capture_output=True, text=True)
                                    if result.returncode:
                                        raise RuntimeError((result.stdout + result.stderr)[-2000:])
                                    state.update(status='comparator_development_complete', report=str(destination),
                                                 competence_gate_passed=read(destination)['competence_gate_passed'])
                                    save(); print(result.stdout, flush=True); return
                                if status in TERMINAL:
                                    raise RuntimeError(f'{status} without complete comparator evaluation')
            save(); time.sleep(30)
        state['status'] = 'watch_timeout_no_resubmission'; save()
    except Exception as error:
        state.update(status='needs_attention', error=str(error)); save(); raise


if __name__ == '__main__':
    main()
