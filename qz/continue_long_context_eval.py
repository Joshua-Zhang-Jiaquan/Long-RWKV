"""Wait for the fixed development checkpoint, submit once, and validate results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import campaign as C
from confirm_capacity import live_usage, CAP

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--training-plan', type=Path, required=True)
    ap.add_argument('--hours', type=float, default=6)
    ap.add_argument('--state-name', default='evaluation_watcher.json')
    args = ap.parse_args()
    training_path = args.training_plan.resolve()
    results = ROOT / 'results/long_context_mvp'
    if Path(args.state_name).name != args.state_name:raise ValueError('state name must be a basename')
    state_path = results / args.state_name
    deadline = time.monotonic() + args.hours * 3600
    state = dict(status='waiting_for_training', training_plan=str(training_path), started_unix=time.time())
    def save():
        state['updated_unix'] = time.time()
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
    try:
        while time.monotonic() < deadline:
            if training_path.exists():
                training = json.loads(training_path.read_text())
                if (Path(training['out']) / 'completion.json').exists():
                    matches = []
                    for path in results.glob('plan_eval_*.json'):
                        record = json.loads(path.read_text())
                        if Path(record['training_plan']).resolve() == training_path and record.get('submission'):
                            matches.append((path, record))
                    if len(matches) > 1:
                        raise ValueError('multiple submitted evaluations for this training run')
                    if not matches:
                        state['capacity'] = live_usage(C.list_all_jobs())
                        if state['capacity']['live_reserved_gpus'] + 8 <= CAP:
                            result = subprocess.run([sys.executable, str(ROOT / 'qz/submit_long_context_eval.py'),
                                                     '--training-plan', str(training_path), '--submit'],
                                                    cwd=ROOT, capture_output=True, text=True)
                            if result.returncode and f'{CAP} GPU cap' not in result.stdout + result.stderr:
                                raise RuntimeError((result.stdout + result.stderr)[-2000:])
                            print(result.stdout, flush=True)
                        state['status'] = 'waiting_for_evaluation_submission'
                    else:
                        path, record = matches[0]
                        job = record['submission']['job_id']
                        status, _ = C.job_status(job)
                        state.update(status='evaluation_running', evaluation_plan=str(path), job_id=job, scheduler_status=status)
                        files = [Path(record['out']) / f'rank{rank}.jsonl' for rank in range(8)]
                        complete = True
                        for file in files:
                            lines = file.read_text().splitlines() if file.exists() else []
                            try:
                                complete = complete and bool(lines) and json.loads(lines[-1]).get('kind') == 'complete'
                            except json.JSONDecodeError:
                                complete = False  # A writer may be in the middle of its final line.
                        if complete:
                            destination = results / f'exact_development_{Path(record["stage"]).name}.json'
                            result = subprocess.run([sys.executable, '-m', 'lrwkv_evidence.long_context_eval.collect',
                                                     '--plan', str(path), '--out', str(destination)],
                                                    cwd=ROOT, capture_output=True, text=True)
                            if result.returncode:
                                raise RuntimeError((result.stdout + result.stderr)[-2000:])
                            report = json.loads(destination.read_text())
                            state.update(status='development_evaluation_complete', report=str(destination),
                                         competence_gate_passed=report['competence_gate_passed'])
                            save()
                            print(result.stdout, flush=True)
                            return
                        if status in ('job_failed', 'job_stopped', 'job_cancelled', 'job_canceled', 'job_succeeded'):
                            raise RuntimeError(f'{status} without complete evaluation shards')
            save()
            time.sleep(30)
        state['status'] = 'watch_timeout_no_resubmission'
        save()
    except Exception as error:
        state.update(status='needs_attention', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
