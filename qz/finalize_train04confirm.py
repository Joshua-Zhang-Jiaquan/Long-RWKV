"""Finish the CPU audits after all frozen jobs complete; never submit GPU jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=6)
    args = parser.parse_args()
    folder = ROOT/'results/train04confirm'
    manifest = ROOT/'theory_mvp/train04confirm/FROZEN_RECIPE.json'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    state_path = folder/'finalization_watcher.json'
    state = dict(status='waiting_for_complete_training', manifest_sha256=digest,
                 started_unix=time.time())
    deadline = time.monotonic()+3600*args.hours
    training_audited = False
    def save():
        state['updated_unix'] = time.time()
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(state_path)
    def ready():
        for seed in (53,71,89):
            for phase in ('parent','independent','complementary'):
                path = folder/f'plan_{digest[:16]}_confirm_{phase}_seed{seed}.json'
                if not path.exists():return False
                plan = json.loads(path.read_text())
                if not (Path(plan['out'])/'completion.json').exists():return False
        return True
    def run(module):
        result = subprocess.run([sys.executable,'-m',module], cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(module+': '+(result.stdout+result.stderr)[-2500:])
        print(result.stdout, flush=True)
    try:
        while time.monotonic() < deadline:
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != digest:
                raise ValueError('frozen manifest changed')
            if not training_audited and ready():
                state['status']='auditing_all_nine_training_runs';save()
                run('theory_mvp.train04confirm.training_audit')
                training_audited=True
                state['training_report']=str(folder/'training_completion.json')
            if training_audited:
                state['status']='waiting_for_all_six_endpoint_audits'
                paths=[folder/f'exact_{digest[:16]}_{arm}_seed{seed}.json'
                       for seed in (53,71,89) for arm in ('independent','complementary')]
                if all(path.exists() for path in paths):
                    state['status']='revalidating_raw_endpoints_and_reporting';save()
                    run('theory_mvp.train04confirm.report')
                    state.update(status='all_training_and_endpoint_reports_complete',
                                 final_report=str(folder/'final_report.json'))
                    save();return
            save();time.sleep(30)
        state['status']='watch_timeout_no_resubmission';save()
    except Exception as error:
        state.update(status='needs_attention',error=str(error));save();raise


if __name__=='__main__':main()
