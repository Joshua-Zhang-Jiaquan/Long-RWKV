"""Submit prespecified length shards from already qualified immutable sources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import campaign as C
import emit_jobs as E
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lrwkv_evidence.e3 import gpu_runner as U


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--model', choices=['f2', 'r0', 'both'], default='both')
    ap.add_argument('--length', type=int, choices=[16384, 32768, 65536])
    args = ap.parse_args()
    protocol = ROOT / 'results/f2_execution_protocol.json'
    U.verify_confirmation_freeze(json.loads(protocol.read_text()))
    plans = {model: json.loads((ROOT / ('results/' + tag + '_submission_plan.json')).read_text())
             for model, tag in [('f2', 'e3'), ('r0', 'e3_causal')]}
    qualification = Path(plans['r0']['out']) / 'dev_qualification.json'
    q = json.loads(qualification.read_text())
    if q.get('qualified') is not True or q.get('n') != 72:
        raise SystemExit('released comparator development qualification is incomplete')
    source = ROOT / 'qz/launch_e3_confirm.sh'
    bundle = {'launcher_sha256': U.file_sha(source),
              'protocol_sha256': U.file_sha(protocol),
              'qualification_sha256': U.file_sha(qualification),
              'source_snapshots': {m: p['snapshot_sha256'] for m, p in plans.items()}}
    identifier = U.digest(bundle)[:16]
    stage = Path(E.G) / 'long_rwkv_e3_confirmation' / identifier
    if args.submit:
        stage.mkdir(parents=True, exist_ok=True)
        for src in [source, protocol]:
            dest = stage / src.name
            if dest.exists() and U.file_sha(dest) != U.file_sha(src):
                raise SystemExit('immutable confirmation bundle conflict')
            if not dest.exists():
                shutil.copyfile(src, dest)
        U.atomic_json(stage / 'bundle.json', bundle)
    jobs = []
    for model in ['f2', 'r0'] if args.model == 'both' else [args.model]:
        for length in [args.length] if args.length else [16384, 32768, 65536]:
            out = Path(E.OUTPUTS) / f'e3_confirm_{model}_L{length}_{identifier}'
            env = {'E3_ROOT': plans[model]['stage'], 'E3_OUT': str(out),
                   'E3_MODEL': model, 'E3_LENGTH': str(length),
                   'E3_PROTOCOL': str(stage / protocol.name),
                   'E3_QUALIFICATION': str(qualification)}
            body = E.make_body(f'lrwkv-e3-confirm-{model}-l{length}-{identifier[:10]}',
                E.wrapped(env, str(stage / source.name)),
                f'Frozen {model} full-canvas confirmation at {length}; all generated cells, 200 examples/cell',
                E.SPEC_8GPU)
            body.update(auto_fault_tolerance=False, fault_tolerance_max_retry=0)
            job = {'model': model, 'length': length, 'out': str(out), 'body': body}
            if args.submit:
                if C.already_submitted(body['name']):
                    job['submission'] = {'state': 'skipped', 'reason': 'existing ledger submission'}
                elif C.allowance_headroom() < 8:
                    job['submission'] = {'state': 'deferred', 'reason': '64-GPU ceiling'}
                else:
                    job['submission'] = C.submit_one(body, dry_run=False)
                    if job['submission'].get('refusal_kind') == 'capacity':
                        jobs.append(job)
                        break
            jobs.append(job)
        if jobs and jobs[-1].get('submission', {}).get('refusal_kind') == 'capacity':
            break
    U.atomic_json(ROOT / 'results/e3_confirmation_plan.json',
                  {'bundle': bundle, 'stage': str(stage), 'jobs': jobs})
    print(json.dumps(jobs, indent=2))


if __name__ == '__main__':
    main()
