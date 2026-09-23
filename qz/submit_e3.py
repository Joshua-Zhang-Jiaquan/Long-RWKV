"""Stage an immutable E3 calibration snapshot and submit one capped 8-H100 job."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import campaign as C
import emit_jobs as E

ROOT = Path(__file__).resolve().parents[1]


def snapshot_files():
    files = sorted((ROOT / 'lrwkv_evidence').rglob('*.py'))
    files += sorted((ROOT / 'results/lc_dev_panel').glob('*.json'))
    files += sorted((ROOT / 'qz').glob('launch_e3*.sh'))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--causal', action='store_true', help='qualify released RWKV-7 comparator')
    args = ap.parse_args()
    files = snapshot_files()
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    stage = Path(E.G) / 'long_rwkv_e3_stages' / digest[:16]
    model_tag = 'r0' if args.causal else 'f2'
    artifact_tag = 'e3_causal' if args.causal else 'e3'
    launcher = 'launch_e3_causal.sh' if args.causal else 'launch_e3.sh'
    out = Path(E.OUTPUTS) / ('e3_dev_' + model_tag + '_' + digest[:16])
    body = E.make_body('lrwkv-e3-dev-' + model_tag + '-' + digest[:12],
        E.wrapped({'E3_ROOT': str(stage), 'E3_OUT': str(out)},
                  str(stage / 'qz' / launcher)),
        model_tag + ' development-only decoder qualification, 72 examples; 8 H100; immutable source ' + digest,
        E.SPEC_8GPU)
    body['auto_fault_tolerance'] = False
    body['fault_tolerance_max_retry'] = 0
    # No automatic retries of an unqualified GPU implementation.
    plan = {'stage': str(stage), 'out': str(out), 'snapshot_sha256': digest,
            'files': manifest, 'body': body}
    (ROOT / ('results/' + artifact_tag + '_submission_plan.json')).write_text(json.dumps(plan, indent=2) + '\n')
    if not args.submit:
        print(json.dumps(plan, indent=2))
        return
    census = C.live_census()
    if census['live_gpus_submitted'] + 8 > C.OWN_LIVE_GPU_ALLOWANCE:
        raise SystemExit('8-GPU calibration exceeds the standing 64-GPU allowance')
    if C.already_submitted(body['name']):
        raise SystemExit('this snapshot already has a blocking submission')
    for src in files:
        dest = stage / src.relative_to(ROOT)
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest() != manifest[str(src.relative_to(ROOT))]:
                raise SystemExit('immutable staging conflict: ' + str(dest))
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    (stage / 'snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
    row = C.submit_one(body, dry_run=False)
    (ROOT / ('results/' + artifact_tag + '_submission_receipt.json')).write_text(json.dumps(row, indent=2) + '\n')
    if row['state'] != 'submitted':
        raise SystemExit('calibration submission failed; inspect receipt before retrying')


if __name__ == '__main__':
    main()
