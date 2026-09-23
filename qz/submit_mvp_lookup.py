"""Review/submit one 8-H100 frozen-model pilot under the new 32-GPU cap."""
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
from lrwkv_evidence.mvp.lookup_pilot import CONTRACT, make_items
CAP = 32


def live_usage(jobs):
    """Queued jobs with zero assigned GPUs still reserve at least eight slots."""
    selected = [j for j in jobs if (j.get('name') or '').startswith('lrwkv-') and j.get('status') in C.LIVE_STATUSES]
    rows = [{'job_id': j.get('job_id'), 'name': j.get('name'), 'status': j['status'],
             'reserved_gpus': max(8, int(j.get('gpu_count') or 0))} for j in selected]
    return {'cap': CAP, 'live_reserved_gpus': sum(r['reserved_gpus'] for r in rows), 'jobs': rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submit', action='store_true')
    args = ap.parse_args(argv)
    # Include unchanged E3 dependencies, but place every copied file in a new MVP stage.
    files = [ROOT / 'lrwkv_evidence/__init__.py', *sorted((ROOT / 'lrwkv_evidence/e3').glob('*.py')),
             ROOT / 'lrwkv_evidence/mvp/__init__.py', ROOT / 'lrwkv_evidence/mvp/lookup_pilot.py',
             ROOT / 'qz/launch_mvp_lookup.sh']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    bundle = {'files': hashes, 'contract': CONTRACT, 'items': make_items(), 'gpu_cap': CAP}
    digest = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()[:16]
    stage = Path(E.G) / 'long_rwkv_mvp_stages' / digest
    out = Path(E.OUTPUTS) / f'mvp_natural_lookup_dev_{digest}'
    body = E.make_body(f'lrwkv-mvp-lookup-{digest[:12]}',
        E.wrapped({'MVP_ROOT': str(stage), 'MVP_OUT': str(out)}, str(stage / 'qz/launch_mvp_lookup.sh')),
        'Frozen F2/R0 natural lookup competence gate; 120 one-hop+120 two-hop per model; no pretraining', E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False, fault_tolerance_max_retry=0, max_running_time_ms="3600000")
    record = {'stage': str(stage), 'out': str(out), 'bundle': bundle, 'body': body,
              'requested_gpus': 8, 'scope': CONTRACT['scope']}
    result_path = ROOT / 'results/mvp_lookup_submission_plan.json'
    if args.submit:
        census = live_usage(C.list_all_jobs())
        record['pre_submit_capacity'] = census
        if census['live_reserved_gpus'] + 8 > CAP:
            result_path.write_text(json.dumps(record, indent=2) + '\n')
            raise SystemExit(f"32-GPU cap: {census['live_reserved_gpus']} already queued/running; need8")
        if C.already_submitted(body['name']):
            raise SystemExit('this exact MVP already submitted; inspect ledger before retry')
        for source in files:
            rel = source.relative_to(ROOT); dest = stage / rel
            if hashlib.sha256(source.read_bytes()).hexdigest() != hashes[str(rel)]:
                raise SystemExit('source changed after review; rerun dry review')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != hashes[str(rel)]:
                raise SystemExit('immutable MVP stage conflict')
            if not dest.exists():
                shutil.copyfile(source, dest)
        # Bank the exact gate and prompts before any job is submitted.
        (stage / 'competence_contract.json').write_text(json.dumps(CONTRACT, indent=2) + '\n')
        (stage / 'dev_items.json').write_text(json.dumps(make_items(), indent=2) + '\n')
        record['submission'] = C.submit_one(body, dry_run=False)
    result_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'bundle'}, indent=2))


if __name__ == '__main__':
    main()
