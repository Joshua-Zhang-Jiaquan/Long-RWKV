"""Stage and launch two post-development competence-control jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import campaign as C
import emit_jobs as E
from submit_e3 import ROOT, snapshot_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submit', action='store_true')
    args = ap.parse_args()
    files = snapshot_files() + [ROOT/'results/f2_execution_protocol.json']
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    stage = Path(E.G)/'long_rwkv_e3_controls'/digest[:16]
    prior = json.loads((ROOT/'results/e3_causal_submission_plan.json').read_text())
    qualification = Path(prior['out'])/'dev_qualification.json'
    if json.loads(qualification.read_text()).get('qualified') is not True:
        raise SystemExit('causal qualification required')
    if args.submit:
        for src in files:
            dest = stage/src.relative_to(ROOT)
            if dest.exists():
                if hashlib.sha256(dest.read_bytes()).hexdigest() != manifest[str(src.relative_to(ROOT))]:
                    raise SystemExit('immutable stage conflict')
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
        (stage/'snapshot.json').write_text(json.dumps(manifest, indent=2)+'\n')
    jobs = []
    for model in ['f2', 'r0']:
        out = Path(E.OUTPUTS)/f'e3_controls_{model}_{digest[:16]}'
        env = {'E3_ROOT':str(stage), 'E3_OUT':str(out), 'E3_MODEL':model,
               'E3_PROTOCOL':str(stage/'results/f2_execution_protocol.json'),
               'E3_QUALIFICATION':str(qualification)}
        body = E.make_body(f'lrwkv-e3-controls-{model}-{digest[:12]}',
            E.wrapped(env, str(stage/'qz/launch_e3_controls.sh')),
            'Post-development evidence-only and no-evidence controls; frozen decoder; 72 examples each', E.SPEC_8GPU)
        body.update(auto_fault_tolerance=False, fault_tolerance_max_retry=0)
        job = {'model':model, 'out':str(out), 'body':body}
        if args.submit:
            if C.already_submitted(body['name']):
                job['submission'] = {'state':'skipped'}
            elif C.allowance_headroom() < 8:
                job['submission'] = {'state':'deferred', 'reason':'64-GPU cap'}
            else:
                job['submission'] = C.submit_one(body, dry_run=False)
        jobs.append(job)
        if job.get('submission', {}).get('refusal_kind') == 'capacity':
            break
    plan = {'stage':str(stage), 'files':manifest, 'jobs':jobs}
    (ROOT/'results/e3_controls_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
