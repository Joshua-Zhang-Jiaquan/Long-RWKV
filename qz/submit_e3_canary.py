"""Submit one small eight-GPU exploratory generation sanity check."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import campaign as C
import emit_jobs as E
from submit_e3 import ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submit',action='store_true')
    args=ap.parse_args()
    files=[ROOT/"lrwkv_evidence/__init__.py", ROOT/"qz/launch_e3_canary.sh"]
    files += [ROOT/"lrwkv_evidence/e3"/name for name in ("__init__.py", "positive_canary.py", "gpu_runner.py", "causal_runner.py", "grid324.py", "dev_grid.py", "lane.py", "runner.py")]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_e3_canary'/digest
    out=Path(E.OUTPUTS)/f'e3_canary_{digest}'
    body=E.make_body(f'lrwkv-e3-canary-{digest[:12]}',
        E.wrapped({'E3_ROOT':str(stage),'E3_OUT':str(out)},str(stage/'qz/launch_e3_canary.sh')),
        'Exploratory positive-generation sanity checks; eight prompts/model; 32 generated tokens; never primary confirmation',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0)
    plan={'stage':str(stage),'out':str(out),'files':hashes,'body':body}
    if args.submit:
        if C.allowance_headroom()<8:
            raise SystemExit('wait for eight GPUs within the 64-GPU cap')
        if C.already_submitted(body['name']):
            raise SystemExit('canary already submitted')
        for src in files:
            dest=stage/src.relative_to(ROOT)
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=hashes[str(src.relative_to(ROOT))]:
                raise SystemExit('immutable canary stage conflict')
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(src,dest)
        plan['submission']=C.submit_one(body,dry_run=False)
    (ROOT/'results/e3_canary_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))


if __name__=='__main__':
    main()
