"""Add separately stored 64K suffix workers without changing inference kernels."""
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
from lrwkv_evidence.e3 import grid324 as G
from lrwkv_evidence.e3.helper_suffix import select_partition


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submit', action='store_true')
    args = ap.parse_args()
    plan_path = ROOT/'results/e3_confirmation_plan.json'
    plan = json.loads(plan_path.read_text())
    primary, helper = select_partition(G.all_cells(), lambda c: json.loads((G.OUT_ROOT/f'{c.cell_id}.json').read_text()))
    files = [ROOT/'lrwkv_evidence/e3/helper_suffix.py', ROOT/'qz/launch_e3_helper.sh']
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest = hashlib.sha256(json.dumps({'files':hashes, 'confirmation_bundle':plan['bundle']},sort_keys=True).encode()).hexdigest()[:16]
    stage = Path(E.G)/'long_rwkv_e3_helpers'/digest
    dev_plans = {m:json.loads((ROOT/f'results/{tag}_submission_plan.json').read_text())
                 for m,tag in [('f2','e3'),('r0','e3_causal')]}
    if args.submit:
        if C.allowance_headroom() < 16:
            raise SystemExit('Both helpers require 16 free GPUs within the 64-GPU cap; wait for controls to finish.')
        stage.mkdir(parents=True,exist_ok=True)
        for src in files:
            dest = stage/src.name
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=hashes[src.name]:
                raise SystemExit('immutable helper stage conflict')
            if not dest.exists():
                shutil.copyfile(src,dest)
    jobs=[]
    for model in ['f2','r0']:
        original=next(j for j in plan['jobs'] if j['model']==model and j['length']==65536)
        out=Path(E.OUTPUTS)/f'e3_helper_{model}_L65536_{digest}'
        env={'E3_ROOT':dev_plans[model]['stage'], 'E3_HELPER_CODE':str(stage/'helper_suffix.py'),
             'E3_MODEL':model, 'E3_OUT':str(out), 'E3_PRIMARY_OUT':original['out'],
             'E3_PROTOCOL':str(Path(plan['stage'])/'f2_execution_protocol.json'),
             'E3_QUALIFICATION':str(Path(dev_plans['r0']['out'])/'dev_qualification.json')}
        body=E.make_body(f'lrwkv-e3-helper-{model}-{digest[:12]}',
                        E.wrapped(env,str(stage/'launch_e3_helper.sh')),
                        '64K fixed suffix worker: 54 cells, 10800 examples; unchanged qualified inference kernels',E.SPEC_8GPU)
        body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0)
        job={'model':model,'length':65536,'out':str(out),'body':body}
        if args.submit:
            if C.already_submitted(body['name']):
                job['submission']={'state':'skipped'}
            elif C.allowance_headroom()<8:
                job['submission']={'state':'deferred'}
            else:
                job['submission']=C.submit_one(body,dry_run=False)
            if job['submission']['state'] in ('submitted','skipped'):
                original['partitions']=[
                    {'out':original['out'],'cell_ids':[c.cell_id for c in primary]},
                    {'out':str(out),'cell_ids':[c.cell_id for c in helper],
                     'helper_wrapper_sha256':hashes['helper_suffix.py']}]
                original['helper_submission']=job['submission']
                plan_path.write_text(json.dumps(plan,indent=2)+'\n')
        jobs.append(job)
        if job.get('submission',{}).get('refusal_kind')=='capacity':
            break
    (ROOT/'results/e3_helpers_plan.json').write_text(json.dumps({'stage':str(stage),'files':hashes,'jobs':jobs},indent=2)+'\n')
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':
    main()
