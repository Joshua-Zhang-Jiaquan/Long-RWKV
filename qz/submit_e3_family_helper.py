"""Move the F2 dataflow cells to a separately audited eight-GPU worker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import campaign as C
import emit_jobs as E

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lrwkv_evidence.e3 import grid324 as G
from lrwkv_evidence.e3.helper_family import select_family


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--length',type=int,choices=[32768,65536],required=True)
    ap.add_argument('--submit',action='store_true')
    args=ap.parse_args()
    plan_path=ROOT/'results/e3_confirmation_plan.json'
    plan=json.loads(plan_path.read_text())
    job=next(j for j in plan['jobs'] if j['model']=='f2' and j['length']==args.length)
    cells,selected=select_family(G.all_cells(),lambda c:json.loads((G.OUT_ROOT/f'{c.cell_id}.json').read_text()),
                                 length=args.length,family='code_dataflow')
    selected_ids={c.cell_id for c in selected}
    files=[ROOT/'lrwkv_evidence/e3/helper_family.py',ROOT/'qz/launch_e3_family_helper.sh']
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest=hashlib.sha256(json.dumps({'files':hashes,'bundle':plan['bundle'],'length':args.length},sort_keys=True).encode()).hexdigest()[:16]
    stage=Path(E.G)/'long_rwkv_e3_family_helpers'/digest
    out=Path(E.OUTPUTS)/f'e3_family_f2_L{args.length}_{digest}'
    dev=json.loads((ROOT/'results/e3_submission_plan.json').read_text())
    env={'E3_ROOT':dev['stage'],'E3_FAMILY_HELPER_CODE':str(stage/'helper_family.py'),
         'E3_PRIMARY_OUT':job['out'],'E3_OUT':str(out),'E3_LENGTH':str(args.length),
         'E3_PROTOCOL':str(Path(plan['stage'])/'f2_execution_protocol.json')}
    body=E.make_body(f'lrwkv-e3-family-f2-l{args.length}-{digest[:10]}',
        E.wrapped(env,str(stage/'launch_e3_family_helper.sh')),
        'Fixed F2 dataflow partition, 27 cells/5400 examples; unchanged qualified inference',E.SPEC_8GPU)
    body.update(auto_fault_tolerance=False,fault_tolerance_max_retry=0)
    record={'length':args.length,'stage':str(stage),'out':str(out),'files':hashes,'body':body}
    if args.submit:
        if C.allowance_headroom()<8:
            raise SystemExit('eight free GPUs required within the 64-GPU cap')
        if C.already_submitted(body['name']):
            raise SystemExit('family helper already submitted; inspect its existing receipt')
        stage.mkdir(parents=True,exist_ok=True)
        for src in files:
            dest=stage/src.name
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=hashes[src.name]:
                raise SystemExit('immutable family helper stage conflict')
            if not dest.exists():
                shutil.copyfile(src,dest)
        record['submission']=C.submit_one(body,dry_run=False)
        if record['submission']['state']=='submitted':
            parts=job.get('partitions',[{'out':job['out'],'cell_ids':[c.cell_id for c in cells]}])
            for part in parts:
                old=set(part['cell_ids'])
                part['cell_ids']=sorted(old-selected_ids)
                if not part['cell_ids']:
                    raise SystemExit('unexpected duplicate family-helper partition')
                if old!=set(part['cell_ids']) and part['out']!=job['out']:
                    part['schedule_policy']='accepted_subset_of_recorded_schedule'
            parts.append({'out':str(out),'cell_ids':sorted(selected_ids),'helper_wrapper_sha256':hashes['helper_family.py']})
            job['partitions']=parts
            job['family_helper_submission']=record['submission']
            plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    (ROOT/f'results/e3_family_helper_{args.length}_plan.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
