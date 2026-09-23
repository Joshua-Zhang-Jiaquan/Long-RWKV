"""Explicit recovery of the inspected startup/stop race; never automatically retry."""
import copy
import hashlib
import json
from pathlib import Path
import campaign as C
from confirm_capacity import live_usage,route_body,PRIMARY
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/header_distance'
with campaign_submission_lock(ROOT):
    path=FOLDER/'plan_balanced_seed20271013.json'
    old=json.loads(path.read_text())
    assert old['submission']['job_id']=='job-ba9bbf91-a57a-4c34-bffd-475cb396bcd9'
    status,job=C.job_status(old['submission']['job_id'])
    assert status=='job_stopped' and int(job['running_time_ms'])==0
    files=sorted(Path(old['out']).iterdir())
    assert {p.name for p in files} <= {'torchrun.log','gpu_usage.csv'}
    log=(Path(old['out'])/'torchrun.log').read_text()
    assert log.count('potentially buggy FLA implementation')==8
    assert 'Traceback' not in log
    evidence=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    record=copy.deepcopy(old)
    record['out']=old['out']+'_primary1'
    assert not Path(record['out']).exists()
    body=record['body'];body['name']+='-primary1'
    assert body['command'].count(old['out'])==1
    body['command']=body['command'].replace(old['out'],record['out'])
    record['capacity']=live_usage(C.list_all_jobs())
    record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
    assert record['budget_project_id']==PRIMARY
    assert not C.already_submitted(body['name'])
    receipt=dict(at=C.timestamp(),reason='Manual operational replacement after startup/stop race; old attempt has zero scheduler runtime and no predictions.',old_job_id=old['submission']['job_id'],old_status=status,old_timeline=job['timeline'],old_files=evidence,new_out=record['out'],science_unchanged=True)
    dest=FOLDER/'replacement_balanced_seed20271013.json'
    with dest.open('x') as f:json.dump(receipt,f,indent=2)
    record['supersedes_stopped_job']=old['submission']['job_id']
    record['submission']=C.submit_one(body,dry_run=False)
    path.write_text(json.dumps(record,indent=2)+'\n')
    receipt['new_submission']=record['submission']
    dest.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
