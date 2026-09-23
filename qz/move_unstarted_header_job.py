"""One explicit queue-only reroute after a primary slot becomes free."""
import argparse
import copy
import json
from pathlib import Path
import time
import campaign as C
from confirm_capacity import live_usage,route_body,PRIMARY,EXTRA
from submission_lock import campaign_submission_lock
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/header_distance'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',required=True);args=parser.parse_args()
    if args.role not in ('near_seed20271013','balanced_seed20271013'):raise ValueError('not a declared extra-project selector')
    with campaign_submission_lock(ROOT):
        path=FOLDER/f'plan_{args.role}.json';old=json.loads(path.read_text())
        if old['budget_project_id']!=EXTRA:raise ValueError('job is not on extra project')
        census=live_usage(C.list_all_jobs())
        if census['project_reserved_gpus'][PRIMARY]+8>32:raise SystemExit('no primary slot yet; no mutation')
        job_id=old['submission']['job_id'];status,job=C.job_status(job_id)
        if status not in ('job_creating','job_pending') or job.get('timeline',{}).get('run') or Path(old['out']).exists():
            raise SystemExit('job has started or is unreadable; leave it untouched')
        archive=FOLDER/'queued_attempts'/f'plan_{args.role}_extra.json';archive.parent.mkdir(exist_ok=True)
        with archive.open('x') as stream:json.dump(old,stream,indent=2);stream.write('\n')
        event=dict(role=args.role,old_job_id=job_id,reason='shared extra-project quota prevents start; primary slot free',before=job.get('timeline'),at=C.timestamp())
        eventpath=FOLDER/f'reroute_{args.role}.json';eventpath.write_text(json.dumps(event,indent=2)+'\n')
        ok,message=C.stop_job(job_id)
        if not ok:raise RuntimeError('queue stop rejected: '+message)
        deadline=time.monotonic()+120
        while True:
            status,job=C.job_status(job_id)
            if status in C.TERMINAL_STATUS:break
            if time.monotonic()>deadline:raise RuntimeError('queue stop not confirmed; no resubmission')
            time.sleep(2)
        event['stopped_status']=status;event['stopped_timeline']=job.get('timeline');eventpath.write_text(json.dumps(event,indent=2)+'\n')
        if job.get('timeline',{}).get('run') or Path(old['out']).exists():raise RuntimeError('execution raced with stop; inspect before any replacement')
        record=copy.deepcopy(old);body=record['body'];body['name']+='-primary1'
        record['capacity']=live_usage(C.list_all_jobs())
        record['budget_project_id']=route_body(body,record['capacity'],preferred_project=PRIMARY)
        if record['budget_project_id']!=PRIMARY:raise RuntimeError('primary capacity changed; do not submit elsewhere')
        if C.already_submitted(body['name']):raise RuntimeError('duplicate replacement name')
        record['supersedes_unstarted_job']=job_id
        record['submission']=C.submit_one(body,dry_run=False)
        path.write_text(json.dumps(record,indent=2)+'\n')
        event['new_job_id']=record['submission']['job_id'];event['complete']=True;eventpath.write_text(json.dumps(event,indent=2)+'\n')
        print(json.dumps(event,indent=2))


if __name__=='__main__':main()
