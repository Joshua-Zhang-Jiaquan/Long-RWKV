"""Scheduler accounting and reservation-interval audit for this follow-up only."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'qz'))
import campaign as C
from confirm_capacity import live_usage, PROJECT_CAPS
from theory_mvp.distance_intervention.resources import utilization


def main():
    folder=ROOT/'results/history_response';rows=[];events=[]
    for path in sorted(folder.glob('plan_*.json')):
        p=json.loads(path.read_text());receipt=p.get('submission')
        if not receipt:raise ValueError('unsubmitted selector')
        status,job=C.job_status(receipt['job_id'])
        if status not in C.TERMINAL_STATUS:raise ValueError('nonterminal or unreadable job')
        project=job['project_id'];gpus=max(8,int(job.get('gpu_count') or 0))
        if project not in PROJECT_CAPS or gpus!=8:raise ValueError('unexpected resource allocation')
        timeline=job['timeline'];created=int(timeline['created']);finished=int(timeline['finished'])
        if not 0<created<finished:raise ValueError('invalid scheduler interval')
        events.extend([(created,gpus,project),(finished,-gpus,project)])
        runtime=int(job['running_time_ms'])/1000
        rows.append(dict(role=p['role'],job_id=receipt['job_id'],status=status,project_id=project,
                         gpus=gpus,created_ms=created,finished_ms=finished,
                         scheduler_running_seconds=runtime,scheduler_running_gpu_hours=gpus*runtime/3600,
                         utilization=utilization(Path(p['out']),[])))
    if len(rows)!=7 or len({r['job_id'] for r in rows})!=7:raise ValueError('wrong job set')
    usage=dict.fromkeys(PROJECT_CAPS,0);peaks=dict(usage);totalpeak=0
    for stamp,delta,project in sorted(events):
        usage[project]+=delta
        if not 0<=usage[project]<=PROJECT_CAPS[project] or sum(usage.values())>48:
            raise ValueError('reservation cap violated')
        peaks[project]=max(peaks[project],usage[project]);totalpeak=max(totalpeak,sum(usage.values()))
    if any(usage.values()):raise ValueError('unterminated intervals')
    census=live_usage(C.list_all_jobs())
    result=dict(observed_at=C.timestamp(),all_jobs_terminal=True,jobs=rows,
                total_scheduler_running_gpu_hours=sum(r['scheduler_running_gpu_hours'] for r in rows),
                reservation_peak=totalpeak,project_reservation_peaks=peaks,live_campaign=census,
                scope='Seven evaluation jobs only; excludes previous study and prior checkpoint training. Intervals include queued reservation. Utilization covers the entire observed job, including loading and numerical screens.')
    (folder/'resource_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','live_campaign')},indent=2))


if __name__=='__main__':main()
