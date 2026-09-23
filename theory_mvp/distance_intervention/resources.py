"""Authoritative scheduler accounting and explicit utilization measurement windows."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'qz'))
import campaign as C
from confirm_capacity import live_usage


def utilization(out, training):
    path=out/'gpu_usage.csv'
    if not path.exists():return None
    rows=[]
    for raw in csv.DictReader(path.read_text().splitlines(),skipinitialspace=True):
        try:
            stamp=datetime.strptime(raw['timestamp'].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
            number=lambda key:float(raw[key].strip().split()[0])
            total=number('memory.total [MiB]')
            rows.append(dict(time=stamp,index=int(raw['index']),compute=number('utilization.gpu [%]'),
                             memory_busy=number('utilization.memory [%]'),allocated_percent=100*number('memory.used [MiB]')/total))
        except (KeyError,ValueError,ZeroDivisionError):continue
    if not rows:return None
    def summarize(selected):
        if not selected:return None
        return dict(samples=len(selected),gpus=sorted({r['index'] for r in selected}),
                    mean_gpu_busy_percent=statistics.mean(r['compute'] for r in selected),
                    mean_memory_used_percent=statistics.mean(r['allocated_percent'] for r in selected),
                    mean_memory_controller_busy_percent=statistics.mean(r['memory_busy'] for r in selected),
                    fraction_samples_gpu_busy_at_least80=statistics.mean(r['compute']>=80 for r in selected))
    result=dict(entire_observed_job=summarize(rows),definition='nvidia-smi GPU busy is not achieved FLOP utilization; memory used/total differs from memory-controller busy')
    provenance=out/'provenance.json'
    if training and provenance.exists():
        # Provenance is written immediately before the training-start barrier.
        # This is an explicitly approximate alignment, not per-step GPU tracing.
        origin=provenance.stat().st_mtime
        low=origin+training[0]['elapsed_seconds'];high=origin+training[-1]['elapsed_seconds']
        result['approximate_steady_training_after_first_update']=summarize([r for r in rows if low<=r['time']<=high])
        result['steady_window_unix_seconds']=[low,high]
        result['alignment']='provenance file mtime plus logged elapsed training time; excludes first update/JIT; approximate barrier alignment'
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results/distance_intervention/resource_audit.json');args=ap.parse_args()
    jobs=C.list_all_jobs();byid={j['job_id']:j for j in jobs};rows=[]
    plans=[*sorted((ROOT/'results/distance_intervention').glob('plan_*.json')),
           *sorted((ROOT/'results/distance_intervention/failed_attempts').glob('plan_*.json')),
           *sorted((ROOT/'results/distance_intervention_cost').glob('plan_*.json'))]
    seen=set()
    for path in plans:
        p=json.loads(path.read_text());receipt=p.get('submission')
        if not receipt:continue
        job_id=receipt['job_id']
        if job_id in seen:raise ValueError('duplicate job receipt')
        seen.add(job_id);job=byid.get(job_id)
        if job is None:raise ValueError('scheduler handle missing: '+job_id)
        out=Path(p['out']);trainpath=out/'train.jsonl';training=[]
        if trainpath.exists():
            training=[json.loads(line) for line in trainpath.read_text().splitlines()]
            if [r['step'] for r in training]!=list(range(1,len(training)+1)):raise ValueError('noncontiguous resource trace')
        gpus=max(8,int(job.get('gpu_count') or 0));runtime=int(job.get('running_time_ms') or 0)/1000
        row=dict(plan=str(path),job_id=job_id,name=job['name'],status=job['status'],project_id=job['project_id'],
                 gpus=gpus,scheduler_running_seconds=runtime,scheduler_running_gpu_hours=gpus*runtime/3600,
                 initial_launch_failed_before_model_load=path.parent.name=='failed_attempts',
                 logged_updates=len(training),logged_input_tokens=sum(r['global_input_tokens'] for r in training),
                 utilization=utilization(out,training))
        if training:row['logged_training_gpu_hours']=gpus*training[-1]['elapsed_seconds']/3600
        rows.append(row)
    result=dict(observed_at=C.timestamp(),all_jobs_terminal=all(r['status'] in C.TERMINAL_STATUS for r in rows),
                campaign_capacity=live_usage(jobs),jobs=rows,
                total_scheduler_running_gpu_hours=sum(r['scheduler_running_gpu_hours'] for r in rows),
                total_logged_training_input_tokens=sum(r['logged_input_tokens'] for r in rows),
                accounting='Includes qualification, original evaluation, all fixed adaptations/evaluations, matched cost and failed launch receipts; scheduler running time excludes queue reservation; logged training time excludes model initialization and teardown')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','campaign_capacity')},indent=2))
    for row in rows:
        if row['logged_updates']>6:
            util=(row['utilization'] or {}).get('approximate_steady_training_after_first_update')
            print(json.dumps(dict(name=row['name'],updates=row['logged_updates'],steady=util)))


if __name__=='__main__':main()
