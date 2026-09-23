"""Keep numerical/request-cost evidence separate from learned endpoint evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from .exact import METHODS,partitions
from .numerical import validate as validate_numerical
from lrwkv_evidence.long_context_mvp import tasks as T


def collect(plan):
    if not plan.get('cost_only'):raise ValueError('requires cost-only receipt')
    kind=plan['comparator']
    if kind not in ('rwkv','attention','causal_rwkv'):raise ValueError('unknown model')
    expected=[(None,'middle')]+[(h,p) for h in (1024,4096,16384) for p in ('far','middle','near')]
    conditions=[];provenances=[];hashes={}
    for rank in range(8):
        path=Path(plan['out'])/f'rank{rank}.jsonl';raw=path.read_bytes()
        hashes[path.name]=hashlib.sha256(raw).hexdigest();rows=[json.loads(s) for s in raw.splitlines()]
        if [r['kind'] for r in rows]!=['provenance','cost_condition','numerical_qualification']+['cost_condition']*9+['complete']:
            raise ValueError('incomplete or unexpected cost shard')
        p=rows[0]
        if not p['cost_only'] or not rows[-1]['cost_only'] or p['rank']!=rank or p.get('comparator','rwkv')!=kind or p['timing_repeats']!=10:
            raise ValueError('wrong cost provenance')
        if any(p[k]!=plan[k] for k in ('checkpoint_sha256','checkpoint_step','contract_sha256')):
            raise ValueError('checkpoint identity mismatch')
        if p['split']!='dev' or p['data_seed']!=20270923 or p['task_version']!=T.VERSION:
            raise ValueError('unexpected task inputs')
        for key,tol in [('within_repeat_max_logprob_diff',1e-5),('cross_rank_max_logprob_diff',1e-4)]:
            if not math.isfinite(p[key]) or not 0<=p[key]<=tol:raise ValueError('short numerical qualification failed')
        if not p['runtime']['gpu_name'] or p['runtime']['gpu_total_memory_bytes']<=0:raise ValueError('hardware missing')
        validate_numerical(rows[2]['checks'],kind)
        items=[r for r in rows if r['kind']=='cost_condition']
        if [(r['requested_context_tokens'],r['position']) for r in items]!=expected:raise ValueError('incomplete length/position grid')
        ex=T.problem('dev',20270923,rank//2,T.FAMILIES[rank%2])
        methods=['cached_causal'] if kind=='causal_rwkv' else list(METHODS)
        for r in items:
            if r['family']!=ex['family'] or r['index']!=ex['index'] or r['exact'] is not None:
                raise ValueError('cost-only study must not contain answer-quality endpoints')
            if r['requested_context_tokens'] is not None and r['context_tokens']!=r['requested_context_tokens']:
                raise ValueError('wrong context length')
            lo,hi=r['evidence_span']
            if not 0<=lo<hi<=r['context_tokens'] or r['records']<1 or r['filler_tokens']<0:
                raise ValueError('invalid serialized input')
            if [c['method'] for c in r['costs']]!=methods:raise ValueError('missing cost policy')
            if not math.isfinite(r['condition_wall_seconds']) or r['condition_wall_seconds']<=0:raise ValueError('invalid runtime')
            for c in r['costs']:
                calls=8 if kind=='causal_rwkv' else len(partitions(ex)[c['method']])
                if c['calls']!=calls or len(c['seconds'])!=10 or any(not math.isfinite(t) or t<=0 for t in c['seconds']):
                    raise ValueError('invalid request timings')
                if c['peak_allocated_bytes']<=0 or len(c['sample_outputs'])!=10 or any(type(y)is not int or not 0<=y<256 for y in c['sample_outputs']):
                    raise ValueError('invalid request metadata')
        conditions.extend(items);provenances.append(p)
    summaries=[]
    for h,pos in expected:
        items=[r for r in conditions if (r['requested_context_tokens'],r['position'])==(h,pos)]
        for method in methods:
            costs=[next(c for c in r['costs'] if c['method']==method) for r in items]
            means=[statistics.mean(c['seconds']) for c in costs]
            summaries.append(dict(context_tokens=h,position=pos,method=method,conditions=8,repeats_per_condition=10,
                                  mean_request_seconds=statistics.mean(means),condition_mean_range=[min(means),max(means)],
                                  max_peak_allocated_bytes=max(c['peak_allocated_bytes'] for c in costs)))
    return dict(execution_complete=True,cost_only=True,comparator=kind,checkpoint_sha256=plan['checkpoint_sha256'],
                checkpoint_step=plan['checkpoint_step'],condition_count=len(conditions),aggregates=summaries,
                conditions=conditions,provenances=provenances,shard_sha256=hashes,
                limitation='Development request-cost screen only. No learned accuracy or joint cost-quality certificate. Eight H100 ranks with ten draws per condition; draws are clustered, not independent quality examples.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();plan=json.loads(args.plan.read_text())
    from lrwkv_evidence.train04.worker import file_sha
    for rel,h in plan['sources'].items():
        if file_sha(Path(plan['stage'])/rel)!=h:raise ValueError('changed staged source')
    training=json.loads(Path(plan['training_plan']).read_text());done=json.loads((Path(training['out'])/'completion.json').read_text())
    if any(done[k]!=plan[k] for k in ('checkpoint_sha256','contract_sha256')) or done['step']!=plan['checkpoint_step']:
        raise ValueError('training receipt mismatch')
    if file_sha(Path(training['out'])/'resume.pt')!=plan['checkpoint_sha256']:raise ValueError('checkpoint changed')
    report=collect(plan);report.update(evaluation_plan=str(args.plan.resolve()),collector_source_sha256=file_sha(Path(__file__)))
    temp=args.out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(args.out)
    print(json.dumps(dict(report=str(args.out),cost_only=True,conditions=report['condition_count'])))


if __name__=='__main__':main()
