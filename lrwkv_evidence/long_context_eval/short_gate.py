"""Validate the complete short gate while the independent long sweep continues."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .collect import validate_condition
from lrwkv_evidence.long_context_mvp import tasks as T


def collect(plan):
    if plan.get('cost_only') or plan.get('diagnostic_step200') or plan.get('comparator','rwkv')!='rwkv':
        raise ValueError('requires primary terminal RWKV audit')
    conditions=[];flags=[];hashes={};contracts=set()
    for rank in range(8):
        path=Path(plan['out'])/f'rank{rank}.jsonl'
        lines=path.read_bytes().splitlines(keepends=True)[:3]
        if len(lines)!=3 or not lines[-1].endswith(b'\n'):raise ValueError('short gate incomplete')
        p,row,gate=[json.loads(s) for s in lines]
        if (p['kind'],row['kind'],gate['kind'])!=('provenance','condition','gate'):
            raise ValueError('unexpected short-gate prefix')
        if p.get('cost_only') or p.get('diagnostic_step200') or p.get('checkpoint_step',600)!=600:
            raise ValueError('wrong checkpoint mode')
        if p['rank']!=rank or p['checkpoint_sha256']!=plan['checkpoint_sha256'] or p['split']!='dev' or p['data_seed']!=20270923 or p['task_version']!=T.VERSION:
            raise ValueError('short-gate provenance mismatch')
        for key,tol in [('within_repeat_max_logprob_diff',1e-5),('cross_rank_max_logprob_diff',1e-4)]:
            if not math.isfinite(p[key]) or not 0<=p[key]<=tol:raise ValueError('numerical short gate failed')
        if row['requested_context_tokens'] is not None or row['position']!='middle':raise ValueError('not evidence-only gate')
        validate_condition(row,rank)
        info=next(r for r in row['exact']['results'] if r['method']=='information_set')
        passed=info['joint_kl_nats']<.1 and info['exact_valid_mass']>.9
        if gate['local_pass']!=passed:raise ValueError('incorrect local gate')
        conditions.append(row);flags.append(gate);contracts.add(p['contract_sha256'])
        hashes[path.name]=hashlib.sha256(b''.join(lines)).hexdigest()
    passed=all(g['local_pass'] for g in flags)
    if len(contracts)!=1 or any(g['all_eight_conditions_pass']!=passed for g in flags):raise ValueError('mixed contracts or global gate')
    return dict(short_gate_complete=True,execution_complete=False,development_only=True,comparator='rwkv',
                competence_gate_passed=passed,checkpoint_sha256=plan['checkpoint_sha256'],contract_sha256=next(iter(contracts)),
                prefix_sha256=hashes,conditions=conditions,
                scope='Validated eight-condition short gate only; long-context endpoints and numerical qualification may still be running. No final result or long-context benefit is claimed.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();plan=json.loads(args.plan.read_text())
    from lrwkv_evidence.train04.worker import file_sha
    for rel,digest in plan['sources'].items():
        if file_sha(Path(plan['stage'])/rel)!=digest:raise ValueError('changed staged source')
    training=json.loads(Path(plan['training_plan']).read_text());out=Path(training['out'])
    done=json.loads((out/'completion.json').read_text())
    if done['step']!=600 or done['checkpoint_sha256']!=plan['checkpoint_sha256'] or file_sha(out/'resume.pt')!=plan['checkpoint_sha256']:
        raise ValueError('wrong completed checkpoint')
    report=collect(plan)
    if report['contract_sha256']!=done['contract_sha256']:raise ValueError('training contract mismatch')
    report.update(evaluation_plan=str(args.plan.resolve()),collector_source_sha256=file_sha(Path(__file__)))
    temp=args.out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(args.out)
    print(json.dumps(dict(report=str(args.out),short_gate_complete=True,competence_gate_passed=report['competence_gate_passed'])))


if __name__=='__main__':main()
