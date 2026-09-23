"""Subtract the exact soft-target entropy floor from development training CE.

This is an in-sample optimization diagnostic, not an endpoint competence test.
It calls the pinned corruption/target code, skipping only prompt serialization.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from unittest.mock import patch
from lrwkv_evidence.long_context_mvp import training as R, tasks as T


class ShapeTokenizer:
    binary_ids=(0,1)
    mask_id=2


def entropy(p):
    if not 0 <= p <= 1:raise ValueError('invalid target probability')
    return -(p*math.log(p) if p else 0.)-((1-p)*math.log1p(-p) if p<1 else 0.)


def floor(data_step):
    # Serialization never feeds the RNG or target solver. Mocking it in this
    # CPU process saves generating thousands of irrelevant distractor rows.
    values=[]
    with patch.object(T,'serialize',return_value={'prefix_ids':[]}):
        for rank in range(8):
            for record in R.records(data_step,rank,ShapeTokenizer()):
                values.append(record['loss_scale']*sum(entropy(p) for p,m in zip(record['oracle'],record['loss_mask']) if m))
    return statistics.mean(values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    if plan['phase']!='development600' or plan.get('kind') is not None:
        raise ValueError('requires main RWKV development recipe')
    for module in (R,T):
        p=Path(module.__file__).resolve();rel=str(p.relative_to(Path(__file__).resolve().parents[2]))
        if hashlib.sha256(p.read_bytes()).hexdigest()!=plan['sources'][rel]:raise ValueError('source differs from trained recipe')
    source=Path(plan['out'])/'train.jsonl';raw=source.read_bytes()
    lines=raw.splitlines();rows=[json.loads(s) for s in lines]
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('noncontiguous training log')
    for row in rows:
        minimum=floor(row['data_step']);row['oracle_ce_floor']=minimum
        row['excess_training_ce']=row['loss']-minimum
        if not math.isfinite(row['excess_training_ce']) or row['excess_training_ce'] < -1e-6:
            raise ValueError('training CE below its oracle entropy floor')
    summaries=[]
    for name,items in [('evidence_only',[r for r in rows if r['data_step']<=200]),
                       ('context_1K',[r for r in rows if r['data_step']>200]),('last50',rows[-50:])]:
        if items:
            summaries.append(dict(interval=name,updates=len(items),first_step=items[0]['step'],last_step=items[-1]['step'],
                                  **{k:statistics.mean(r[k] for r in items) for k in ('loss','oracle_ce_floor','excess_training_ce')}))
    report=dict(training_plan=str(args.plan.resolve()),observed_log_sha256=hashlib.sha256(raw).hexdigest(),
                observed_steps=len(rows),summaries=summaries,rows=rows,
                limitation='In-sample pre-update loss on each update batch, not held-out competence, sampler KL, or a long-context gain.')
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
