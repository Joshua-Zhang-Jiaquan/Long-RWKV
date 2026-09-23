"""Numerical screen at the actual long-context lengths, before endpoint sweeps."""
import math
import torch
import torch.distributed as dist


def specifications(comparator='rwkv'):
    histories = ('cached_path',) if comparator == 'causal_rwkv' else ('all_masked', 'first_four_visible')
    return [dict(context_tokens=n,position=p,history=h) for n in (1024,4096,16384)
            for p in ('far','near') for h in histories]


def validate(rows, comparator='rwkv'):
    if [{k:r[k] for k in ('context_tokens','position','history')} for r in rows] != specifications(comparator):
        raise ValueError('incomplete long-context numerical screen')
    for row in rows:
        a=row['within_repeat_max_logprob_diff'];b=row['cross_rank_max_logprob_diff']
        if not all(math.isfinite(v) and v>=0 for v in (a,b)) or a>1e-5 or b>1e-4:
            raise ValueError('long-context numerical screen failed')


@torch.inference_mode()
def run(predict, comparator='rwkv'):
    rows=[]
    for spec in specifications(comparator):
        repeat=torch.stack([predict(spec) for _ in range(3)])
        within=float((repeat-repeat[0]).abs().max())
        values=[torch.empty_like(repeat[0]) for _ in range(dist.get_world_size())]
        dist.all_gather(values,repeat[0])
        cross=max(float((v-values[0]).abs().max()) for v in values)
        row=dict(spec,within_repeat_max_logprob_diff=within,cross_rank_max_logprob_diff=cross)
        rows.append(row)
        # All ranks participate in the same collectives before checking errors.
        failed=torch.tensor(int(not math.isfinite(within) or not math.isfinite(cross) or within>1e-5 or cross>1e-4),device=repeat.device)
        dist.all_reduce(failed,op=dist.ReduceOp.MAX)
        if bool(failed):raise ValueError('long-context numerical failure: '+str(row))
    validate(rows,comparator)
    return rows
