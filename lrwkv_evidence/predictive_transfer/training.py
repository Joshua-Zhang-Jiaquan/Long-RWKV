"""Fixed full-lineage curriculum; no held-out examples are accessible here."""
from . import tasks as T
from lrwkv_evidence.train04 import tasks as OLD
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.posterior_context_probe.tasks import serialize
SEEDS = (202709231, 202709232, 202709233, 202709234, 202709235, 202709236)
RECIPE = dict(world_size=8, global_batch=32, steps=3100, short_end=2500,
              curriculum={'1-300':'ordinary_n2', '301-700':'label_mixture_n4',
                          '701-2500':'source_codes_equal_basis_policy', '2501-3100':'source_codes_16k_balanced'},
              short_lr=3e-5, long_lr=1e-5, warmups=[50,20], weight_decay=.01,
              betas=[.9,.95], eps=1e-8, clip=1., precision='FP32 IEEE',
              initialization='public pretrained backbone; independent conditioning initialization and data seeds')


def canvas(ex, tok, policy, round_index, length=None, position='evidence_only', bits=None):
    bits = ex['bits'] if bits is None else bits
    first = T.public_bases(ex)[policy]
    group = first if round_index == 0 else [i for i in range(8) if i not in first]
    visible = {} if round_index == 0 else {i: bits[i] for i in first}
    masked = [i not in visible for i in range(8)]
    oracle = OLD.oracle_marginals(ex['matrix_rows'], ex['syndrome'], visible, 8)
    encoded = serialize(ex, tok, length, position)
    c = dict(ids=encoded['ids']+[tok.mask_id if masked[i] else tok.binary_ids[bits[i]] for i in range(8)],
             prefix=len(encoded['ids']), stage=8 if round_index==0 else 4, masked=masked,
             gold=bits, oracle=oracle, family=ex['family'], instance_id=ex['instance_id'],
             loss_mask=[i in group for i in range(8)], loss_scale=2/8, branch='policy',
             policy=policy, position=position, round_index=round_index)
    return B.decorate(ex, c, tok)


def records(step, rank, tok, seed, qualification=False):
    if not 1 <= step <= 3100 or not 0 <= rank < 8: raise ValueError('invalid update')
    if step <= 700:
        n=2 if step<=300 else 4
        examples=[C.example('train', seed, (step-1)*32+rank*4+j, n) for j in range(4)]
        return [B.decorate(ex, M.canvas(ex,tok,seed,branch='corruption' if n==2 else None),tok) for ex in examples]
    position=('far','middle','near')[(step-1+rank)%3] if step>2500 else 'evidence_only'
    length=16384 if step>2500 else None
    result=[]
    for j in range(2):
        index=(step-701)*16+rank*2+j
        ex=T.make_example('qualification' if qualification else 'train',seed,index)
        for policy in (0,1):
            rr=T.rng('history',seed,ex['instance_id'],policy)
            bits=rr.choice(T.support(ex)); round_index=rr.randrange(2)
            result.append(canvas(ex,tok,policy,round_index,length,position,bits))
    return result

batch=B.pack

def loss(logits,batch):return M.loss(logits,batch,'rao_blackwell')
