"""Matched 16K distance intervention, with balanced two-policy supervision."""
from lrwkv_evidence.train04 import tasks
from lrwkv_evidence.train04paired import core as P
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.posterior_context_probe.tasks import serialize, partitions

RECIPE = dict(version=1, context_tokens=16384, world_size=8, global_batch=32,
              learning_rate=1e-5, warmup=20, weight_decay=.01, clip=1.,
              objective='equal information-set and intact-pair policies; uniform round; exact conditional soft CE',
              microbatch_documents=1, accumulation_steps=4,
              precision='FP32; torch TF32 off; Triton IEEE',
              initialization='selected seed71 independent2500; fresh optimizer',
              index_offset=2000000)
ARMS = ('near', 'balanced')
SEEDS = (20271011, 20271012, 20271013)


def records(step, rank, tok, arm, seed):
    if step < 1 or not 0 <= rank < 8 or arm not in ARMS:
        raise ValueError('invalid matched training request')
    result = []
    position = 'near' if arm == 'near' else ('far', 'middle', 'near')[(step - 1 + rank) % 3]
    for j in range(2):
        index = RECIPE['index_offset'] + (step - 1) * 16 + rank * 2 + j
        ex = tasks.make_example('train', seed, index)
        encoded = serialize(ex, tok, 16384, position)
        round_index = P.randomizer('distance_round_v1', seed, ex['instance_id']).randrange(2)
        for member, policy in zip(P.history_pair(ex, seed, 'independent'),
                                  ('information_set', 'pair_preserving_halves')):
            groups = partitions(member)[policy]
            visible = {} if round_index == 0 else {i: member['bits'][i] for i in groups[0]}
            masked = [i not in visible for i in range(8)]
            oracle = tasks.oracle_marginals(member['matrix_rows'], member['syndrome'], visible, 8)
            c = dict(ids=encoded['ids'] + [tok.mask_id if masked[i] else tok.binary_ids[visible[i]] for i in range(8)],
                     prefix=len(encoded['ids']), stage=8 if round_index == 0 else 4,
                     masked=masked, gold=member['bits'], oracle=oracle,
                     family=member['family'], instance_id=member['instance_id'],
                     loss_mask=[i in groups[round_index] for i in range(8)],
                     loss_scale=2/8, branch='policy', policy=policy,
                     position=position, round_index=round_index)
            result.append(B.decorate(member, c, tok))
    return result


def batch(records, device):
    return B.pack(records, device)


def loss(logits, batch):
    return M.loss(logits, batch, 'rao_blackwell')
