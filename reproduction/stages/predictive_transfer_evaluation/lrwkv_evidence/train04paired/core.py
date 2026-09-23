"""Matched policy-history batches with independent or complementary free bits."""
import hashlib
import random
from lrwkv_evidence.train04 import tasks
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.train04binding import core as B


def randomizer(*parts):
    return random.Random(int(hashlib.sha256('/'.join(map(str,parts)).encode()).hexdigest(),16))


def with_free_values(example,values):
    free=example['information_set'];n=example['n']
    if len(values)!=len(free) or any(v not in (0,1) for v in values):raise ValueError('invalid free bits')
    solved=tasks.solve_affine(example['matrix_rows']+[1<<i for i in free],example['syndrome']+list(values),n)
    if solved is None or solved[1]:raise ValueError('information set must identify one feasible vector')
    vector=solved[0]
    return {**example,'bits':[(vector>>i)&1 for i in range(n)]}


def history_pair(example,seed,mode):
    if mode not in ('independent','complementary'):raise ValueError(mode)
    rng=randomizer('paired_history_v1',seed,example['instance_id'])
    first=[rng.randrange(2) for _ in example['information_set']]
    second=[rng.randrange(2) for _ in first] if mode=='independent' else [1-v for v in first]
    return [with_free_values(example,v) for v in (first,second)]


def training_canvases(update,rank,tokenizer,mode,seed=17):
    # Two public conditions per rank, one per family; two histories per condition.
    result=[]
    for j in range(2):
        index=1500*32+(update-1)*16+rank*2+j
        ex=C.example('train',seed,index,8)
        round_index=randomizer('paired_round_v1',seed,ex['instance_id']).randrange(2)
        for member in history_pair(ex,seed,mode):
            canvas=M.canvas(member,tokenizer,seed,branch='policy',round_index=round_index)
            result.append(B.decorate(member,canvas,tokenizer))
    return result
