"""Seed-parameterized candidate replication of the complete paired-history recipe.

This training-only module does not select the recipe or access confirmation
conditions. Seed17 is reserved for development equivalence qualification;
53/71/89 are the prospective fresh training replicates.
"""
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M
from lrwkv_evidence.train04binding import core as B
from lrwkv_evidence.train04paired import core as P

SEEDS=(53,71,89)


def phase(step):
    if not isinstance(step,int) or not 1<=step<=2500:raise ValueError('step outside fixed curriculum')
    if step<=300:return 'ordinary_n2'
    if step<=700:return 'label_mixture_n4'
    if step<=1500:return 'label_mixture_n8'
    return 'paired_n8'


def canvases(step,rank,tokenizer,seed,mode):
    current=phase(step)
    if seed not in (*SEEDS,17):raise ValueError('unregistered data seed')
    if not 0<=rank<8:raise ValueError('requires eight ranks')
    if mode not in ('independent','complementary'):raise ValueError('unknown history coupling')
    if current=='paired_n8':return P.training_canvases(step-1500,rank,tokenizer,mode,seed)
    n=C.curriculum_n(step)
    examples=[C.example('train',seed,(step-1)*32+rank*4+j,n) for j in range(4)]
    if current=='ordinary_n2':return [C.make_canvas(ex,tokenizer,seed) for ex in examples]
    return [B.decorate(ex,M.canvas(ex,tokenizer,seed),tokenizer) for ex in examples]


def batch(step,rank,tokenizer,seed,mode,device):
    records=canvases(step,rank,tokenizer,seed,mode)
    return (C.pack if step<=300 else B.pack)(records,device)


def loss(step,logits,packed):
    return (C.loss if step<=300 else M.loss)(logits,packed,'rao_blackwell')
