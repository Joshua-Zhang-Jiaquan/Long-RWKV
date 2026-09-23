"""Development recipe for the long-context task; no test split access."""
import torch
from . import tasks as T
from lrwkv_evidence.train04dev import core as C

RECIPE = dict(version=1,seed=17,initial_seed=0,world_size=8,global_batch=32,
              terminal_step=600,curriculum={'1-200':'evidence_only','201-600':1024},
              objective='equal mixture of ordinary absorbing and two-round policy soft CE',
              learning_rate=3e-5,warmup=50,weight_decay=.01,clip=1.,
              precision='FP32; torch TF32 off; Triton default',
              scope='development only; no positive result presumed; qualify before training')


def records(step, rank, tokenizer):
    if not 1 <= step <= 600 or not 0 <= rank < 8:
        raise ValueError('outside development recipe')
    result=[]
    for j in range(4):
        index=(step-1)*32+rank*4+j
        ex=T.problem('train',17,index,T.FAMILIES[index%2])
        length=None if step<=200 else 1024
        serial=T.serialize(ex,tokenizer,length,('near','middle','far')[index%3])
        r=T.rng('corruption',17,index)
        if (index//2)%2:
            second=bool(r.randrange(2));visible={i:ex['bits'][i] for i in range(4)} if second else {}
            c=T.canvas(ex,serial,tokenizer,visible,4 if second else 8)
            c['loss_mask']=[i>=4 if second else i<4 for i in range(8)]
            c['loss_scale']=2/8
        else:
            stage=r.randrange(1,9);visible={i:b for i,b in enumerate(ex['bits']) if r.random()>=stage/8}
            c=T.canvas(ex,serial,tokenizer,visible,stage)
            c['loss_mask']=c['masked'];c['loss_scale']=8/(stage*8)
        result.append(c)
    return result


def batch(records,device):
    b=C.pack(records,device)
    b['loss_mask']=torch.tensor([c['loss_mask'] for c in records],device=device,dtype=torch.bool)
    b['loss_scale']=torch.tensor([c['loss_scale'] for c in records],device=device,dtype=torch.float32)
    return b


def loss(logits,batch):
    lp=logits.float().log_softmax(-1);p=batch['oracle']
    ce=-(p*lp[...,1]+(1-p)*lp[...,0])
    return ((ce*batch['loss_mask']).sum(-1)*batch['loss_scale']).mean()
