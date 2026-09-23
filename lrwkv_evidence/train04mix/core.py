"""Half corruption, half information-set reveal-history supervision."""
import hashlib
import random
import torch
from lrwkv_evidence.train04 import tasks
from lrwkv_evidence.train04.evaluate import fixed_groups
from lrwkv_evidence.train04dev import core as C


def canvas(ex,tok,seed,branch=None,round_index=None):
    rng=random.Random(int(hashlib.sha256(f"policy_mixture_v1/{seed}/{ex['instance_id']}".encode()).hexdigest(),16))
    branch=('policy' if rng.random()<.5 else 'corruption') if branch is None else branch
    n=ex['n']
    if branch=='corruption':
        c=C.make_canvas(ex,tok,seed);c.update(loss_mask=c['masked'],loss_scale=8/(c['stage']*n),branch=branch)
        return c
    if branch!='policy':raise ValueError(branch)
    k=rng.randrange(2) if round_index is None else round_index
    if k not in (0,1):raise ValueError('two policy rounds')
    groups=fixed_groups(ex,'information_set');prior=[] if k==0 else groups[0]
    visible={i:ex['bits'][i] for i in prior};masked=[i not in visible for i in range(n)]
    stage=8 if k==0 else 4;prefix=tok.encode(ex['prompt'])
    oracle=tasks.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,n)
    ids=prefix+[tok.mask_id if m else tok.binary_ids[ex['bits'][i]] for i,m in enumerate(masked)]
    return dict(ids=ids,prefix=len(prefix),stage=stage,masked=masked,gold=ex['bits'],oracle=oracle,
                family=ex['family'],instance_id=ex['instance_id'],loss_mask=[i in groups[k] for i in range(n)],
                loss_scale=2/n,branch=branch)


def pack(canvases,device):
    batch=C.pack(canvases,device)
    batch['loss_mask']=torch.tensor([c['loss_mask'] for c in canvases],device=device,dtype=torch.bool)
    batch['loss_scale']=torch.tensor([c['loss_scale'] for c in canvases],device=device,dtype=torch.float32)
    batch['policy_branch']=torch.tensor([c['branch']=='policy' for c in canvases],device=device,dtype=torch.bool)
    return batch


def loss(logits,batch,objective):
    lp=logits.float().log_softmax(-1)
    if objective=='hard':ce=-lp.gather(-1,batch['gold'].unsqueeze(-1)).squeeze(-1)
    elif objective=='rao_blackwell':ce=-(batch['oracle']*lp[...,1]+(1-batch['oracle'])*lp[...,0])
    else:raise ValueError(objective)
    return ((ce*batch['loss_mask']).sum(-1)*batch['loss_scale']).mean()


def oracle_entropy(batch):
    entropy=torch.where(batch['oracle']==.5,torch.log(torch.tensor(2.,device=batch['oracle'].device)),0.)
    return ((entropy*batch['loss_mask']).sum(-1)*batch['loss_scale']).mean()
