"""Stable binary log-odds, packed independent examples, and Rao--Blackwell loss.

The terminal N8 task is the original held-out-structure task. Small dimensions
are development curriculum only, never substituted for the final task.
"""
import hashlib
import math
import random
import torch
from torch import nn
import torch.nn.functional as F
from lrwkv_evidence.train04 import tasks


class StableBinaryHead(nn.Module):
    """Same binary softmax in exact arithmetic, without large common-logit rounding."""
    def __init__(self,head):
        super().__init__();self.weight=head.weight
        self.register_buffer('binary_ids',head.binary_ids.detach().clone())
    def forward(self,hidden):
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            selected=self.weight.index_select(0,self.binary_ids).float()
            logodds=F.linear(hidden.float(),(selected[1]-selected[0]).unsqueeze(0))
            return torch.cat((torch.zeros_like(logodds),logodds),dim=-1)


def example(split,seed,index,n):
    if n==8:return tasks.make_example(split,seed,index)
    if n not in (2,4) or split not in ('train','dev'):raise ValueError('curriculum only supports N2/4 training/development')
    key=f'curriculum_v1/{split}/{seed}/{index}/{n}'
    def rng(domain):return random.Random(int(hashlib.sha256((key+'/'+domain).encode()).hexdigest(),16))
    family=tasks.FAMILIES[index%2];permutation=list(range(n));rng('structure').shuffle(permutation)
    rows=[1<<j for j in permutation[:n//2]] if family=='systematic' else [(1<<permutation[j])|(1<<permutation[j+1]) for j in range(0,n,2)]
    rng('row_order').shuffle(rows);labels=list(range(1,n+1));rng('surface').shuffle(labels)
    names=[f'bit{v}' for v in labels];randomizer=rng('target');bits=[randomizer.randrange(2) for _ in range(n)]
    y=sum(v<<i for i,v in enumerate(bits));syndrome=[(y&row).bit_count()%2 for row in rows]
    equations=[' XOR '.join(names[j] for j in range(n) if row>>j&1)+f' = {c}' for row,c in zip(rows,syndrome)]
    prompt=tasks.INSTRUCTION+'\nBits: '+', '.join(names)+'\nConstraints:\n'+'\n'.join(equations)+'\nOutput order: '+', '.join(names)+'\nAnswer:'
    return dict(prompt=prompt,bits=bits,matrix_rows=rows,syndrome=syndrome,n=n,family=family,
                information_set=tasks.information_set(rows,n),instance_id=hashlib.sha256(key.encode()).hexdigest(),split=split,index=index,seed=seed)


def curriculum_n(step):
    if not 1<=step<=1500:raise ValueError('fixed development budget is1500updates')
    return 2 if step<=300 else (4 if step<=700 else 8)


def make_canvas(ex,tokenizer,seed,stage=None):
    rng=random.Random(int(hashlib.sha256(f"mask/{seed}/{ex['instance_id']}/{stage}".encode()).hexdigest(),16))
    stage=stage or rng.randrange(1,9);masked=[rng.random()<stage/8 for _ in ex['bits']]
    visible={i:b for i,b in enumerate(ex['bits']) if not masked[i]}
    oracle=tasks.oracle_marginals(ex['matrix_rows'],ex['syndrome'],visible,ex['n'])
    if oracle is None:raise ValueError('authentic visible bits must be feasible')
    prefix=tokenizer.encode(ex['prompt'])
    ids=prefix+[tokenizer.mask_id if selected else tokenizer.binary_ids[b] for b,selected in zip(ex['bits'],masked)]
    return dict(ids=ids,prefix=len(prefix),stage=stage,masked=masked,gold=ex['bits'],oracle=oracle,family=ex['family'],instance_id=ex['instance_id'])


def pack(canvases,device):
    sizes={len(c['gold']) for c in canvases}
    if len(sizes)!=1:raise ValueError('one curriculum dimension per batch required')
    ids=[];codes=[];times=[];starts=[];gather=[]
    for c in canvases:
        starts.append(len(ids));lo=len(ids)+c['prefix'];n=len(c['gold'])
        gather.extend(range(lo,lo+n));ids.extend(c['ids']);codes.extend([1]*c['prefix']+[3 if m else 2 for m in c['masked']]);times.extend([0.]*c['prefix']+[c['stage']/8]*n)
    return dict(input_ids=torch.tensor([ids],device=device,dtype=torch.long),doc_starts=torch.tensor([starts],device=device,dtype=torch.long),
                attention_mask=torch.ones((1,len(ids)),device=device,dtype=torch.bool),codes=torch.tensor([codes],device=device,dtype=torch.long),
                block_t=torch.tensor([times],device=device,dtype=torch.float32),gather_idx=torch.tensor(gather,device=device,dtype=torch.long),
                gold=torch.tensor([c['gold'] for c in canvases],device=device,dtype=torch.long),
                oracle=torch.tensor([c['oracle'] for c in canvases],device=device,dtype=torch.float32),
                masked=torch.tensor([c['masked'] for c in canvases],device=device,dtype=torch.bool),
                stage=torch.tensor([c['stage'] for c in canvases],device=device,dtype=torch.float32))


class PackedDenoiser(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,batch):
        logits=self.model.backbone(**{k:batch[k] for k in ('input_ids','doc_starts','attention_mask','codes','block_t','gather_idx')},block_size=1,R=1,force_forward=False)
        return logits.reshape(*batch['gold'].shape,2)


class SerialDenoiser(PackedDenoiser):
    """Evaluate each document with its own recurrent kernel invocation."""
    def forward(self,batch):
        starts=batch['doc_starts'][0].tolist()
        ends=starts[1:]+[batch['input_ids'].shape[1]]
        outputs=[]
        n=batch['gold'].shape[1]
        for j,(lo,hi) in enumerate(zip(starts,ends)):
            one={k:batch[k][:,lo:hi] for k in ('input_ids','attention_mask','codes','block_t')}
            one['doc_starts']=torch.zeros((1,1),dtype=torch.long,device=batch['input_ids'].device)
            one['gather_idx']=batch['gather_idx'][j*n:(j+1)*n]-lo
            one['gold']=batch['gold'][j:j+1]
            outputs.append(super().forward(one))
        return torch.cat(outputs,dim=0)


def loss(logits,batch,objective):
    lp=logits.float().log_softmax(-1)
    if objective=='hard':ce=-lp.gather(-1,batch['gold'].unsqueeze(-1)).squeeze(-1)
    elif objective=='rao_blackwell':ce=-(batch['oracle']*lp[...,1]+(1-batch['oracle'])*lp[...,0])
    else:raise ValueError('unknown objective')
    n=ce.shape[1]
    return ((ce*batch['masked']).sum(-1)*8/(batch['stage']*n)).mean()
