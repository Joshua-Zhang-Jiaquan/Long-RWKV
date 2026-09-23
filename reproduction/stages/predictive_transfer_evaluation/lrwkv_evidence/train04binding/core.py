"""Candidate public-label scaffold; not enabled in existing experiments."""
import re
import torch


def decorate(example,canvas,tokenizer):
    # Obtain labels from public text, never from gold values or the parity solver.
    names=example['prompt'].rsplit('\nOutput order: ',1)[1].split('\n',1)[0].split(', ')
    n=len(canvas['masked']);prefix=canvas['prefix']
    if len(names)!=n or len(set(names))!=n or any(not re.fullmatch(r'bit[0-9]+',v) for v in names):
        raise ValueError('invalid public output order')
    target_ids=canvas['ids'][prefix:]
    if len(target_ids)!=n:raise ValueError('expected original contiguous canvas')
    ids=list(canvas['ids'][:prefix]);positions=[];codes=[1]*prefix;times=[0.]*prefix
    for name,value,masked in zip(names,target_ids,canvas['masked']):
        marker=tokenizer.encode(' '+name+'=')
        ids.extend(marker);codes.extend([1]*len(marker));times.extend([0.]*len(marker))
        positions.append(len(ids));ids.append(value);codes.append(3 if masked else 2);times.append(canvas['stage']/8)
        separator=tokenizer.encode(';')
        ids.extend(separator);codes.extend([1]*len(separator));times.extend([0.]*len(separator))
    return {**canvas,'ids':ids,'target_positions':positions,'position_codes':codes,'position_times':times}


def pack(canvases,device):
    if len({len(c['gold']) for c in canvases})!=1:raise ValueError('mixed dimensions')
    ids=[];codes=[];times=[];starts=[];gather=[]
    for c in canvases:
        offset=len(ids);starts.append(offset);ids.extend(c['ids']);codes.extend(c['position_codes']);times.extend(c['position_times'])
        gather.extend(offset+i for i in c['target_positions'])
    batch=dict(input_ids=torch.tensor([ids],device=device,dtype=torch.long),
        doc_starts=torch.tensor([starts],device=device,dtype=torch.long),attention_mask=torch.ones((1,len(ids)),device=device,dtype=torch.bool),
        codes=torch.tensor([codes],device=device,dtype=torch.long),block_t=torch.tensor([times],device=device,dtype=torch.float32),
        gather_idx=torch.tensor(gather,device=device,dtype=torch.long),gold=torch.tensor([c['gold'] for c in canvases],device=device,dtype=torch.long),
        oracle=torch.tensor([c['oracle'] for c in canvases],device=device,dtype=torch.float32),masked=torch.tensor([c['masked'] for c in canvases],device=device,dtype=torch.bool),
        stage=torch.tensor([c['stage'] for c in canvases],device=device,dtype=torch.float32))
    if any('loss_mask' in c for c in canvases):
        if not all('loss_mask' in c for c in canvases):raise ValueError('mixed loss schemas')
        batch.update(loss_mask=torch.tensor([c['loss_mask'] for c in canvases],device=device,dtype=torch.bool),
                     loss_scale=torch.tensor([c['loss_scale'] for c in canvases],device=device,dtype=torch.float32),
                     policy_branch=torch.tensor([c['branch']=='policy' for c in canvases],device=device,dtype=torch.bool))
    return batch
