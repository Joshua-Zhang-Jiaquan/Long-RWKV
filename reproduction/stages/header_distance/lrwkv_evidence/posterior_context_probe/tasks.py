import hashlib
from lrwkv_evidence.train04 import tasks as T
from lrwkv_evidence.train04.evaluate import fixed_groups,support
SEED=20270929

def example(index):
    ex=T.make_example('dev',SEED,index)
    ex['support']=[sum(b<<i for i,b in enumerate(y)) for y in support(ex)]
    return ex

def partitions(ex):
    result={k:fixed_groups(ex,k) for k in ('one','information_set')}
    if ex['family']=='paired_parity':
        pairs=sorted([sorted(i for i in range(8) if row>>i&1) for row in ex['matrix_rows']])
        first=sorted(pairs[0]+pairs[1])
    else:first=list(range(4))
    result['pair_preserving_halves']=[first,[i for i in range(8) if i not in first]]
    return result

def serialize(ex,tok,length=None,position='evidence_only'):
    original=tok.encode(ex['prompt']);footer=tok.encode('\nAnswer:')
    if original[-len(footer):]!=footer:raise ValueError('footer is not a native token boundary')
    block=original[:-len(footer)]
    if length is None:
        ids=original;start=0;count=0
    else:
        if position not in ('far','middle','near'):raise ValueError(position)
        budget=length-len(original)
        if budget<0:raise ValueError('context shorter than original')
        records=[];used=0
        for i in range(100000):
            record=tok.encode('\narchive_'+str(i)+' = unused;')
            if used+len(record)>budget:break
            records.append(record);used+=len(record)
        before={'far':0,'middle':len(records)//2,'near':len(records)}[position]
        left=sum(records[:before],[]);right=sum(records[before:],[])
        start=len(left);count=len(records);ids=left+block+right+footer
        # Whole records only; no arbitrary padding tokens.
        if not 0<=length-len(ids)<len(record):raise ValueError('bad whole-record packing')
    return dict(ids=ids,context_tokens=len(ids),requested_context_tokens=length,position=position,
                original_prompt_tokens=len(original),public_block_span=[start,start+len(block)],
                distractor_records=count,token_sha256=hashlib.sha256(str(ids).encode()).hexdigest())
