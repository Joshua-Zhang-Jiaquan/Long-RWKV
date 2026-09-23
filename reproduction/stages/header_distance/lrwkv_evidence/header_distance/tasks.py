"""Fresh problems and independent, fixed-position header/task-block swaps."""
from collections import Counter
import hashlib
import itertools
from lrwkv_evidence.train04 import tasks as BASE
from lrwkv_evidence.train04.evaluate import support
from lrwkv_evidence.posterior_context_probe.tasks import serialize as legacy_serialize

SEED=2026092301
CELLS=('header_far_task_far','header_far_task_near',
       'header_near_task_far','header_near_task_near','legacy_far','legacy_near')


def example(index):
    if index not in range(32):raise ValueError('outside frozen panel')
    ex=BASE.make_example('test',SEED,index,family='paired_parity')
    ex['support']=[sum(b<<i for i,b in enumerate(y)) for y in support(ex)]
    return ex


def histories(ex):
    return [dict(zip(ex['information_set'],bits)) for bits in itertools.product((0,1),repeat=4)]


def take_records(records,budget):
    """First deterministic subset-sum solution; no record is split or repeated."""
    states={0:()}
    for i,record in enumerate(records):
        if record is None:continue
        for size,indices in list(states.items()):
            new=size+len(record)
            if new<=budget and new not in states:states[new]=indices+(i,)
        if budget in states:return states[budget]
    raise ValueError('no whole-record filler of required native length')


def serialize(ex,tok,cell,length=16384):
    if cell not in CELLS:raise ValueError('unknown layout')
    if cell.startswith('legacy_'):
        result=legacy_serialize(ex,tok,length,cell.split('_')[1])
        result['cell']=cell
        return result
    original=tok.encode(ex['prompt']);header=tok.encode(BASE.INSTRUCTION);footer=tok.encode('\nAnswer:')
    if original[:len(header)]!=header or original[-len(footer):]!=footer:
        raise ValueError('header/footer must have exact native token boundaries')
    block=original[len(header):-len(footer)]
    if tok.encode(ex['prompt'][len(BASE.INSTRUCTION):-len('\nAnswer:')])!=block:
        raise ValueError('task-block token boundary mismatch')
    records=[];used=0;budget=length-len(original)
    for i in range(100000):
        record=tok.encode('\narchive_'+str(i)+' = unused;')
        if used+len(record)>budget:break
        records.append(record);used+=len(record)
    if not records or not 0<=budget-used<len(record):raise ValueError('bad record packing')
    a_indices=take_records(records,len(header))
    available=[None if i in a_indices else r for i,r in enumerate(records)]
    b_indices=take_records(available,len(block))
    filler_a=sum((records[i] for i in a_indices),[])
    filler_b=sum((records[i] for i in b_indices),[])
    middle=sum((r for i,r in enumerate(records) if i not in (*a_indices,*b_indices)),[])
    _,a,_,b=cell.split('_')
    ids=(header if a=='far' else filler_a)+(block if b=='far' else filler_b)+middle
    ids+=(header if a=='near' else filler_a)+(block if b=='near' else filler_b)+footer
    late=len(filler_a)+len(filler_b)+len(middle)
    header_start=0 if a=='far' else late
    block_start=len(header) if b=='far' else late+len(header)
    legacy=legacy_serialize(ex,tok,length,'near')
    if len(ids)!=len(legacy['ids']) or Counter(ids)!=Counter(legacy['ids']):
        raise ValueError('length or token multiset changed')
    if ids[header_start:header_start+len(header)]!=header or ids[block_start:block_start+len(block)]!=block:
        raise ValueError('fixed slot misplaced')
    return dict(ids=ids,cell=cell,context_tokens=len(ids),requested_context_tokens=length,
                header_position=a,task_position=b,header_span=[header_start,header_start+len(header)],
                task_span=[block_start,block_start+len(block)],distractor_records=len(records),
                filler_header_records=list(a_indices),filler_task_records=list(b_indices),
                token_sha256=hashlib.sha256(str(ids).encode()).hexdigest())


def endpoint(ex,initial,history_lp):
    """Exact one-call and information-set laws; raw binary probabilities retained."""
    import math
    if len(history_lp)!=16:raise ValueError('all target histories required')
    for lp in [initial,*history_lp]:
        if len(lp)!=8:raise ValueError('eight coordinates required')
        for pair in lp:
            if len(pair)!=2 or not all(math.isfinite(v) and v<=1e-8 for v in pair) or abs(sum(math.exp(v) for v in pair)-1)>1e-6:
                raise ValueError('invalid normalized probabilities')
    first=ex['information_set'];second=[i for i in range(8) if i not in first]
    lookup={tuple(sorted(v.items())):lp for v,lp in zip(histories(ex),history_lp)}
    one=[];info=[];second_error=0.
    for y in ex['support']:
        visible={i:(y>>i)&1 for i in first};lp=lookup[tuple(sorted(visible.items()))]
        one.append(sum(initial[i][(y>>i)&1] for i in range(8)))
        error=-sum(lp[i][(y>>i)&1] for i in second);second_error+=error/16
        info.append(sum(initial[i][(y>>i)&1] for i in first)-error)
    first_error=sum(-math.log(2)-sum(initial[i])/2 for i in first)
    one_kl=-math.log(16)-sum(one)/16;info_kl=-math.log(16)-sum(info)/16
    if abs(info_kl-first_error-second_error)>1e-8 or one_kl<4*math.log(2)-1e-8:
        raise ValueError('exact decomposition failure')
    return dict(one_kl=one_kl,info_kl=info_kl,first_error=first_error,second_error=second_error,
                one_estimation=one_kl-4*math.log(2),gain_over_one=one_kl-info_kl,
                support_logq_one=one,support_logq_info=info,
                valid_mass_one=sum(math.exp(v) for v in one),valid_mass_info=sum(math.exp(v) for v in info))
