"""Paired-length lookup tasks with known deterministic or dependent answer laws.

Prompt construction reads public records only. Gold draws and oracle conditionals
are separate outputs for supervised training/evaluation, never prompt inputs.
"""
import hashlib
import random
from lrwkv_evidence.train04.tasks import oracle_marginals

VERSION = 'long_context_lookup_posterior_v1'
FAMILIES = ('retrieval', 'dependent')
SPLITS = ('train', 'dev', 'test')
BANK_SIZE = 32768
N = 8


def rng(*parts):
    return random.Random(int(hashlib.sha256('/'.join(map(str, (VERSION, *parts))).encode()).hexdigest(), 16))


def key_bank(split):
    if split not in SPLITS:
        raise ValueError('unknown split')
    offset = SPLITS.index(split) * BANK_SIZE
    return range(offset, offset + BANK_SIZE)


def record_text(key, value):
    return f'key{key:05x}: ' + ''.join(map(str, value)) + '\n'


def problem(split, seed, index, family):
    if family not in FAMILIES or type(index) is not int or index < 0:
        raise ValueError('invalid family or index')
    # Sharing this RNG across the two families pairs public facts as well as lengths.
    r = rng(split, seed, index, 'public')
    key = r.choice(key_bank(split)); value = [r.randrange(2) for _ in range(4)]
    if family == 'retrieval':
        rows = [1 << i for i in range(N)]; syndrome = value + value
        rule = 'Look up the query key. Repeat its four bits twice in slots a,b,c,d,e,f,g,h.'
    else:
        rows = [(1 << i) | (1 << (i + 4)) for i in range(4)]; syndrome = value
        rule = ('Look up the query key with value v1,v2,v3,v4. Return eight bits uniformly '
                'subject to a XOR e=v1; b XOR f=v2; c XOR g=v3; d XOR h=v4.')
    support = [y for y in range(1 << N) if all((y & a).bit_count() % 2 == c for a,c in zip(rows, syndrome))]
    draw = rng(split, seed, index, family, 'gold').choice(support)
    return dict(split=split, seed=seed, index=index, family=family, key=key,
                value=value, rule=rule, matrix_rows=rows, syndrome=syndrome,
                support=support, bits=[(draw >> i) & 1 for i in range(N)], n=N,
                instance_id=hashlib.sha256(f'{VERSION}/{split}/{seed}/{index}/{family}'.encode()).hexdigest())


def serialize(ex, tokenizer, context_tokens=None, position='middle'):
    if position not in ('near', 'middle', 'far'):
        raise ValueError('unknown evidence position')
    if context_tokens is not None and (type(context_tokens) is not int or context_tokens < 1):
        raise ValueError('context token budget must be positive or None')
    header = tokenizer.encode(ex['rule'] + '\nRecords:\n')
    evidence = tokenizer.encode(record_text(ex['key'], ex['value']))
    footer = tokenizer.encode(f'Query: key{ex["key"]:05x}\nOutput slots: a,b,c,d,e,f,g,h\nAnswer:')
    minimum = len(header) + len(evidence) + len(footer)
    if context_tokens is None:
        prefix = header + evidence + footer
        return dict(prefix_ids=prefix, evidence_span=[len(header), len(header)+len(evidence)],
                    context_tokens=len(prefix), records=1, filler_tokens=0)
    if context_tokens < minimum:
        raise ValueError(f'context budget below evidence-only length {minimum}')
    available = context_tokens - minimum
    # Query follows the records: near is near the query, far is near the start.
    before_budget = available if position == 'near' else 0 if position == 'far' else available // 2
    after_budget = available - before_budget
    keys = [k for k in key_bank(ex['split']) if k != ex['key']]
    r = rng(ex['split'], ex['seed'], ex['index'], 'distractors'); r.shuffle(keys)
    newline = tokenizer.encode('\n')
    if len(newline) != 1:
        raise ValueError('exact-length padding requires one native newline token')
    cursor = 0
    def fill(budget):
        nonlocal cursor
        tokens = []; count = 0
        while cursor < len(keys):
            key = keys[cursor]
            # Values depend on key, not how many rows fit at a different length.
            v_rng = rng(ex['split'], ex['seed'], ex['index'], key, 'record')
            row = tokenizer.encode(record_text(key, [v_rng.randrange(2) for _ in range(4)]))
            if len(tokens) + len(row) > budget:
                break
            cursor += 1; count += 1; tokens.extend(row)
        slack = budget - len(tokens)
        if slack and cursor == len(keys):
            raise ValueError('distractor bank exhausted')
        return tokens + newline * slack, count, slack
    before, b_count, b_slack = fill(before_budget)
    after, a_count, a_slack = fill(after_budget)
    lo = len(header) + len(before)
    prefix = header + before + evidence + after + footer
    if len(prefix) != context_tokens:
        raise AssertionError('incorrect exact context length')
    return dict(prefix_ids=prefix, evidence_span=[lo, lo+len(evidence)],
                context_tokens=len(prefix), records=b_count+a_count+1,
                filler_tokens=b_slack+a_slack)


def inference_canvas(ex, serialized, tokenizer, visible=None, stage=8):
    """Build model inputs even for an infeasible model-generated history.

    Dummy labels supply shapes only; no posterior solver or gold draw is read.
    """
    visible = {} if visible is None else dict(visible)
    if any(type(i) is not int or not 0 <= i < N or type(b) is not int or b not in (0,1)
           for i,b in visible.items()):
        raise ValueError('invalid visible coordinates or bits')
    if type(stage) is not int or not 1 <= stage <= 8:
        raise ValueError('invalid corruption stage')
    prefix = serialized['prefix_ids']; masked = [i not in visible for i in range(N)]
    ids = prefix + [tokenizer.mask_id if masked[i] else tokenizer.binary_ids[visible[i]] for i in range(N)]
    return dict(ids=ids, prefix=len(prefix), stage=stage, masked=masked,
                gold=[0]*N, oracle=[.5]*N, family=ex['family'], instance_id=ex['instance_id'])


def canvas(ex, serialized, tokenizer, visible=None, stage=8):
    visible = {} if visible is None else dict(visible)
    result = inference_canvas(ex, serialized, tokenizer, visible, stage)
    oracle = oracle_marginals(ex['matrix_rows'], ex['syndrome'], visible, N)
    if oracle is None:
        raise ValueError('inconsistent visible assignment')
    return dict(result, gold=ex['bits'], oracle=oracle)
