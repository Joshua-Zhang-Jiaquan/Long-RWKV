"""Pure CPU task generator. Class allocation uses no neural outcomes."""
import hashlib
import itertools
import random
from functools import lru_cache
from .algebra import rank, inverse, profile
from lrwkv_evidence.train04.tasks import INSTRUCTION, oracle_marginals


def digest(x):
    return hashlib.sha256(str(x).encode()).hexdigest()


def rng(*parts):
    return random.Random(int(digest(parts), 16))


@lru_cache(None)
def catalog():
    groups = {}
    for rows in itertools.product(range(1, 16), repeat=4):
        if rank(rows) == 4:
            groups.setdefault(profile(rows), []).append(rows)
    ordered = sorted(groups, key=lambda x: digest(('transfer_class_split_v1', x)))
    return {str(i): dict(weight_enumerator=list(key), split='source' if i < 15 else 'heldout',
                        matrices=groups[key]) for i, key in enumerate(ordered)}


def make_example(split, seed, index, class_id=None):
    if split not in ('train', 'calibration', 'heldout', 'qualification'):
        raise ValueError(split)
    allowed = list(range(15, 25)) if split == 'heldout' else list(range(15))
    class_id = allowed[index % len(allowed)] if class_id is None else int(class_id)
    if class_id not in allowed:
        raise ValueError('class outside split')
    key = ('transfer_v1', split, seed, index, class_id)
    randomizer = rng(*key)
    B = randomizer.choice(catalog()[str(class_id)]['matrices'])
    old_to_new = list(range(8)); randomizer.shuffle(old_to_new)
    names = ['bit'+str(v) for v in randomizer.sample(range(1, 9), 8)]
    bits = [randomizer.randrange(2) for _ in range(8)]
    y = sum(v << i for i, v in enumerate(bits))
    original = [row | (1 << (4+j)) for j, row in enumerate(B)]
    rows = [sum(((row >> i) & 1) << old_to_new[i] for i in range(8)) for row in original]
    randomizer.shuffle(rows)
    syndrome = [(y & row).bit_count() % 2 for row in rows]
    bases = [sorted(old_to_new[:4]), sorted(old_to_new[4:])]
    randomizer.shuffle(bases)  # public A/B do not encode generator orientation
    equations = [' XOR '.join(names[j] for j in range(8) if row >> j & 1)+f' = {s}'
                 for row, s in zip(rows, syndrome)]
    prompt = (INSTRUCTION+'\nBits: '+', '.join(names)+'\nConstraints:\n'+'\n'.join(equations)
              +'\nBasis A: '+', '.join(names[i] for i in bases[0])
              +'\nBasis B: '+', '.join(names[i] for i in bases[1])
              +'\nOutput order: '+', '.join(names)+'\nAnswer:')
    return dict(prompt=prompt, bits=bits, matrix_rows=rows, syndrome=syndrome, n=8,
                family='invertible_linear_code', bases=bases, class_id=class_id,
                instance_id=digest(key), split=split, seed=seed, index=index)


def public_bases(ex):
    names = ex['prompt'].split('\nOutput order: ')[1].split('\n')[0].split(', ')
    return [[names.index(v) for v in ex['prompt'].split('\nBasis '+label+': ')[1].split('\n')[0].split(', ')]
            for label in ('A', 'B')]


def support(ex):
    return [[(y >> i) & 1 for i in range(8)] for y in range(256)
            if all((y & row).bit_count() % 2 == s for row, s in zip(ex['matrix_rows'], ex['syndrome']))]


def features(ex, policy, position):
    """Public structural bins, one per revealed output, in round order."""
    first = public_bases(ex)[policy]; second = [i for i in range(8) if i not in first]
    result = []
    for i in first:
        degree = sum((r >> i) & 1 for r in ex['matrix_rows'])
        result.append((i, ('fair', 0, degree, position)))
    for i in second:
        matches = []
        for mask in range(1, 16):
            relation = 0
            for j, row in enumerate(ex['matrix_rows']):
                if mask >> j & 1: relation ^= row
            if relation >> i & 1 and all(j == i or j in first or not (relation >> j & 1) for j in range(8)):
                matches.append((relation.bit_count()-1, mask.bit_count()))
        if len(matches) != 1:
            raise ValueError('invertible basis must uniquely determine each output')
        fanin, equations = matches[0]
        result.append((i, ('deterministic', fanin, equations, position)))
    return result
