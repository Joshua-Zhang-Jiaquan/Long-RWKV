"""Deterministic finite GF(2) posterior tasks for the 0.4B training study.

Coordinate i is bit i of every integer matrix mask and position i in `bits`.
Random one-based surface names are mapped by coordinate_names and explicit output
order. N=8 train/dev/test hold out canonical labeled encoder structures.
This is not unseen graph topology: every family is isomorphic under relabeling.
Syndromes and targets can repeat; OOD at N=8 reuses test structures.
No neural framework or GPU imports; elimination is polynomial in dimension.
"""
from __future__ import annotations
import hashlib
import json
import random
from functools import lru_cache
from itertools import combinations

VERSION = 'train04_gf2_tasks_v2_structure_split'
FAMILIES = ('systematic', 'paired_parity')
SPLITS = ('train', 'dev', 'test', 'ood')
INSTRUCTION = ('Complete the binary vector satisfying all listed XOR constraints. '
               'Return only the bits in the stated output order, without separators.')


def _validate(rows, syndrome, n):
    if type(n) is not int or not 1 <= n <= 16:
        raise ValueError('n must be an integer from 1 through 16')
    if len(rows) != len(syndrome):
        raise ValueError('one syndrome bit per matrix row required')
    if any(type(row) is not int or not 0 <= row < 1 << n for row in rows):
        raise ValueError('matrix masks must lie inside n coordinates')
    if any(type(bit) is not int or bit not in (0, 1) for bit in syndrome):
        raise ValueError('syndrome values must be integer bits')


def solve_affine(rows, syndrome, n=8):
    """Return (particular integer vector, nullspace basis, pivots), or None.

    Free coordinates of a uniform affine solution are independent fair bits.
    Choosing any of 2**len(basis) basis combinations gives each solution once.
    """
    rows, syndrome = list(rows), list(syndrome)
    _validate(rows, syndrome, n)
    equations = [[row, bit] for row, bit in zip(rows, syndrome)]
    pivots = []
    rank = 0
    for column in range(n):
        selected = next((i for i in range(rank, len(equations))
                         if equations[i][0] >> column & 1), None)
        if selected is None:
            continue
        equations[rank], equations[selected] = equations[selected], equations[rank]
        mask, rhs = equations[rank]
        for i in range(len(equations)):
            if i != rank and equations[i][0] >> column & 1:
                equations[i][0] ^= mask
                equations[i][1] ^= rhs
        pivots.append(column)
        rank += 1
    if any(mask == 0 and rhs for mask, rhs in equations):
        return None
    particular = sum(equations[i][1] << p for i, p in enumerate(pivots))
    basis = []
    for free in range(n):
        if free in pivots:
            continue
        vector = 1 << free
        for i, pivot in enumerate(pivots):
            if equations[i][0] >> free & 1:
                vector |= 1 << pivot
        basis.append(vector)
    return particular, basis, pivots


def information_set(rows, n=8):
    """Original-coordinate free positions giving a bijection on each affine code."""
    rows = list(rows)
    _, _, pivots = solve_affine(rows, [0] * len(rows), n)
    return [i for i in range(n) if i not in pivots]


def oracle_marginals(rows, syndrome, visible=None, n=8):
    """P(Y_i=1 | A Y=C, visible coordinates), or None if infeasible.

    `visible` maps zero-based output coordinate indices to integer bits. Pass n
    explicitly for OOD dimensions; n cannot be inferred from unconstrained bits.
    The returned list includes the deterministic visible-coordinate marginals.
    """
    rows, syndrome = list(rows), list(syndrome)
    _validate(rows, syndrome, n)
    for coordinate, bit in (visible or {}).items():
        if type(coordinate) is not int or not 0 <= coordinate < n:
            raise ValueError('visible coordinate outside output vector')
        if type(bit) is not int or bit not in (0, 1):
            raise ValueError('visible values must be integer bits')
        rows.append(1 << coordinate)
        syndrome.append(bit)
    solution = solve_affine(rows, syndrome, n)
    if solution is None:
        return None
    particular, basis, _ = solution
    varying = 0
    for vector in basis:
        varying |= vector
    return [.5 if varying >> i & 1 else float(particular >> i & 1) for i in range(n)]


@lru_cache(maxsize=None)
def structure_catalog(family, n=8):
    """All N=8 labeled structures in a reproducible content-hash order.

    Systematic structures are the 70 four-coordinate subsets. Paired-parity
    structures are the 105 perfect matchings. No row-order duplicates occur.
    Larger OOD dimensions use direct construction, avoiding huge catalogs.
    """
    if n != 8 or family not in FAMILIES:
        raise ValueError('canonical split catalogs support N=8 and known families')
    if family == 'systematic':
        catalog = [tuple(sorted(1 << j for j in subset)) for subset in combinations(range(n), n//2)]
    else:
        def matchings(remaining):
            if not remaining:
                yield ()
                return
            first = remaining[0]
            for partner in remaining[1:]:
                rest = tuple(j for j in remaining if j not in (first, partner))
                for suffix in matchings(rest):
                    yield tuple(sorted(((1 << first) | (1 << partner),) + suffix))
        catalog = list(matchings(tuple(range(n))))
    def order(rows):
        identity = [VERSION, 'canonical_structure_partition', family, n, rows]
        return hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest(), rows
    return tuple(sorted(catalog, key=order))


@lru_cache(maxsize=None)
def split_catalog(split, family, n=8):
    """Disjoint structure splits; sizes 49/10/11 or 73/16/16.

    Train takes floor(0.7 * catalog size), then dev takes floor(remaining/2)
    and test takes the rest. N=8 OOD intentionally reuses test structures with
    a separate sampling namespace; it is not another disjoint structure split.
    """
    if split in ('ood', 'OOD'):
        split = 'test'
    if split not in ('train', 'dev', 'test'):
        raise ValueError('unknown structure split')
    catalog = structure_catalog(family, n)
    train_end = 7 * len(catalog) // 10
    dev_end = train_end + (len(catalog) - train_end) // 2
    return {'train': catalog[:train_end], 'dev': catalog[train_end:dev_end],
            'test': catalog[dev_end:]}[split]


def make_example(split, seed, index, n=8, family=None):
    """Generate uniform Y first, then C=AY: Y|A,C is exactly affine-uniform.

    Family assignment is balanced by index parity when not specified. Matrix
    and surface permutations are chosen independently before the random target.
    N=8 train/dev/test select disjoint canonical labeled encoder structures.
    Dimensions 12/16 are OOD-only, with rank n/2 and n/2 latent bits. IDs are
    receipt metadata only; prompts contain no seed or target-derived nonce.
    """
    if split == 'OOD':
        split = 'ood'
    if split not in SPLITS:
        raise ValueError('split must be train, dev, test, or ood')
    if type(seed) is not int or type(index) is not int or index < 0:
        raise ValueError('integer seed and nonnegative integer index required')
    if n not in (8, 12, 16) or type(n) is not int or (n != 8 and split != 'ood'):
        raise ValueError('n=8 for train/dev/test; n=12/16 only for ood')
    if family is None:
        family = FAMILIES[index % len(FAMILIES)]
    if family not in FAMILIES:
        raise ValueError('unknown linear-compression family')
    namespace = [VERSION, split, seed, index, n, family]
    instance_id = hashlib.sha256(json.dumps(namespace, separators=(',', ':')).encode()).hexdigest()
    # Separate streams prevent prompt-format changes from changing target draws.
    def stream(domain):
        return random.Random(int(hashlib.sha256((instance_id + '/' + domain).encode()).hexdigest(), 16))
    structural = stream('matrix')
    permutation = list(range(n)); structural.shuffle(permutation)
    if n == 8:
        catalog = split_catalog(split, family, n)
        selected = catalog[structural.randrange(len(catalog))]
        matrix_rows = list(selected)
        # Preserve metadata convention: constrained singleton coordinates first,
        # or consecutive pair coordinates. Row order is a separate nuisance draw.
        permutation = [j for row in selected for j in range(n) if row >> j & 1]
        permutation += [j for j in range(n) if j not in permutation]
    elif family == 'systematic':
        matrix_rows = [1 << j for j in permutation[:n//2]]
    else:
        matrix_rows = [(1 << permutation[j]) | (1 << permutation[j+1]) for j in range(0, n, 2)]
    stream('row_order').shuffle(matrix_rows)
    labels = list(range(1, n+1)); stream('surface').shuffle(labels)
    coordinate_names = [f'bit{label}' for label in labels]
    target_rng = stream('target')
    bits = [target_rng.randrange(2) for _ in range(n)]
    target = sum(bit << j for j, bit in enumerate(bits))
    syndrome = [(target & row).bit_count() % 2 for row in matrix_rows]
    equations = [' XOR '.join(coordinate_names[j] for j in range(n) if row >> j & 1)
                 + f' = {value}' for row, value in zip(matrix_rows, syndrome)]
    prompt = (INSTRUCTION + '\nBits: ' + ', '.join(coordinate_names)
              + '\nConstraints:\n' + '\n'.join(equations)
              + '\nOutput order: ' + ', '.join(coordinate_names) + '\nAnswer:')
    return dict(prompt=prompt, bits=bits, matrix_rows=matrix_rows, syndrome=syndrome,
                information_set=information_set(matrix_rows, n), family=family,
                instance_id=instance_id, split=split, seed=seed, index=index, n=n,
                rank=n//2, latent_bits=n//2, coordinate_names=coordinate_names,
                coordinate_permutation=permutation, schema=VERSION,
                structure_id=hashlib.sha256(json.dumps([family, n, sorted(matrix_rows)], separators=(',', ':')).encode()).hexdigest(),
                structure_split=('test' if split == 'ood' and n == 8 else split),
                split_semantics='N=8 train/dev/test canonical labeled encoder structures are disjoint, not unseen graph topologies. Targets/syndromes may repeat. N=8 OOD reuses test structures; N=12/16 are dimension OOD.')
