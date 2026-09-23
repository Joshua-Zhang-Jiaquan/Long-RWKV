"""Exact finite posterior checks independent of Gaussian-elimination internals."""
from itertools import product
import re

import pytest

from lrwkv_evidence.train04.tasks import (
    make_example, oracle_marginals, solve_affine, information_set, FAMILIES)


def support(rows, syndrome, n, visible):
    return [x for x in range(1 << n)
            if all((x & row).bit_count() % 2 == bit for row, bit in zip(rows, syndrome))
            and all((x >> j) & 1 == bit for j, bit in visible.items())]


@pytest.mark.parametrize('family', FAMILIES)
def test_exact_uniform_posterior_and_all_visible_masks(family):
    example = make_example('test', 73, 4, family=family)
    rows, syndrome = example['matrix_rows'], example['syndrome']
    gold = sum(v << j for j, v in enumerate(example['bits']))
    feasible = support(rows, syndrome, 8, {})
    assert len(feasible) == 16 and gold in feasible
    info = example['information_set']
    assert len(info) == 4
    assert len({tuple(x >> j & 1 for j in info) for x in feasible}) == 16
    for mask in range(256):
        visible = {j: gold >> j & 1 for j in range(8) if mask >> j & 1}
        choices = support(rows, syndrome, 8, visible)
        exact = [sum(x >> j & 1 for x in choices) / len(choices) for j in range(8)]
        assert oracle_marginals(rows, syndrome, visible) == exact
    # Constraint-inconsistent completed output must be infeasible.
    bad = {j: gold >> j & 1 for j in range(8)}
    bad[(rows[0] & -rows[0]).bit_length()-1] ^= 1
    assert oracle_marginals(rows, syndrome, bad) is None


def test_prompt_names_map_exactly_to_output_coordinates_without_gold_fields():
    for index in range(20):
        e = make_example('train', 9, index)
        assert e['family'] == FAMILIES[index % 2]
        name_to_coordinate = {name: j for j, name in enumerate(e['coordinate_names'])}
        lines = e['prompt'].split('Constraints:\n')[1].split('\nOutput order:')[0].splitlines()
        reconstructed = []
        for line in lines:
            lhs, rhs = line.split(' = ')
            mask = sum(1 << name_to_coordinate[name] for name in lhs.split(' XOR '))
            reconstructed.append((mask, int(rhs)))
        assert reconstructed == list(zip(e['matrix_rows'], e['syndrome']))
        assert e['prompt'].endswith('Output order: '+', '.join(e['coordinate_names'])+'\nAnswer:')
        assert 'gold' not in e['prompt'].lower()
        assert 'Example:' not in e['prompt']
        assert e['instance_id'][:16] not in e['prompt']
        assert len(support(e['matrix_rows'], e['syndrome'], 8, {})) == 16


def test_split_determinism_random_syndromes_and_ood_dimension_contract():
    examples = [make_example(split, 17, 0) for split in ('train','dev','test','ood')]
    assert len({e['instance_id'] for e in examples}) == 4
    assert len({e['prompt'] for e in examples}) == 4
    assert all(e == make_example(e['split'], 17, 0) for e in examples)
    assert make_example('OOD',17,0) == make_example('ood',17,0)
    for family in FAMILIES:
        codes = {tuple(make_example('train', 42, i, family=family)['syndrome']) for i in range(128)}
        assert len(codes) == 16  # fixed reproducible seed coverage, not a statistical threshold
    for n in (12,16):
        for family in FAMILIES:
            e = make_example('ood', 19, 3, n=n, family=family)
            solution = solve_affine(e['matrix_rows'], e['syndrome'], n)
            assert len(solution[1]) == n//2
            assert len(e['information_set']) == n//2
            assert oracle_marginals(e['matrix_rows'],e['syndrome'],dict(enumerate(e['bits'])),n) == list(map(float,e['bits']))
        with pytest.raises(ValueError): make_example('test',19,3,n=n)


def test_dependent_rows_inconsistent_syndromes_and_underspecified_systematic_coordinates():
    assert solve_affine([1,1],[0,1],8) is None
    assert oracle_marginals([1],[1],{},n=8) == [1.] + [.5]*7
    assert information_set([1,1],8) == list(range(1,8))
    assert oracle_marginals([],[],{},n=8) == [.5]*8
    with pytest.raises(ValueError): oracle_marginals([256],[0],n=8)
    with pytest.raises(ValueError): oracle_marginals([1],[0],{8:1},n=8)


def test_complete_structure_catalogs_are_disjoint_and_exhaustive():
    from lrwkv_evidence.train04.tasks import structure_catalog, split_catalog
    from itertools import combinations
    for family, total, sizes in [('systematic',70,(49,10,11)),('paired_parity',105,(73,16,16))]:
        catalog=structure_catalog(family)
        assert len(catalog)==len(set(catalog))==total
        splits=[set(split_catalog(split,family)) for split in ('train','dev','test')]
        assert tuple(map(len,splits))==sizes
        assert not (splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])
        assert set.union(*splits)==set(catalog)
        for rows in catalog:
            assert tuple(sorted(rows))==rows
            assert len(rows)==4
            assert all(row.bit_count()==(1 if family=='systematic' else 2) for row in rows)
            assert all(a&b==0 for a,b in combinations(rows,2))
        for split, allowed in zip(('train','dev','test'),splits):
            for i in range(200):
                e=make_example(split,43,i,family=family)
                assert tuple(sorted(e['matrix_rows'])) in allowed
                assert e['structure_split']==split
        assert split_catalog('ood',family)==split_catalog('test',family)
