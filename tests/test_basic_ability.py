"""Checks for paired inference and fail-closed artifact selection."""
import json

import pytest

from lrwkv_evidence.basic_ability import audit_file, paired_contrast, _paired_values


def test_identical_outputs_have_exact_zero_interval():
    values = {'a': 0, 'b': 1, 'c': 1}
    row = paired_contrast(values, values)
    assert row['delta_percentage_points'] == 0
    assert row['ci95_percentage_points'] == [0, 0]
    assert row['left_only_correct'] == row['right_only_correct'] == 0


def test_mismatched_ids_rejected_even_at_same_size():
    with pytest.raises(ValueError, match='document ID sets'):
        paired_contrast({'a': 1, 'b': 0}, {'a': 1, 'c': 0})


def test_exact_difference_discordance_and_order_independence():
    left = {'a': 1, 'b': 1, 'c': 1, 'd': 0}
    right = {'d': 1, 'c': 0, 'b': 0, 'a': 1}
    row = paired_contrast(left, right, replicates=1000)
    assert row['delta_percentage_points'] == 25
    assert (row['left_only_correct'], row['right_only_correct']) == (2, 1)
    assert (row['both_correct'], row['both_incorrect']) == (1, 0)
    assert row == paired_contrast(dict(reversed(list(left.items()))), right, replicates=1000)
    other = paired_contrast(left, right, replicates=1000, batch_size=13)
    assert row['ci95_percentage_points'] == other['ci95_percentage_points']


def source(tmp_path, records):
    path = tmp_path / 'merged.json'
    selected = [r for r in records if r['arm'] == 'raw']
    data = dict(records=records, record_count=len(records), metrics=[
        dict(arm='raw', metric='correct', mean=sum(r['metrics']['correct'] for r in selected)/len(selected),
             n_documents=len({r['document_id'] for r in selected}))])
    path.write_text(json.dumps(data))
    return path


def record(i, value=0, arm='raw'):
    return dict(document_id=f'{i:06d}', arm=arm, seed=42, metrics={'correct': value})


def test_duplicate_ids_fail_closed(tmp_path):
    row = audit_file(source(tmp_path, [record(0), record(0), record(1, 1)]), 2, 'raw')
    assert row['status'] == 'incomplete'
    assert row['duplicate_documents'] == {'000000': 2}
    assert row['accuracy'] is None


def test_other_arm_excluded_and_mutated_source_rejected(tmp_path):
    path = source(tmp_path, [record(0), record(1, 1), record(0, 1, 'fwdce')])
    row = audit_file(path, 2, 'raw')
    assert row['status'] == 'complete'
    assert row['accuracy'] == .5
    assert row['excluded_records'] == 1
    assert _paired_values(row) == {'000000': 0, '000001': 1}
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='source changed'):
        _paired_values(row)
