"""Competence controls preserve task identity and never put gold into decoder input."""
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from lrwkv_evidence.e3 import controls as C
from lrwkv_evidence.e3 import grid324 as G


@pytest.fixture(scope='module')
def context():
    hop, tp, registry = G._import_generator()
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    return hop, tp, encoder, G._PermutingPoolFactory(hop._Pool)


def make(context, family='associative_recall', *, length=16384, position=50, load=8):
    hop, tp, enc, pools = context
    cell = G.Cell(family, length, position, load, 'similar')
    task, canvas, reason = G.build_instance(hop, tp, enc, cell, 201, 0, pools, 'dev')
    assert reason is None
    return cell, task, canvas


@pytest.mark.parametrize('family', G.PAPER_FAMILIES)
def test_content_and_gold_preserved_without_leakage(context, family):
    _, task, canvas = make(context, family)
    short = C.evidence_only(canvas)
    assert short.input_ids == tuple(i for i, filler in zip(canvas.input_ids, canvas.filler_mask) if not filler)
    a, b = canvas.evidence_span
    x, y = short.evidence_span
    assert short.input_ids[x:y] == canvas.input_ids[a:b]
    assert short.answer_ids == canvas.answer_ids
    assert short.input_ids[y:short.answer_span[0]] == tuple(
        i for i, filler in zip(canvas.input_ids[b:canvas.answer_span[0]], canvas.filler_mask[b:canvas.answer_span[0]]) if not filler)
    assert short.input_ids[short.answer_span[0]:] == (canvas.mask_id,) * len(short.answer_ids[0])
    removed = C.no_evidence(canvas)
    assert len(removed.input_ids) == len(canvas.input_ids)
    assert removed.input_ids[a:b] == (canvas.pad_id,) * (b-a)
    assert removed.input_ids[:a] == canvas.input_ids[:a]
    assert removed.input_ids[b:] == canvas.input_ids[b:]
    assert removed.answer_ids == canvas.answer_ids
    assert removed.target_mask == canvas.target_mask


def test_anchors_render_same_task_without_new_draw(context):
    _, task, canvas = make(context)
    _, tp, enc, _ = context
    with patch.object(G, 'build_instance', side_effect=AssertionError('redrew logical task')):
        rows = C.transform_task(task, canvas, tp, enc, position=50, provenance={})
    assert len({r['provenance']['logical_task_sha256'] for r in rows}) == 1
    for row, length in zip(rows[2:], (4096, 8192)):
        ids = row['decoder_input']['input_ids']
        assert len(ids) == length
        assert row['gold_answers'] == list(task.answers)
        assert row['queries'] == list(task.queries)
        a, b = row['evidence_span']
        assert a == length // 2
        assert ids[a:b] == list(canvas.input_ids[slice(*canvas.evidence_span)])
        lo, hi = row['decoder_input']['target_span']
        assert ids[lo:hi] == [canvas.mask_id] * (hi-lo)
        assert set(row['decoder_input']) == {'input_ids', 'target_span'}
    assert rows == C.transform_task(task, canvas, tp, enc, position=50, provenance={})


def test_oversize_anchor_retains_refusal(context):
    _, task, canvas = make(context, length=32768, position=90, load=128)
    _, tp, enc, _ = context
    rows = C.transform_task(task, canvas, tp, enc, position=90, provenance={})
    assert all(r['status'] == 'unsupported' for r in rows[2:])
    assert all('no room' in r['unsupported_reason'] for r in rows[2:])
    assert all('decoder_input' not in r for r in rows[2:])


def test_rebuild_banked_dev_and_reject_tampered_metadata(context, tmp_path):
    hop, tp, enc, pools = context
    cell, _, _ = make(context)
    bank = G.build_cell(hop, tp, enc, cell, 1, pools, 'dev', (201,))
    path = tmp_path / 'dev.json'
    path.write_text(json.dumps(bank))
    rows = C.build_controls(path, 201, 0)
    assert rows == C.build_controls(path, 201, 0)
    assert all(len(r['transformation_sha256']) == 64 for r in rows)
    bank['instances'][0]['answers'] = ['v9999']
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match='metadata mismatch'):
        C.build_controls(path, 201, 0)
    bank['instances'][0]['input_ids_sha256'] = '0'*64
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match='hash mismatch'):
        C.build_controls(path, 201, 0)
    with pytest.raises(ValueError, match='dev data seed'):
        C.build_controls(path, 301, 0)


def test_filler_must_never_overlap_evidence(context):
    _, _, canvas = make(context)
    filler = list(canvas.filler_mask)
    filler[canvas.evidence_span[0]] = True
    with pytest.raises(ValueError, match='filler overlaps'):
        C.evidence_only(replace(canvas, filler_mask=tuple(filler)))
