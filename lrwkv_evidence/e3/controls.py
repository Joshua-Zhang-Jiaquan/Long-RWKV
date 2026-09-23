"""CPU-only competence controls derived from verified, banked dev instances.

No new logical draw occurs at an anchor length: render the original task again.
Gold is audit metadata only; decoder_input contains only masked IDs and a span.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from . import grid324 as G


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validate_canvas(canvas):
    n = len(canvas.input_ids)
    if any(len(getattr(canvas, name)) != n for name in ('target_mask', 'filler_mask', 'attention_mask')):
        raise ValueError('canvas mask lengths disagree')
    positions = [i for i, active in enumerate(canvas.target_mask) if active]
    if not positions or positions != list(range(positions[0], n)):
        raise ValueError('target mask must be a nonempty contiguous suffix')
    if any(canvas.input_ids[i] != canvas.mask_id for i in positions):
        raise ValueError('target suffix contains an unmasked token')
    lo, hi = canvas.evidence_span
    if not 0 <= lo < hi <= positions[0]:
        raise ValueError('evidence span overlaps target or is invalid')
    if any(canvas.filler_mask[i] for i in range(lo, hi)) or any(canvas.filler_mask[i] for i in positions):
        raise ValueError('filler overlaps evidence or target')
    if canvas.pad_id == canvas.mask_id:
        raise ValueError('filler ID cannot be the mask ID')


def evidence_only(canvas):
    """Remove ONLY explicitly marked filler tokens, preserving all public content."""
    _validate_canvas(canvas)
    kept = [i for i, filler in enumerate(canvas.filler_mask) if not filler]
    # Prefix counts map old half-open boundaries to their new coordinates.
    boundary = [0]
    for filler in canvas.filler_mask:
        boundary.append(boundary[-1] + int(not filler))
    span = lambda pair: (boundary[pair[0]], boundary[pair[1]])
    transformed = replace(canvas,
        input_ids=tuple(canvas.input_ids[i] for i in kept),
        attention_mask=tuple(canvas.attention_mask[i] for i in kept),
        target_mask=tuple(canvas.target_mask[i] for i in kept),
        filler_mask=(False,) * len(kept),
        answer_spans=tuple(span(pair) for pair in canvas.answer_spans),
        evidence_span=span(canvas.evidence_span),
        length_kind='evidence_only_remove_marked_filler')
    _validate_canvas(transformed)
    return transformed


def no_evidence(canvas):
    """Replace the entire evidence payload with existing neutral filler, same length.

    The query and answer mask are unchanged. The evidence span includes the task
    label as well as records; its original coordinates remain in audit metadata.
    """
    _validate_canvas(canvas)
    lo, hi = canvas.evidence_span
    ids = list(canvas.input_ids)
    ids[lo:hi] = [canvas.pad_id] * (hi-lo)
    transformed = replace(canvas, input_ids=tuple(ids),
                          length_kind='full_length_evidence_replaced_with_filler')
    _validate_canvas(transformed)
    return transformed


def _record(kind, canvas, task, provenance, **parameters):
    _validate_canvas(canvas)
    result = dict(schema='lrwkv_e3_control_v1', kind=kind, status='ok',
                  provenance=provenance, parameters=parameters,
                  decoder_input=dict(input_ids=list(canvas.input_ids),
                                     target_span=[canvas.target_mask.index(True), len(canvas.input_ids)]),
                  target_mask=list(canvas.target_mask), evidence_span=list(canvas.evidence_span),
                  answer_spans=[list(s) for s in canvas.answer_spans],
                  queries=list(task.queries), gold_answers=list(task.answers),
                  answer_ids=[list(ids) for ids in canvas.answer_ids],
                  input_ids_sha256=G.ids_sha256(canvas.input_ids))
    result['transformation_sha256'] = _hash(result)
    return result


def transform_task(task, canvas, token_prompt, encoder, *, position, provenance, anchors=(4096, 8192)):
    """Build controls on one already verified task, never redrawing salt or gold."""
    logical_hash = _hash(dict(public_prompt=task.public_prompt, queries=list(task.queries),
                              answers=list(task.answers)))
    provenance = dict(provenance, logical_task_sha256=logical_hash)
    rows = [_record('evidence_only', evidence_only(canvas), task, provenance,
                    operation='delete canvas.filler_mask positions only'),
            _record('no_evidence', no_evidence(canvas), task, provenance,
                    operation='replace evidence_span with original filler ID', filler_id=canvas.pad_id)]
    for length in anchors:
        try:
            # Crucial: no G.build_instance at the new length; that changes cell salt.
            anchor = token_prompt.render_token_prompt(encoder, task, target_tokens=length,
                                                       position_fraction=position, mask_id=canvas.mask_id)
        except token_prompt.TokenPromptRefusal as exc:
            row = dict(schema='lrwkv_e3_control_v1', kind='paired_anchor', status='unsupported',
                       provenance=provenance, parameters=dict(target_tokens=length, position=position),
                       unsupported_reason=str(exc))
            row['transformation_sha256'] = _hash(row)
        else:
            if not token_prompt.validate_rendered_prompt(encoder, task, anchor):
                raise ValueError('anchor no longer represents the original task')
            row = _record('paired_anchor', anchor, task, provenance,
                          target_tokens=length, position=position,
                          operation='rerender same logical task without new draw')
        rows.append(row)
    return rows


def build_controls(source_path: Path, data_seed: int, instance_index: int, *, anchors=(4096, 8192)):
    """Rebuild a dev instance from banked bytes, validate prompt/gold, then transform.

    Returns model-free dictionaries. A changed prompt hash, gold, query, span or
    seed identity is an error, not an unsupported control. Only renderer refusal
    at a new length is an explicitly unsupported anchor.
    """
    source_path = Path(source_path)
    content = source_path.read_bytes()
    bank = json.loads(content)
    if bank['status'] != 'ok':
        raise ValueError('banked cell is not supported')
    hop, tp, registry = G._import_generator()
    if data_seed not in hop.SPLIT_DATA_SEEDS['dev']:
        raise ValueError('competence controls require a dev data seed')
    hits = [r for r in bank['instances']
            if r['data_seed'] == data_seed and r['instance_index'] == instance_index]
    if len(hits) != 1:
        raise ValueError('banked instance identity is missing or duplicated')
    inst = hits[0]
    fraction = bank['evidence_position_fraction'] * 100
    if not float(fraction).is_integer():
        raise ValueError('invalid evidence position')
    cell = G.Cell(bank['family'], bank['history_length'], int(fraction),
                  bank['binding_load'], bank['distractors'])
    if bank['cell_id'] != cell.cell_id or inst['cell_id'] != cell.cell_id:
        raise ValueError('banked cell identity disagrees with coordinates')
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    pools = G._PermutingPoolFactory(hop._Pool)
    task, canvas, reason = G.build_instance(hop, tp, encoder, cell, data_seed,
                                           instance_index, pools, 'dev')
    if reason:
        raise ValueError('banked instance cannot rebuild: ' + reason)
    if G.ids_sha256(canvas.input_ids) != inst['input_ids_sha256']:
        raise ValueError('banked input hash mismatch')
    checks = dict(answers=list(task.answers), queries=list(task.queries),
                  realized_tokens=len(canvas.input_ids), metric=task.metric,
                  generator_seed=hop.generator_seed('dev', data_seed, instance_index),
                  target_span=[canvas.target_mask.index(True), len(canvas.input_ids)])
    if any(inst.get(k) != value for k, value in checks.items()):
        raise ValueError('banked gold/query/span/seed metadata mismatch')
    provenance = dict(source_path=str(source_path), source_sha256=hashlib.sha256(content).hexdigest(),
                      source_input_ids_sha256=inst['input_ids_sha256'], cell_id=cell.cell_id,
                      data_seed=data_seed, instance_index=instance_index, split='dev')
    return transform_task(task, canvas, tp, encoder, position=cell.position,
                          provenance=provenance, anchors=anchors)
