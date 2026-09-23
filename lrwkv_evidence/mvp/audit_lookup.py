"""Read-only GPU-artifact audit; writes only a separate local portable report.

Reconstructs every pilot input, decodes every output, checks model/source/vocab
hashes and rank assignment, and reports the frozen gate with Wilson intervals.
No inference, prompt tuning, expansion, or modification of source receipts.
"""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
from . import lookup_pilot as P


def wilson(successes, total, z=1.959963984540054):
    if not 0 <= successes <= total or total <= 0:
        raise ValueError('invalid binomial counts')
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def mask_constant(source):
    tree = ast.parse(Path(source).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MASK_TOKEN_ID' for t in node.targets):
            value = ast.literal_eval(node.value)
            if type(value) is not int:
                break
            return value
    raise ValueError('cannot recover the audited model mask constant')


def validate_item(row, item, ordinal, encoder, mask_id, provenance):
    model = row.get('model')
    rank = ordinal % 4 + (0 if model == 'f2' else 4)
    canvas, span = P.canvas_for(encoder, item['prompt'], mask_id)
    if (model not in P.CONTRACT['models'] or row.get('rank') != rank
        or row.get('scope') != P.CONTRACT['scope'] or row.get('status') != 'ok'
        or row.get('item_id') != item['item_id'] or row.get('depth') != item['depth']
        or row.get('gold') != item['gold'] or row.get('prompt') != item['prompt']
        or row.get('prompt_sha256') != P.U.digest(item['prompt'])
        or row.get('input_ids_sha256') != P.G.ids_sha256(canvas)
        or row.get('target_span') != list(span) or row.get('prompt_tokens') != span[0]
        or row.get('total_tokens') != len(canvas) or row.get('padding_tokens') != 0
        or row.get('decode_seed') != 17 + ordinal
        or row.get('provenance_sha256') != P.U.digest(provenance)):
        raise ValueError(f'input/item/rank/provenance mismatch: {item["item_id"]}')
    output = row.get('output_ids')
    if not isinstance(output, list) or len(output) != 32 or any(type(t) is not int or not 0 <= t < 65536 for t in output):
        raise ValueError('invalid output token IDs or budget')
    if model == 'f2' and any(t in (0, mask_id) for t in output):
        raise ValueError('F2 output contains a forbidden pad or mask token')
    if encoder.decode_sample_ids(output) != row.get('text'):
        raise ValueError('text is not the native decoding of recorded output IDs')
    parsed = P.parse_answer(row['text'])
    if (row.get('prediction') != parsed or row.get('correct') is not (parsed == item['gold'])
        or row.get('substring_correct') is not P.substring_correct(row['text'], item['gold'])):
        raise ValueError('parser/substring result mismatch')
    check = row.get('tokenizer_check', {})
    if check.get('native_decode_equal') is not True or check.get('prefix_ids_sha256') != P.G.ids_sha256(canvas[:span[0]]):
        raise ValueError('missing native-tokenizer prefix verification')
    calls = row.get('actual_nfe')
    if type(calls) is not int or not 1 <= calls <= (8 if model == 'f2' else 32):
        raise ValueError('invalid network evaluation count')
    if model == 'r0' and calls != 32:
        raise ValueError('causal fixed-budget generation did not execute 32 calls')
    if model == 'f2':
        trace = row.get('commits_per_step')
        if not isinstance(trace, list) or len(trace) != calls or any(type(n) is not int or n < 0 for n in trace) or sum(trace) != 32:
            raise ValueError('invalid F2 reveal trace')
    for name in ('wall_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes'):
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid timing/memory metric')


def audit(plan, *, model_dir, checkpoint):
    out = Path(plan['out'])
    items = P.make_items()
    if plan.get('bundle', {}).get('contract') != P.CONTRACT or plan['bundle'].get('items') != items:
        raise ValueError('submission contract/items differ from the frozen pilot')
    source_cache = {}
    def checked_hash(path, expected):
        name = str(path)
        if name not in source_cache:
            source_cache[name] = P.U.file_sha(path)
        if source_cache[name] != expected:
            raise ValueError(f'artifact hash mismatch: {path}')
    manifests = [json.loads((out / f'provenance_rank{r}.json').read_text()) for r in range(8)]
    for rank, prov in enumerate(manifests):
        model = 'f2' if rank < 4 else 'r0'
        if (prov.get('rank') != rank or prov.get('model') != model or prov.get('contract') != P.CONTRACT
            or prov.get('scope') != P.CONTRACT['scope'] or prov.get('items_sha256') != P.U.digest(items)):
            raise ValueError('invalid worker manifest')
        for source, sha in prov['source_sha256'].items():
            checked_hash(Path(source), sha)
        checked_hash(model_dir / 'config.json', prov['model_config_sha256'])
        checked_hash(P.G.VOCAB, prov['tokenizer']['vocab_sha256'])
        paths = {'model.pt': checkpoint / 'model.pt'} if model == 'f2' else {p.name: p for p in model_dir.glob('*.safetensors')}
        if not paths or set(paths) != set(prov['weights']):
            raise ValueError('weight artifact inventory mismatch')
        for name, path in paths.items():
            checked_hash(path, prov['weights'][name])
    for ranks in (range(4), range(4, 8)):
        identities = [{k: v for k, v in manifests[r].items() if k != 'rank'} for r in ranks]
        if any(p != identities[0] for p in identities[1:]):
            raise ValueError('inconsistent provenance within a model')
    # Verify staged source identity in addition to the runtime manifests.
    stage = Path(plan['stage'])
    for module, relative in ((P, 'lrwkv_evidence/mvp/lookup_pilot.py'),
                             (P.U, 'lrwkv_evidence/e3/gpu_runner.py'),
                             (P.A, 'lrwkv_evidence/e3/causal_runner.py'),
                             (P.G, 'lrwkv_evidence/e3/grid324.py')):
        checked_hash(Path(module.__file__), plan['bundle']['files'][relative])
    for rel, sha in plan['bundle']['files'].items():
        checked_hash(stage / rel, sha)
    model_source = next(Path(p) for p in manifests[0]['source_sha256'] if p.endswith('/models/birwkv7_diffusion.py'))
    mask_id = mask_constant(model_source)
    if mask_id != 65535:
        raise ValueError('pilot causal mask placeholder differs from the F2 source')
    _, _, registry = P.G._import_generator()
    encoder = registry.TrieEncoder.from_vocab(str(P.G.VOCAB))
    rows = []
    for model in P.CONTRACT['models']:
        paths = sorted((out / model).glob('*.json'))
        if len(paths) != 240:
            raise ValueError(f'incomplete/extra model receipts: {model} has {len(paths)}/240')
        for ordinal, item in enumerate(items):
            path = out / model / f"{item['item_id']}.json"
            row = json.loads(path.read_text())
            rank = ordinal % 4 + (0 if model == 'f2' else 4)
            if model == 'r0' and any(t not in encoder._tokenizer.idx2token for t in row.get('output_ids', [])):
                raise ValueError('R0 output includes a token outside the qualified native vocabulary')
            validate_item(row, item, ordinal, encoder, mask_id, manifests[rank])
            rows.append(row)
    report = P.gate_summary(rows)
    for group in report['groups']:
        group['exact_wilson95'] = wilson(group['correct'], group['n'])
        group['substring_accuracy'] = group['substring_correct'] / group['n']
        group['substring_wilson95'] = wilson(group['substring_correct'], group['n'])
    report.update({'audit': 'all480canonicalinputs,decodedoutputs,parsers,ranks,networkcalls,source/weight/config/vocab hashes verified',
        'inference_scope': 'fixed balanced synthetic dev tasks; Wilson intervals descriptive and not adjusted for four gates',
        'artifact_hashes': source_cache, 'records': len(rows), 'source_out': str(out)})
    portable = {'schema': 1, 'contract': P.CONTRACT, 'items': items, 'worker_manifests': manifests, 'records': rows}
    return report, portable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, default=Path('results/mvp_lookup_submission_plan.json'))
    ap.add_argument('--out', type=Path, default=Path('results/mvp_lookup_audit'))
    ap.add_argument('--model-dir', type=Path, default=P.G.G / 'models/RWKV7-Goose-World3-2.9B-HF')
    ap.add_argument('--checkpoint', type=Path, default=P.G.G / 'research/lacesmm_assets/birwkv_f2_step14000')
    args = ap.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    report, portable = audit(plan, model_dir=args.model_dir, checkpoint=args.checkpoint)
    payload = gzip.compress(json.dumps(portable, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(), mtime=0)
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / 'portable_records.json.gz'
    archive.write_bytes(payload)
    report['portable_archive'] = {'file': archive.name, 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    report['submission_plan_sha256'] = P.U.file_sha(args.plan)
    P.U.atomic_json(args.out / 'audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'artifact_hashes'}, indent=2))


if __name__ == '__main__':
    main()
