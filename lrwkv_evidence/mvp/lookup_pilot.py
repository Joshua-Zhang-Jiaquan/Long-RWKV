"""Fixed natural-language lookup pilot, gated before long-context expansion.

One predeclared chat format, fresh deterministic dev tasks, and 32 output tokens
for both frozen models. This is a competence screen, not primary confirmation.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from lrwkv_evidence.e3 import grid324 as G, gpu_runner as U, causal_runner as A

COLORS = ('red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown',
          'black', 'white', 'gray', 'cyan', 'teal', 'amber', 'gold', 'silver',
          'beige', 'navy', 'maroon', 'coral', 'olive', 'ivory', 'violet', 'indigo')
FIRST = ('Alice', 'Brian', 'Carla', 'Daniel', 'Elena', 'Frank', 'Grace', 'Henry',
         'Iris', 'James', 'Karen', 'Leo', 'Maya', 'Nora', 'Oscar', 'Paula')
LAST = ('Adams', 'Baker', 'Clark', 'Davis', 'Evans', 'Foster', 'Grant', 'Hayes')
CONFIG = {'sampler': 'matched_bernoulli', 'steps': 8, 'tau': 0.7, 'commit_order': 'confidence'}
CONTRACT = {'schema': 1, 'scope': 'fresh_dev_competence_gate_not_confirmation',
    'data_seed': 2027092201, 'decode_seed': 17, 'items_per_depth': 120, 'depths': [1, 2],
    'target_budget': 32, 'distractor_entities': 7, 'gold_balance': 'exactly five examples per color per depth', 'gate_threshold': 0.8,
    'gate_required_correct_per_depth': 96, 'models': ['f2', 'r0'],
    'gate_rule': 'each model independently needs >=96 exact first-line answers at each depth; no long-context expansion unless both pass',
    'parser': 'first nonempty line; casefold; optional Answer: prefix; strip terminal period/exclamation and enclosing quotes; otherwise exact one declared color word',
    'substring_diagnostic': 'case-insensitive whole-word gold occurrence anywhere in full output; never used for gate',
    'f2_config': CONFIG, 'r0_policy': 'native legal-token greedy; 32 steps; use_cache=False'}
TEMPLATE = ('User: Use only the facts below to answer the question. Reply with exactly one color word on the first line. Do not explain.\n'
            'Facts:\n{facts}\nQuestion: {question}\nAssistant:')


def make_items():
    names = [f'{first} {last}' for first in FIRST for last in LAST]
    result = []
    for depth in CONTRACT['depths']:
        for index in range(CONTRACT['items_per_depth']):
            rng = random.Random(CONTRACT['data_seed'] + depth * 100000 + index)
            people = rng.sample(names, 16)
            gold = COLORS[index % len(COLORS)]
            colors = rng.sample([c for c in COLORS if c != gold], 7)
            target = rng.randrange(8)
            colors.insert(target, gold)
            if depth == 1:
                facts = [f"{people[i]}'s favorite color is {colors[i]}." for i in range(8)]
                question = f"What is {people[target]}'s favorite color?"
            else:
                facts = [f"{people[i]}'s assigned guide is {people[i + 8]}." for i in range(8)]
                facts += [f"{people[i + 8]}'s favorite color is {colors[i]}." for i in range(8)]
                question = f"What is the favorite color of {people[target]}'s assigned guide?"
            rng.shuffle(facts)
            prompt = TEMPLATE.format(facts='\n'.join(facts), question=question)
            result.append({'item_id': f'natural_lookup_dev_d{depth}_{index:03d}',
                'depth': depth, 'index': index, 'prompt': prompt, 'gold': colors[target],
                'logical_facts': facts, 'question': question})
    if len({x['prompt'] for x in result}) != 240:
        raise ValueError('pilot prompts are not unique')
    return result


def parse_answer(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    line = re.sub(r'^answer\s*:\s*', '', lines[0], flags=re.IGNORECASE).strip().casefold()
    line = line.rstrip('.!').strip().strip('"\'').strip()
    return line if line in COLORS else None


def substring_correct(text, gold):
    return re.search(r'(?<!\w)' + re.escape(gold) + r'(?!\w)', text, flags=re.IGNORECASE) is not None


def canvas_for(encoder, prompt, mask_id):
    ids = list(encoder.encode_text(prompt))
    lo = len(ids)
    if not lo:
        raise ValueError('empty prompt')
    return ids + [mask_id] * CONTRACT['target_budget'], (lo, lo + CONTRACT['target_budget'])


def gate_summary(rows):
    expected = {(model, item['item_id']): item for model in CONTRACT['models'] for item in make_items()}
    seen = {}
    for row in rows:
        key = row.get('model'), row.get('item_id')
        if key not in expected or key in seen:
            raise ValueError('unknown or duplicate pilot item')
        item = expected[key]
        prediction = parse_answer(row.get('text', ''))
        correct = prediction == item['gold']
        if (row.get('status') != 'ok' or row.get('gold') != item['gold'] or row.get('depth') != item['depth']
            or row.get('prediction') != prediction or row.get('correct') is not correct
            or row.get('substring_correct') is not substring_correct(row.get('text', ''), item['gold'])
            or row.get('prompt_sha256') != U.digest(item['prompt'])
            or not isinstance(row.get('output_ids'), list) or len(row['output_ids']) != 32
            or not math.isfinite(row.get('wall_seconds', float('nan'))) or row['wall_seconds'] < 0):
            raise ValueError('invalid pilot receipt')
        seen[key] = row
    if set(seen) != set(expected):
        raise ValueError(f'incomplete pilot: {len(seen)}/{len(expected)}')
    groups = []
    for model in CONTRACT['models']:
        for depth in CONTRACT['depths']:
            subset = [r for (m, _), r in seen.items() if m == model and r['depth'] == depth]
            correct = sum(r['correct'] for r in subset)
            groups.append({'model': model, 'depth': depth, 'n': len(subset), 'correct': correct,
                'accuracy': correct / len(subset), 'substring_correct': sum(r['substring_correct'] for r in subset),
                'passed': correct >= CONTRACT['gate_required_correct_per_depth']})
    return {'scope': CONTRACT['scope'], 'complete': True, 'contract': CONTRACT,
            'groups': groups, 'gate_passed': all(g['passed'] for g in groups),
            'long_context_expansion_allowed': all(g['passed'] for g in groups)}


def collect(out):
    manifests = [json.loads((out / f'provenance_rank{rank}.json').read_text()) for rank in range(8)]
    if any(p.get('contract') != CONTRACT or p.get('items_sha256') != U.digest(make_items()) for p in manifests):
        raise ValueError('unrecognized pilot contract/provenance')
    for ranks in (range(4), range(4, 8)):
        identities = [{k: v for k, v in manifests[r].items() if k != 'rank'} for r in ranks]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError('model/source/runtime provenance differs across model workers')
    rows = []
    for rank, manifest in enumerate(manifests):
        if manifest.get('rank') != rank or manifest.get('model') != ('f2' if rank < 4 else 'r0'):
            raise ValueError('wrong model/rank provenance')
        for source, claimed in manifest['source_sha256'].items():
            if U.file_sha(source) != claimed:
                raise ValueError('pilot source changed after execution')
    for model in CONTRACT['models']:
        for path in (out / model).glob('*.json'):
            row = json.loads(path.read_text())
            rank = row.get('rank', -1)
            if not isinstance(rank, int) or not 0 <= rank < 8 or row.get('provenance_sha256') != U.digest(manifests[rank]):
                raise ValueError('receipt provenance mismatch')
            if manifests[rank]['model'] != model or row.get('model') != model:
                raise ValueError('misfiled pilot model')
            rows.append(row)
    result = gate_summary(rows)
    result['provenance_sha256'] = [U.digest(p) for p in manifests]
    U.atomic_json(out / 'competence_gate.json', result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--scale', type=Path, default=G.G / 'qz_stage_traj4096_v7/scale')
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args(argv)
    if args.collect:
        print(json.dumps(collect(args.out), indent=2)); return
    if not args.model_dir or not args.checkpoint:
        ap.error('model-dir and checkpoint required')
    rank, world, local = (int(os.environ.get(k, d)) for k, d in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if world != 8:
        raise ValueError('exactly eight workers required')
    model_kind = 'f2' if rank < 4 else 'r0'
    import torch
    torch.set_num_threads(4); torch.manual_seed(17); torch.cuda.set_device(local)
    device = f'cuda:{local}'
    sys.path.insert(0, str(args.scale))
    print(f'rank={rank} model={model_kind} loading frozen competence pilot', flush=True)
    if model_kind == 'f2':
        from eval.capability.birwkv_diffusion_model import load_birwkv_diffusion
        loaded = load_birwkv_diffusion(str(args.checkpoint), str(args.model_dir), device=device)
    else:
        from eval.capability.hf_causal_model import load_hf_causal
        loaded = load_hf_causal(str(args.model_dir), device=device)
    _, _, registry = G._import_generator()
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    native, legal, token_manifest = A.qualify_tokenizer(encoder, loaded.tokenizer)
    sources = [Path(__file__), Path(U.__file__), Path(A.__file__), Path(G.__file__),
        args.scale / 'eval/capability/birwkv_diffusion_model.py', args.scale / 'eval/capability/hf_causal_model.py',
        args.scale / 'models/birwkv7_diffusion.py']
    import fla.models.rwkv7.modeling_rwkv7 as rwkv_source
    import fla.layers.rwkv7 as layer_source
    sources += [Path(rwkv_source.__file__), Path(layer_source.__file__)]
    items = make_items()
    provenance = {'scope': CONTRACT['scope'], 'contract': CONTRACT, 'items_sha256': U.digest(items),
        'model': model_kind, 'rank': rank, 'seed': 17, 'source_sha256': {str(p): U.file_sha(p) for p in sources},
        'weights': ({'model.pt': U.file_sha(args.checkpoint / 'model.pt')} if model_kind == 'f2'
                    else {p.name: U.file_sha(p) for p in sorted(args.model_dir.glob('*.safetensors'))}),
        'model_config_sha256': U.file_sha(args.model_dir / 'config.json'), 'tokenizer': token_manifest,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
        'execution_env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BIRWKV_META_CONSTRUCT', 'BIRWKV_LOAD_SLOTS', 'LOOP_REPS_OVERRIDE')}}
    path = args.out / f'provenance_rank{rank}.json'
    if path.exists() and json.loads(path.read_text()) != provenance:
        raise ValueError('resume provenance changed')
    U.atomic_json(path, provenance)
    phash = U.digest(provenance)
    for ordinal, item in enumerate(items):
        if ordinal % 4 != rank % 4:
            continue
        dest = args.out / model_kind / f"{item['item_id']}.json"
        if dest.exists():
            old = json.loads(dest.read_text())
            if old.get('provenance_sha256') != phash or old.get('status') != 'ok':
                raise ValueError('stale resume receipt')
            continue
        mask_id = loaded.model.mask_token_id if model_kind == 'f2' else 65535
        ids_list, span = canvas_for(encoder, item['prompt'], mask_id)
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        check = A.verify_prefix(ids_list[:span[0]], encoder, native, loaded.tokenizer)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        if model_kind == 'f2':
            tokens, calls, trace = U.sample_target(loaded.model, ids, span, CONFIG,
                mask_id=loaded.model.mask_token_id, pad_id=loaded.model.pad_token_id,
                generator=torch.Generator(device=device).manual_seed(17 + ordinal))
        else:
            tokens, calls = A.greedy_suffix(loaded.model, ids, span, legal); trace = None
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        text = encoder.decode_sample_ids(tokens); prediction = parse_answer(text)
        U.atomic_json(dest, {'scope': CONTRACT['scope'], 'status': 'ok', 'model': model_kind, 'rank': rank,
            'item_id': item['item_id'], 'depth': item['depth'], 'gold': item['gold'],
            'prompt': item['prompt'], 'prompt_sha256': U.digest(item['prompt']),
            'input_ids_sha256': G.ids_sha256(ids_list), 'target_span': span,
            'prompt_tokens': span[0], 'total_tokens': len(ids_list), 'padding_tokens': 0,
            'provenance_sha256': phash, 'tokenizer_check': check, 'decode_seed': 17 + ordinal,
            'output_ids': tokens, 'text': text, 'prediction': prediction, 'correct': prediction == item['gold'],
            'substring_correct': substring_correct(text, item['gold']), 'actual_nfe': calls,
            'commits_per_step': trace, 'wall_seconds': elapsed,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        print(f"{model_kind} {item['item_id']} parsed={prediction!r} gold={item['gold']} seconds={elapsed:.3f}", flush=True)


if __name__ == '__main__':
    main()
