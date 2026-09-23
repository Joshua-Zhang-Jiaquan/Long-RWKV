"""Released RWKV7 2.9B comparator: greedy fixed-length suffix infilling.

Both arms receive the same visible prefix and supplied target length. The causal
arm recomputes its entire prefix (use_cache=False) for each generated token;
this is a quality comparator, not an optimized AR throughput benchmark.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from . import grid324 as G, dev_grid as D, lane as L, runner as R
from .gpu_runner import atomic_json, digest, file_sha, verified_instance, pin_provenance

POLICY = {'name': 'greedy_legal_native_rwkv_fixed_suffix_v1', 'temperature': 0,
          'use_cache': False, 'supplied_target_length': True,
          'input_policy': 'native_rwkv_ids_no_chat_template',
          'access_mode': 'full_visible_prefix', 'seed': 0}


def qualify_tokenizer(encoder, tokenizer):
    """Byte-level mapping equality avoids UTF-8 replacement hiding wrong IDs."""
    native = getattr(tokenizer, 'trie_tokenizer', None)
    if native is None:
        raise ValueError('released RWKV native trie tokenizer required')
    left, right = encoder._tokenizer.idx2token, native.idx2token
    if left != right:
        raise ValueError('model tokenizer byte-to-ID mapping differs from G.VOCAB')
    legal = sorted(int(i) for i in left)
    return native, legal, {'policy': 'exact_native_byte_id_map', 'tokens': len(legal),
        'vocab_sha256': file_sha(G.VOCAB)}


def verify_prefix(ids, encoder, native, wrapper):
    text = encoder.decode_ids(ids)
    if native.decode([list(ids)])[0] != text:
        raise ValueError('native tokenizer decoded different banked prefix bytes')
    # The renderer deliberately repeats single-token filler. Its native IDs can
    # be a noncanonical segmentation of exactly the same decoded text; encoding
    # that text again would change the frozen input and is never used here.
    # This diagnostic is explicitly not used to retokenize or alter the model input.
    return {'native_decode_equal': True,
            'native_roundtrip_exact': list(native.encode(text)[0]) == list(ids),
            'hf_wrapper_roundtrip_exact': list(wrapper.encode(text, add_special_tokens=False)) == list(ids),
            'prefix_ids_sha256': G.ids_sha256(ids)}


def greedy_suffix(model, ids, span, legal_ids):
    import torch
    lo, hi = span
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < lo < hi == ids.shape[1]:
        raise ValueError('only a nonempty suffix target with a visible prefix is supported')
    cur = ids[:, :lo].clone()  # gold target tokens never reach the model
    tokens = []
    legal = torch.tensor(legal_ids, dtype=torch.long, device=ids.device)
    with torch.inference_mode():
        for _ in range(hi - lo):
            result = model(input_ids=cur, use_cache=False)
            logits = result.logits[0, -1].float()
            candidates = logits[legal]
            if not bool(torch.isfinite(candidates).all()):
                raise ValueError('nonfinite legal-token logits')
            token = legal[candidates.argmax()].reshape(1, 1)
            tokens.append(int(token.item()))
            # Do not retain the previous full-canvas vocabulary tensor during
            # the next forward (8 GiB per BF16 64K canvas at this vocabulary).
            del result, logits, candidates
            cur = torch.cat([cur, token], dim=1)
    return tokens, len(tokens)


def require_qualification(receipt, provenance):
    if receipt.get('qualified') is not True or receipt.get('n') != 72 or receipt.get('policy') != POLICY:
        raise ValueError('confirmation requires complete dev qualification under the prespecified policy')
    if receipt.get('model_identity') != provenance['model_identity']:
        raise ValueError('dev-qualified model/tokenizer differs from confirmation model')
    if not receipt.get('dependencies') or receipt.get('dependencies') != provenance.get('dependencies'):
        raise ValueError('dev-qualified dependency sources differ from confirmation')
    if receipt.get('runner_sha256') != file_sha(Path(__file__)):
        raise ValueError('dev-qualified runner source differs from confirmation runner')


def collect_dev(out, panel):
    manifest = json.loads((panel / 'dev_manifest.json').read_text())
    if manifest.get('split') != 'dev' or manifest.get('instances') != 72:
        raise ValueError('qualification requires the complete 72-example dev panel')
    rows = []
    for cell in D.panel_cells():
        for inst in L.instances_of(L.load_cell_record(panel / f'{cell.cell_id}.json')):
            if inst.data_seed not in D.dev_seeds():
                raise ValueError('invalid dev split seed')
            path = out / f'{cell.cell_id}__{inst.data_seed}__{inst.instance_index}.json'
            if not path.exists():
                raise ValueError(f'incomplete dev qualification: {path}')
            row = json.loads(path.read_text())
            if (row.get('status') != 'ok' or row.get('split') != 'dev'
                or row.get('input_ids_sha256') != inst.input_ids_sha256
                or row.get('policy') != POLICY or row.get('prompt_hash_verified') is not True
                or row.get('tokenizer_check', {}).get('native_decode_equal') is not True
                or type(row.get('correct')) is not bool
                or row.get('item_id') != f'{cell.cell_id}__{inst.data_seed}__{inst.instance_index}'
                or row.get('cell_id') != cell.cell_id or row.get('data_seed') != inst.data_seed
                or row.get('instance_index') != inst.instance_index
                or row.get('actual_nfe') != inst.target_span[1] - inst.target_span[0]
                or row.get('wall_seconds', -1) < 0
                or not math.isfinite(row.get('wall_seconds', float('nan')))):
                raise ValueError(f'invalid dev qualification receipt: {path}')
            rows.append(row)
    if len(rows) != 72 or len({r['item_id'] for r in rows}) != 72:
        raise ValueError('missing or duplicate dev records')
    hashes = {r['provenance_sha256'] for r in rows}
    manifests = [json.loads(p.read_text()) for p in out.glob('provenance_rank*.json')]
    if len(hashes) != 1 or not manifests or any(digest(p) not in hashes for p in manifests):
        raise ValueError('mixed or missing dev provenance')
    p = manifests[0]
    if p['panel_sha256'] != digest({f.name: file_sha(f) for f in sorted(panel.glob('*.json'))}):
        raise ValueError('dev panel changed after measurement')
    receipt = {'qualified': True, 'n': 72, 'policy': POLICY,
        'accuracy': sum(r['correct'] for r in rows) / 72,
        'model_identity': p['model_identity'], 'dependencies': p['dependencies'], 'runner_sha256': file_sha(Path(__file__)),
        'provenance_sha256': next(iter(hashes)),
        'meaning': 'execution and tokenizer qualification only; no accuracy threshold or tuning'}
    atomic_json(out / 'dev_qualification.json', receipt)
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['dev', 'confirm'], required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--scale', type=Path, default=G.G / 'qz_stage_traj4096_v7/scale')
    ap.add_argument('--qualification', type=Path)
    ap.add_argument('--length', type=int, choices=[16384, 32768, 65536], help='confirmation length shard')
    ap.add_argument('--cpu-threads', type=int, default=4)
    ap.add_argument('--max-items', type=int, default=0)
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args(argv)
    if args.collect:
        if args.mode != 'dev':
            ap.error('only dev qualification can be collected here')
        print(json.dumps(collect_dev(args.out, args.panel), indent=2))
        return
    if not args.model_dir:
        ap.error('--model-dir is required')
    if args.mode == 'confirm' and not args.qualification:
        ap.error('--qualification is required before confirmation')
    if args.mode == 'dev' and args.length is not None:
        ap.error('--length is for confirmation only; dev always uses all 72 items')
    cfg = json.loads((args.model_dir / 'config.json').read_text())
    if cfg.get('architectures') != ['RWKV7ForCausalLM'] or cfg.get('hidden_size') != 2560:
        raise ValueError('this runner is restricted to released RWKV7 2.9B geometry')
    split = 'dev' if args.mode == 'dev' else G.GRID_SPLIT
    manifest_name = 'dev_manifest.json' if args.mode == 'dev' else 'grid_manifest.json'
    manifest = json.loads((args.panel / manifest_name).read_text())
    if manifest.get('split') != split or manifest.get('partial_run', False):
        raise ValueError('wrong or partial panel')
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    rank, world, local = (int(os.environ.get(k, d)) for k, d in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if world not in (1, 8):
        raise ValueError('use one or eight GPU workers')
    torch.cuda.set_device(local)
    sys.path.insert(0, str(args.scale))
    from eval.capability.hf_causal_model import load_hf_causal
    print(f'rank={rank} loading released RWKV7 cpu_threads={args.cpu_threads}', flush=True)
    start = time.perf_counter()
    loaded = load_hf_causal(str(args.model_dir), device=f'cuda:{local}')
    print(f'rank={rank} load_seconds={time.perf_counter()-start:.3f}', flush=True)
    hop, renderer, registry = G._import_generator()
    G.register_grid_split(hop)
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    native, legal, token_manifest = qualify_tokenizer(encoder, loaded.tokenizer)
    model_files = sorted(set(args.model_dir.glob('*.safetensors')) | set(args.model_dir.glob('*.json')) | set(args.model_dir.glob('*.py')))
    if not any(p.suffix == '.safetensors' for p in model_files):
        raise ValueError('no released model safetensors found')
    identity = {'model_dir': str(args.model_dir.resolve()),
                'files': {p.name: file_sha(p) for p in model_files}, 'tokenizer': token_manifest}
    provenance = {'mode': args.mode, 'split': split, 'policy': POLICY, 'model_identity': identity,
        'runner_sha256': file_sha(Path(__file__)), 'length_filter': args.length,
        'dependencies': {str(p): file_sha(p) for p in [Path(G.__file__), Path(R.__file__), Path(D.__file__), Path(L.__file__),
             Path(hop.__file__), Path(renderer.__file__), Path(registry.__file__), Path(__file__).with_name('gpu_runner.py'),
             args.scale / 'eval/capability/hf_causal_model.py']},
        'panel_sha256': digest({p.name: file_sha(p) for p in sorted(args.panel.glob('*.json'))}),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cpu_threads': args.cpu_threads, 'device_name': torch.cuda.get_device_name(),
        'runtime_env': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}
    if args.mode == 'confirm':
        require_qualification(json.loads(args.qualification.read_text()), provenance)
        provenance['qualification_sha256'] = file_sha(args.qualification)
    phash = digest(provenance)
    pin_provenance(args.out / f'provenance_rank{rank}.json', provenance)
    pools = G._PermutingPoolFactory(hop._Pool)
    ordinal = measured = 0
    for cell in D.panel_cells() if args.mode == 'dev' else G.all_cells():
        if args.length is not None and cell.length != args.length:
            continue
        path = args.panel / f'{cell.cell_id}.json'
        rec = json.loads(path.read_text())
        if rec['status'] != G.STATUS_OK:
            if args.mode == 'dev':
                raise ValueError('unsupported dev cell')
            if rank == 0:
                atomic_json(args.out / 'unsupported' / path.name, rec)
            continue
        for inst in L.instances_of(L.load_cell_record(path)):
            assigned = ordinal % world == rank
            ordinal += 1
            if not assigned:
                continue
            item_id = f'{cell.cell_id}__{inst.data_seed}__{inst.instance_index}'
            dest = args.out / f'{item_id}.json'
            if dest.exists():
                old = json.loads(dest.read_text())
                if (old.get('provenance_sha256') != phash or old.get('input_ids_sha256') != inst.input_ids_sha256
                    or old.get('status') != 'ok' or old.get('policy') != POLICY
                    or old.get('split') != split or old.get('item_id') != item_id
                    or old.get('prompt_hash_verified') is not True
                    or old.get('tokenizer_check', {}).get('native_decode_equal') is not True
                    or type(old.get('correct')) is not bool
                    or old.get('actual_nfe') != inst.target_span[1] - inst.target_span[0]):
                    raise ValueError('stale resume receipt')
                continue
            if args.max_items and measured >= args.max_items:
                return
            task, canvas = verified_instance(hop, renderer, encoder, pools, cell, inst, split)
            check = verify_prefix(canvas.input_ids[:inst.target_span[0]], encoder, native, loaded.tokenizer)
            ids = torch.tensor([canvas.input_ids], dtype=torch.long, device=f'cuda:{local}')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            tokens, calls = greedy_suffix(loaded.model, ids, inst.target_span, legal)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            text = encoder.decode_sample_ids(tokens)
            answer = R.extract_answer(text, task.request.family)
            correct = len(inst.answers) == 1 and answer == inst.answers[0]
            atomic_json(dest, {'status': 'ok', 'split': split, 'policy': POLICY,
                'item_id': item_id, 'cell_id': cell.cell_id, 'data_seed': inst.data_seed,
                'instance_index': inst.instance_index, 'input_ids_sha256': inst.input_ids_sha256,
                'prompt_hash_verified': True, 'tokenizer_check': check,
                'provenance_sha256': phash, 'correct': correct, 'joint_exact': correct,
                'parse_ok': answer is not None, 'prediction': answer, 'gold': inst.answers,
                'text': text, 'output_ids': tokens, 'actual_nfe': calls,
                'wall_seconds': seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'rank': rank, 'device': torch.cuda.get_device_name()})
            measured += 1
            print(f'{item_id} nfe={calls} seconds={seconds:.3f}', flush=True)


if __name__ == '__main__':
    main()
