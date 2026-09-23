"""F2 E3 GPU measurement (torchrun --standalone --nproc_per_node=8 -m ...).

Dev runs all 16 preregistered configurations; confirmation requires a frozen
protocol. Outputs are individual atomic JSON receipts, resumable only under the
same provenance. No GPU or torch import is needed to inspect the run contract.
Matched Bernoulli refers to the reveal law, not an exact joint token sampler:
within one reveal step token predictions factorize across positions. Temperature
0.7 changes the model posterior; both arms exclude mask and padding outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from . import dev_grid as D, grid324 as G, lane as L, runner as R

DEFAULT_SCALE = G.G / 'qz_stage_traj4096_v7' / 'scale'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def pin_provenance(path, provenance):
    """Never destroy the manifest explaining receipts from an earlier attempt."""
    path = Path(path)
    for previous in path.parent.glob('provenance_rank*.json'):
        if json.loads(previous.read_text()) != provenance:
            raise ValueError(f'refusing changed run provenance: {previous}')
    atomic_json(path, provenance)


def validate_resume(row, *, provenance_hash, instance, config, split):
    if (row.get('status') != 'ok' or row.get('provenance_sha256') != provenance_hash
            or row.get('input_ids_sha256') != instance.input_ids_sha256
            or row.get('prompt_hash_verified') is not True
            or row.get('config') != config or row.get('split') != split
            or row.get('cell_id') != instance.cell_id
            or row.get('data_seed') != instance.data_seed
            or row.get('instance_index') != instance.instance_index
            or type(row.get('correct')) is not bool
            or type(row.get('actual_nfe')) is not int
            or not 1 <= row['actual_nfe'] <= config['steps']):
        raise ValueError('refusing incomplete or stale resume receipt')


def configs_for(mode, protocol):
    if mode == 'dev':
        return D.sweep_configs()
    decoder = L.require_frozen_decoder(protocol)
    config = {key: decoder[key] for key in D.DECODER_KEYS}
    if config not in D.sweep_configs():
        raise ValueError('frozen decoder is outside the declared dev sweep')
    if decoder.get('dev_panel') != 'lrwkv_evidence.e3.dev_grid':
        raise ValueError('confirmation requires dev-panel selection provenance')
    return [config]


def item_seed(split, cell_id, data_seed, index, seed=0):
    # Deliberately independent of rank, restart order and selected config.
    return int(digest([split, cell_id, data_seed, index, seed])[:15], 16)


def assigned_rank(item_ordinal, config_ordinal, world):
    # Rotate configurations across workers. Flattened modulo would bind each
    # worker to a single step budget (8 versus 64), leaving seven workers idle
    # while the expensive configuration finished. RNG remains item-owned.
    return (item_ordinal + config_ordinal) % world


def verify_confirmation_freeze(protocol):
    decoder = L.require_frozen_decoder(protocol)
    required = ('dev_measurements_path', 'dev_measurements_sha256',
                'dev_provenance_sha256', 'checkpoint_sha256', 'dev_panel_sha256')
    if any(not decoder.get(key) for key in required):
        raise ValueError('confirmation requires a complete measured calibration freeze')
    path = Path(decoder['dev_measurements_path'])
    if file_sha(path) != decoder['dev_measurements_sha256']:
        raise ValueError('calibration measurements changed after freeze')
    rows = json.loads(path.read_text())
    if (len(rows) != len(D.sweep_configs()) or
            {D.config_id(row) for row in rows} != {D.config_id(row) for row in D.sweep_configs()} or
            any(row.get('n') != 72 for row in rows)):
        raise ValueError('frozen calibration lacks the complete declared sweep')
    selected = D.select_decoder(rows)
    if any(selected[key] != decoder[key] for key in D.DECODER_KEYS):
        raise ValueError('frozen operating point differs from the declared selection rule')
    return decoder


def sample_target(model, ids, span, config, *, mask_id, pad_id, generator,
                  reuse_unchanged_logits=False, repeat_logit_audit=None):
    """Unshifted masked prediction; visible tokens are immutable throughout."""
    import torch
    if config not in D.sweep_configs():
        raise ValueError('unknown decoder configuration')
    lo, hi = span
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 <= lo < hi <= ids.shape[1]:
        raise ValueError('expected one canvas and a nonempty valid target span')
    cur = ids.clone()
    cur[:, lo:hi] = mask_id
    still = torch.ones(hi - lo, dtype=torch.bool, device=ids.device)
    calls = 0
    cached_logits = None
    audited_states = {}
    trace = []
    steps = config['steps']
    with torch.inference_mode():
        for step in range(1, steps + 1):
            if not bool(still.any()):
                break
            # F2 has no timestep argument. A zero-commit stage leaves the entire
            # canvas unchanged; only logits may be reused, never RNG draws.
            if cached_logits is None:
                full_logits = model(cur)
                logits = full_logits[0, lo:hi].float().clone()
                del full_logits
                calls += 1
                logits[:, mask_id] = -torch.inf
                logits[:, pad_id] = -torch.inf
                if not torch.isfinite(logits.logsumexp(-1)).all():
                    raise ValueError('nonfinite target logits')
                if repeat_logit_audit is not None:
                    state = tuple(cur[0, lo:hi].tolist())
                    if state in audited_states:
                        if not torch.equal(logits, audited_states[state]):
                            raise ValueError('identical canvas gave different target logits; reuse refused')
                        repeat_logit_audit['bitwise_equal_repeats'] = repeat_logit_audit.get('bitwise_equal_repeats', 0) + 1
                    else:
                        audited_states[state] = logits.clone()
                if reuse_unchanged_logits:
                    cached_logits = logits
            else:
                logits = cached_logits
            probs = logits.softmax(-1)
            pred = torch.multinomial((logits / config['tau']).softmax(-1), 1,
                                     generator=generator).squeeze(-1)
            if config['sampler'] == 'matched_bernoulli':
                # alpha(t)=1-t: conditional reveal (t-s)/t.
                commit = still & (torch.rand(still.shape, device=ids.device,
                                            generator=generator) < 1 / (steps - step + 1))
            else:
                need = (hi - lo) * step // steps - int((~still).sum())
                conf = probs.gather(-1, pred[:, None]).squeeze(-1).masked_fill(~still, -torch.inf)
                commit = torch.zeros_like(still)
                if need:
                    commit[torch.topk(conf, need).indices] = True
            cur[0, lo:hi][commit] = pred[commit]
            still &= ~commit
            committed = int(commit.sum())
            trace.append(committed)
            if committed:
                cached_logits = None
    if bool(still.any()):
        raise RuntimeError('schedule left target tokens masked')
    return cur[0, lo:hi].tolist(), calls, trace


def qualify_logit_reuse(model, ids, span, config, *, mask_id, pad_id, seed):
    """Real-device replay gate; qualification work is outside item measurements."""
    import torch
    runs = {}
    for enabled in (False, True):
        audit = {} if not enabled else None
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = sample_target(model, ids, span, config, mask_id=mask_id,
            pad_id=pad_id, generator=torch.Generator(device=ids.device).manual_seed(seed),
            reuse_unchanged_logits=enabled, repeat_logit_audit=audit)
        torch.cuda.synchronize()
        runs['cached' if enabled else 'uncached'] = {
            'output_ids': result[0], 'actual_nfe': result[1],
            'commits_per_step': result[2], 'wall_seconds': time.perf_counter() - start,
            'repeat_logit_audit': audit}
    if (runs['cached']['output_ids'] != runs['uncached']['output_ids']
        or runs['cached']['commits_per_step'] != runs['uncached']['commits_per_step']):
        raise ValueError('GPU unchanged-logit replay parity failed; optimization refused')
    return {'qualified': True, 'device': torch.cuda.get_device_name(),
            'config': config, 'rng_seed': seed, 'runs': runs,
            'meaning': 'first-request output and reveal-trace equality, not universal numerical proof'}


def verified_instance(hop, renderer, encoder, pools, cell, inst, split):
    task, canvas, reason = G.build_instance(hop, renderer, encoder, cell,
        inst.data_seed, inst.instance_index, pools, split)
    if reason or G.ids_sha256(canvas.input_ids) != inst.input_ids_sha256:
        raise ValueError(f'{inst.cell_id}: prompt reconstruction mismatch ({reason})')
    actual_span = (canvas.target_mask.index(True), len(canvas.target_mask))
    if not all(canvas.target_mask[actual_span[0]:]):
        raise ValueError('target mask must be a contiguous suffix')
    if len(inst.answers) != 1 or len(inst.queries) != 1:
        raise ValueError('this endpoint requires exactly one query and answer')
    if tuple(task.answers) != inst.answers or tuple(task.queries) != inst.queries or actual_span != inst.target_span:
        raise ValueError(f'{inst.cell_id}: banked gold/query/target differs from reconstruction')
    return task, canvas


def collect_dev(out, panel):
    """Require every expected item of every config before producing freeze input."""
    manifest = json.loads((Path(panel) / 'dev_manifest.json').read_text())
    if (manifest.get('split') != D.DEV_SPLIT or manifest.get('instances') != 72
            or manifest.get('instances_per_seed') != D.DEV_INSTANCES_PER_SEED):
        raise ValueError('selection requires the complete preregistered 72-item dev panel')
    expected = []
    for cell in D.panel_cells():
        rec = L.load_cell_record(Path(panel) / f'{cell.cell_id}.json')
        for inst in L.instances_of(rec):
            if inst.data_seed not in D.dev_seeds():
                raise ValueError('dev seed is not disjoint from confirmation')
            expected.append((cell.cell_id, inst.data_seed, inst.instance_index, inst.input_ids_sha256))
    if len(expected) != 72 or len(set(expected)) != 72:
        raise ValueError('duplicate or missing dev examples')
    rows = []
    provenance = set()
    for config in D.sweep_configs():
        scored = []
        for cell, seed, index, prompt_sha in expected:
            path = Path(out) / D.config_id(config) / f'{cell}__{seed}__{index}.json'
            if not path.exists():
                raise ValueError(f'incomplete dev sweep: missing {path}')
            row = json.loads(path.read_text())
            if row['split'] != D.DEV_SPLIT or row['config'] != config or row['status'] != 'ok':
                raise ValueError(f'invalid dev receipt: {path}')
            if (row.get('input_ids_sha256') != prompt_sha or row.get('prompt_hash_verified') is not True
                    or row.get('cell_id') != cell or row.get('data_seed') != seed
                    or row.get('instance_index') != index or type(row.get('correct')) is not bool
                    or not isinstance(row.get('actual_nfe'), int) or not 1 <= row['actual_nfe'] <= config['steps']
                    or not math.isfinite(row.get('wall_seconds', float('nan')))
                    or row['wall_seconds'] < 0):
                raise ValueError(f'invalid measured outcome: {path}')
            provenance.add(row['provenance_sha256'])
            scored.append(row)
        rows.append({**config, 'accuracy': sum(r['correct'] for r in scored) / len(scored),
                     'n': len(scored), 'actual_nfe': sum(r['actual_nfe'] for r in scored)})
    if len(provenance) != 1:
        raise ValueError('dev receipts combine different run provenance')
    manifests = [json.loads(p.read_text()) for p in Path(out).glob('provenance_rank*.json')]
    if not manifests or any(digest(p) not in provenance for p in manifests):
        raise ValueError('missing or inconsistent run provenance manifests')
    if any(p.get('split') != D.DEV_SPLIT or p.get('manifest_sha256') != file_sha(Path(panel) / 'dev_manifest.json') for p in manifests):
        raise ValueError('provenance is not bound to this dev panel')
    if manifests[0].get('reuse_unchanged_logits'):
        qualified_configs = set()
        parity_hashes = set()
        for rank in range(8):
            for sampler in D.DECODER_GRID['sampler']:
                path = Path(out) / 'cache_parity' / f'rank{rank}_{sampler}.json'
                if not path.is_file():
                    raise ValueError(f'missing GPU cache parity: {path}')
                q = json.loads(path.read_text())
                a, b = q['runs']['uncached'], q['runs']['cached']
                if (q.get('qualified') is not True or q.get('rank') != rank
                    or q.get('provenance_sha256') not in provenance
                    or q['config']['sampler'] != sampler
                    or a['output_ids'] != b['output_ids']
                    or a['commits_per_step'] != b['commits_per_step']
                    or not 0 < b['actual_nfe'] <= a['actual_nfe']):
                    raise ValueError(f'invalid GPU cache parity: {path}')
                qualified_configs.add(D.config_id(q['config']))
                parity_hashes.add(digest(q))
        if qualified_configs != {D.config_id(c) for c in D.sweep_configs()}:
            raise ValueError('GPU cache parity does not cover all 16 configurations')
        for path in Path(out).glob('*/*.json'):
            if path.parent.name == 'cache_parity':
                continue
            row = json.loads(path.read_text())
            if row.get('status') == 'ok' and row.get('logit_reuse_parity_sha256') not in parity_hashes:
                raise ValueError(f'measurement missing cache parity binding: {path}')
    atomic_json(Path(out) / 'dev_measurements.json', rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['dev', 'confirm'], required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--protocol', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--model-dir', type=Path)
    ap.add_argument('--scale', type=Path, default=DEFAULT_SCALE)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--length', type=int, choices=G.LENGTHS,
                    help='one prespecified confirmation length; all cells retained within it')
    ap.add_argument('--max-items', type=int, default=0, help='per-rank smoke cap; cannot qualify selection')
    ap.add_argument('--cpu-threads', type=int, default=4)
    ap.add_argument('--reuse-unchanged-logits', action='store_true',
                    help='reuse logits after zero commits; requires per-worker GPU replay parity')
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args(argv)
    if args.collect:
        if args.mode != 'dev':
            ap.error('--collect only aggregates the dev selection table')
        print(json.dumps(collect_dev(args.out, args.panel), indent=2))
        return
    if not args.checkpoint or not args.model_dir:
        ap.error('--checkpoint and --model-dir are required for GPU runs')
    protocol = json.loads(args.protocol.read_text()) if args.protocol else None
    configs = configs_for(args.mode, protocol)
    if args.mode == 'confirm':
        verify_confirmation_freeze(protocol)
    elif args.length is not None:
        ap.error('--length is only for confirmation; the dev panel is fixed')
    split = D.DEV_SPLIT if args.mode == 'dev' else G.GRID_SPLIT
    manifest_name = 'dev_manifest.json' if args.mode == 'dev' else 'grid_manifest.json'
    manifest = json.loads((args.panel / manifest_name).read_text())
    if manifest.get('split') != split or manifest.get('partial_run', False):
        raise ValueError('panel split mismatch or partial confirmation grid')
    import torch
    torch.set_num_threads(args.cpu_threads)
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world not in (1, 8):
        raise ValueError('use one GPU for debugging or torchrun with eight GPUs')
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    sys.path.insert(0, str(args.scale))
    from eval.capability.birwkv_diffusion_model import load_birwkv_diffusion
    load_start = time.perf_counter()
    print(f'rank={rank} loading checkpoint with {args.cpu_threads} CPU threads', flush=True)
    loaded = load_birwkv_diffusion(str(args.checkpoint), str(args.model_dir), device=device)
    print(f'rank={rank} load_seconds={time.perf_counter() - load_start:.3f}', flush=True)
    model = loaded.model
    hop, renderer, registry = G._import_generator()
    sources = [Path(__file__), Path(D.__file__), Path(G.__file__), Path(R.__file__), Path(L.__file__),
               Path(hop.__file__), Path(renderer.__file__), Path(registry.__file__),
               args.scale / 'eval/capability/birwkv_diffusion_model.py',
               args.scale / 'models/birwkv7_diffusion.py']
    provenance = {'schema': 1, 'mode': args.mode, 'split': split, 'configs': configs,
                  'checkpoint': str(args.checkpoint.resolve()),
                  'checkpoint_sha256': file_sha(args.checkpoint / 'model.pt'),
                  'model_dir': str(args.model_dir.resolve()), 'step': loaded.step,
                  'vocab_sha256': file_sha(G.VOCAB),
                  'model_config_sha256': file_sha(args.model_dir / 'config.json'),
                  'panel_sha256': digest({p.name: file_sha(p) for p in sorted(args.panel.glob('*.json'))}),
                  'seed': args.seed, 'length_filter': args.length,
                  'manifest_sha256': file_sha(args.panel / manifest_name),
                  'sources': {str(p): file_sha(p) for p in sources},
                  'protocol_sha256': file_sha(args.protocol) if args.protocol else None,
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'cpu_threads': args.cpu_threads,
                  'execution_env': {k: os.environ.get(k) for k in (
                      'LOOP_REPS_OVERRIDE', 'BIRWKV_META_CONSTRUCT', 'BIRWKV_DIRECT_LOAD',
                      'BIRWKV_META_LOAD', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                  'mask_id': model.mask_token_id, 'pad_id': model.pad_token_id,
                  'device_name': torch.cuda.get_device_name(),
                  'tokenizer_files': {p.name: file_sha(p) for p in sorted(args.model_dir.iterdir())
                      if p.is_file() and (p.name.startswith('tokeniz') or p.name.startswith('rwkv_vocab')
                                         or p.name == 'special_tokens_map.json')},
                  'reuse_unchanged_logits': args.reuse_unchanged_logits,
                  'head': 'unmodified_full_logits_target_slice', 'access_mode': 'full_canvas'}
    provenance_hash = digest(provenance)
    if args.mode == 'confirm' and provenance['checkpoint_sha256'] != protocol['quality']['decoder']['checkpoint_sha256']:
        raise ValueError('confirmation checkpoint differs from calibrated checkpoint')
    pin_provenance(args.out / f'provenance_rank{rank}.json', provenance)
    hop, renderer, registry = G._import_generator()
    G.register_grid_split(hop)
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    pools = G._PermutingPoolFactory(hop._Pool)
    cells = D.panel_cells() if args.mode == 'dev' else G.all_cells()
    if args.length is not None:
        cells = [cell for cell in cells if cell.length == args.length]
    ordinal = 0
    measured = 0
    reuse_qualifications = {}
    if args.reuse_unchanged_logits:
        for sampler in D.DECODER_GRID['sampler']:
            path = args.out / 'cache_parity' / f'rank{rank}_{sampler}.json'
            if path.exists():
                q = json.loads(path.read_text())
                if (q.get('provenance_sha256') != provenance_hash or q.get('qualified') is not True
                    or q.get('device') != torch.cuda.get_device_name()):
                    raise ValueError('stale GPU cache parity receipt')
                reuse_qualifications[sampler] = q
    for cell in cells:
        path = args.panel / f'{cell.cell_id}.json'
        rec = json.loads(path.read_text())
        if rec['status'] != G.STATUS_OK:
            if args.mode == 'dev':
                raise ValueError('dev panel cannot have unsupported cells')
            if rank == 0:
                atomic_json(args.out / 'unsupported' / path.name, rec)
            continue
        for inst in L.instances_of(L.load_cell_record(path)):
            item_ordinal = ordinal
            ordinal += 1
            for config_ordinal, config in enumerate(configs):
                assigned = assigned_rank(item_ordinal, config_ordinal, world) == rank
                if not assigned:
                    continue
                dest = args.out / D.config_id(config) / f'{cell.cell_id}__{inst.data_seed}__{inst.instance_index}.json'
                if dest.exists():
                    old = json.loads(dest.read_text())
                    validate_resume(old, provenance_hash=provenance_hash, instance=inst,
                                    config=config, split=split)
                    continue
                if args.max_items and measured >= args.max_items:
                    return
                task, canvas = verified_instance(hop, renderer, encoder, pools, cell, inst, split)
                ids = torch.tensor([canvas.input_ids], dtype=torch.long, device=device)
                rng_seed = item_seed(split, cell.cell_id, inst.data_seed, inst.instance_index, args.seed)
                if args.reuse_unchanged_logits and config['sampler'] not in reuse_qualifications:
                    reuse_qualification = qualify_logit_reuse(model, ids, inst.target_span,
                        config, mask_id=model.mask_token_id, pad_id=model.pad_token_id,
                        seed=rng_seed)
                    reuse_qualification.update({'rank': rank,
                        'input_ids_sha256': inst.input_ids_sha256,
                        'provenance_sha256': provenance_hash})
                    reuse_qualifications[config['sampler']] = reuse_qualification
                    atomic_json(args.out / 'cache_parity' / f"rank{rank}_{config['sampler']}.json", reuse_qualification)
                    print(f'rank={rank} unchanged-logit GPU replay parity passed', flush=True)
                generator = torch.Generator(device=device).manual_seed(rng_seed)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                tokens, calls, trace = sample_target(model, ids, inst.target_span, config,
                    mask_id=model.mask_token_id, pad_id=model.pad_token_id, generator=generator,
                    reuse_unchanged_logits=args.reuse_unchanged_logits)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                text = loaded.tokenizer.decode(tokens)
                answer = R.extract_answer(text, task.request.family)
                correct = len(inst.answers) == 1 and answer == inst.answers[0]
                atomic_json(dest, {'status': 'ok', 'split': split, 'cell_id': cell.cell_id,
                    'data_seed': inst.data_seed, 'instance_index': inst.instance_index,
                    'input_ids_sha256': inst.input_ids_sha256, 'prompt_hash_verified': True,
                    'config': config, 'provenance_sha256': provenance_hash,
                    'rng_seed': rng_seed, 'correct': correct, 'joint_exact': correct,
                    'parse_ok': answer is not None, 'prediction': answer, 'gold': inst.answers,
                    'text': text, 'output_ids': tokens, 'actual_nfe': calls,
                    'commits_per_step': trace, 'wall_seconds': seconds,
                    'logit_reuse_parity_sha256': digest(reuse_qualifications[config['sampler']]) if args.reuse_unchanged_logits else None,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'device': torch.cuda.get_device_name(), 'rank': rank})
                measured += 1
                print(f'{dest.name} {D.config_id(config)} nfe={calls} seconds={seconds:.3f}', flush=True)


if __name__ == '__main__':
    main()
