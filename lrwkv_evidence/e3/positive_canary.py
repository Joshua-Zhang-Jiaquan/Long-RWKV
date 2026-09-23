"""Exploratory positive-generation canaries; never primary confirmation evidence.

Eight workers: ranks 0--3 F2, ranks 4--7 released R0. Every model receives the
same eight fixed prompts, allocated by prompt index modulo four. Target budget
is always 32 tokens, deliberately different from gold-length confirmation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from . import grid324 as G, gpu_runner as U, causal_runner as A

BUDGET = 32
CONFIG = {'sampler': 'matched_bernoulli', 'steps': 8, 'tau': 0.7, 'commit_order': 'confidence'}
PROMPTS = (
    ('capital_completion', 'The capital of France is', ' Paris'),
    ('arithmetic_completion', '1 + 1 =', '2'),
    ('lookup_instruction', 'The stored record is key093=v1911. What value is stored for key093? Return only the value.\nAnswer: ', 'v1911'),
    ('lookup_one_shot', 'Records: key010=v1020\nQuestion: What is the value of key010?\nAnswer: v1020\n\nRecords: key093=v1911\nQuestion: What is the value of key093?\nAnswer: ', 'v1911'),
    ('symbolic_raw', 'BEGIN EVIDENCE\nTASK associative_recall\ndvk093=v1911\nQUERY dvk093\nEND EVIDENCE\nANSWER\n', 'v1911'),
    ('chat_capital', 'User: What is the capital of France? Answer with the city name only.\nAssistant:', ' Paris'),
    ('chat_lookup', 'User: The record is key093=v1911. Return only the value stored for key093.\nAssistant:', 'v1911'),
    ('chat_arithmetic', 'User: What is 1 + 1? Return only the number.\nAssistant:', '2'),
)


def make_canvas(encoder, prompt, mask_id):
    prefix = list(encoder.encode_text(prompt))
    if not prefix:
        raise ValueError('empty canary prefix')
    return prefix + [mask_id] * BUDGET, (len(prefix), len(prefix) + BUDGET)


def first_step(model_kind, model, ids, span, legal_ids, gold_ids, encoder):
    import torch
    with torch.inference_mode():
        if model_kind == 'f2':
            output = model(ids)
            logits = output[0, span[0]].float().clone()
            logits[model.mask_token_id] = -torch.inf
            logits[model.pad_token_id] = -torch.inf
        else:
            output = model(input_ids=ids[:, :span[0]], use_cache=False)
            logits = output.logits[0, -1].float().clone()
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            allowed[torch.tensor(legal_ids, device=ids.device)] = True
            logits[~allowed] = -torch.inf
        del output
        if not torch.isfinite(logits.logsumexp(-1)):
            raise ValueError('nonfinite canary logits')
        logprobs = logits.log_softmax(-1)
        top = torch.topk(logprobs, 10)
        result = {'position': 'first_mask' if model_kind == 'f2' else 'last_visible_prefix',
                  'distribution': 'untempered legal-token distribution',
                  'top10': [{'id': int(i), 'text': encoder.decode_sample_ids([int(i)]),
                             'logprob': float(p)} for p, i in zip(top.values.tolist(), top.indices.tolist())]}
        if gold_ids:
            token = gold_ids[0]
            result['expected_first_token'] = {'id': token, 'text': encoder.decode_sample_ids([token]),
                'rank': int((logits > logits[token]).sum()) + 1,
                'logprob': float(logprobs[token])}
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--scale', type=Path, default=G.G / 'qz_stage_traj4096_v7/scale')
    args = ap.parse_args(argv)
    rank, world, local = (int(os.environ.get(k, d)) for k, d in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if world != 8:
        raise ValueError('canary launcher requires eight workers')
    model_kind = 'f2' if rank < 4 else 'r0'
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = f'cuda:{local}'
    torch.manual_seed(17)
    sys.path.insert(0, str(args.scale))
    load_start = time.perf_counter()
    print(f'rank={rank} model={model_kind} loading exploratory canary', flush=True)
    if model_kind == 'f2':
        from eval.capability.birwkv_diffusion_model import load_birwkv_diffusion
        loaded = load_birwkv_diffusion(str(args.checkpoint), str(args.model_dir), device=device)
    else:
        from eval.capability.hf_causal_model import load_hf_causal
        loaded = load_hf_causal(str(args.model_dir), device=device)
    load_seconds = time.perf_counter() - load_start
    _, _, registry = G._import_generator()
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    native, legal, tokenizer_manifest = A.qualify_tokenizer(encoder, loaded.tokenizer)
    sources = [Path(__file__), Path(U.__file__), Path(A.__file__), Path(G.__file__),
        args.scale / 'eval/capability/birwkv_diffusion_model.py',
        args.scale / 'eval/capability/hf_causal_model.py',
        args.scale / 'models/birwkv7_diffusion.py']
    import fla.models.rwkv7.modeling_rwkv7 as rwkv_source
    import fla.layers.rwkv7 as layer_source
    sources += [Path(rwkv_source.__file__), Path(layer_source.__file__)]
    provenance = {'scope': 'exploratory_positive_canary_not_primary', 'model': model_kind,
        'prompt_set_sha256': U.digest(PROMPTS), 'budget_tokens': BUDGET,
        'policy': CONFIG if model_kind == 'f2' else A.POLICY,
        'checkpoint': str(args.checkpoint) if model_kind == 'f2' else str(args.model_dir),
        'weight_sha256': ({'model.pt': U.file_sha(args.checkpoint / 'model.pt')} if model_kind == 'f2'
            else {p.name: U.file_sha(p) for p in sorted(args.model_dir.glob('*.safetensors'))}),
        'source_sha256': {str(p): U.file_sha(p) for p in sources},
        'model_config_sha256': U.file_sha(args.model_dir / 'config.json'),
        'tokenizer': tokenizer_manifest, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(), 'load_seconds': load_seconds,
        'rank': rank, 'seed': 17, 'cpu_threads': 4,
        'execution_env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'BIRWKV_META_CONSTRUCT', 'BIRWKV_LOAD_SLOTS', 'LOOP_REPS_OVERRIDE')},
        'limitation': 'fixed diagnostic prompts and 32-token budget; no tuning or replacement of locked confirmation'}
    U.atomic_json(args.out / f'provenance_rank{rank}.json', provenance)
    for index, (name, prompt, expected) in enumerate(PROMPTS):
        if index % 4 != rank % 4:
            continue
        mask_id = loaded.model.mask_token_id if model_kind == 'f2' else 65535
        canvas, span = make_canvas(encoder, prompt, mask_id)
        ids = torch.tensor([canvas], dtype=torch.long, device=device)
        gold_ids = list(encoder.encode_text(expected))
        diagnostics = first_step(model_kind, loaded.model, ids, span, legal, gold_ids, encoder)
        check = A.verify_prefix(canvas[:span[0]], encoder, native, loaded.tokenizer)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if model_kind == 'f2':
            tokens, calls, trace = U.sample_target(loaded.model, ids, span, CONFIG,
                mask_id=loaded.model.mask_token_id, pad_id=loaded.model.pad_token_id,
                generator=torch.Generator(device=device).manual_seed(17 + index))
        else:
            tokens, calls = A.greedy_suffix(loaded.model, ids, span, legal)
            trace = None
        torch.cuda.synchronize()
        text = encoder.decode_sample_ids(tokens)
        U.atomic_json(args.out / model_kind / f'{name}.json', {
            'scope': 'exploratory_positive_canary_not_primary', 'status': 'ok', 'rank': rank,
            'model': model_kind, 'prompt_name': name, 'prompt': prompt,
            'expected_text_reference_only': expected, 'expected_ids': gold_ids,
            'input_ids_sha256': G.ids_sha256(canvas), 'target_span': span,
            'provenance_sha256': U.digest(provenance), 'tokenizer_check': check,
            'first_step': diagnostics, 'text': text, 'output_ids': tokens,
            'actual_nfe_generation': calls, 'diagnostic_nfe': 1, 'commits_per_step': trace,
            'wall_seconds_generation': time.perf_counter() - start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'scoring': 'qualitative canary; no post-hoc accuracy threshold'})
        print(f'{model_kind} {name}: {text!r}', flush=True)


if __name__ == '__main__':
    main()
