"""Bounded full-model A1 training under the exact eight-stage absorbing loss.

FP32 master parameters and Adam states, BF16 autocast, one unpacked example per
rank/microstep. Targets use a two-token vocabulary head; visible prefix is fixed.
No causal auxiliary, hidden-depth loop, gate modulation, or adapter-only training.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

STAGES = 8


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def state_digest(module):
    """Tensor-content identity independent of torch serialization container bytes."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        data = memoryview(value.reshape(-1).view(torch.uint8).numpy())
        for start in range(0, len(data), 8 * 1024 * 1024):
            h.update(data[start:start + 8 * 1024 * 1024])
    return h.hexdigest()


class BinaryHead(nn.Module):
    """Retain the original trainable head parameter; project only two rows."""
    def __init__(self, original, binary_ids):
        super().__init__()
        if original.bias is not None:
            raise ValueError('expected the bias-free released vocabulary head')
        self.weight = original.weight
        self.register_buffer('binary_ids', torch.tensor(binary_ids, dtype=torch.long))

    def forward(self, hidden):
        return F.linear(hidden, self.weight.index_select(0, self.binary_ids))


class BinaryDenoiser(nn.Module):
    def __init__(self, backbone, binary_ids, mask_id):
        super().__init__()
        if len(binary_ids) != 2 or len(set(binary_ids)) != 2 or mask_id in binary_ids:
            raise ValueError('two distinct binary IDs, separate from mask ID, are required')
        if backbone.arm != 'A1' or not backbone.spec.bidirectional or backbone.spec.hidden_loop or backbone.spec.gate_mod:
            raise ValueError('only tied bidirectional A1 without loops/gate modules is allowed')
        self.backbone = backbone
        self.binary_ids = tuple(int(i) for i in binary_ids)
        self.mask_id = int(mask_id)
        backbone.lm_head = BinaryHead(backbone.lm_head, binary_ids)

    def forward(self, corrupted, target_span, stage):
        lo, hi = target_span
        if corrupted.ndim != 2 or corrupted.shape[0] != 1 or not 0 < lo < hi == corrupted.shape[1]:
            raise ValueError('expected B=1 with a nonempty target suffix')
        if not 1 <= stage <= STAGES:
            raise ValueError('stage must be in 1..8')
        # longrwkv.conditioning constants: immutable=1, filled=2, masked=3.
        codes = torch.ones_like(corrupted)
        codes[:, lo:hi] = torch.where(corrupted[:, lo:hi] == self.mask_id, 3, 2)
        block_t = torch.zeros_like(corrupted, dtype=torch.float32)
        block_t[:, lo:hi] = stage / STAGES
        gather = torch.arange(lo, hi, device=corrupted.device)
        return self.backbone(corrupted, attention_mask=torch.ones_like(corrupted, dtype=torch.bool),
            block_t=block_t, codes=codes, block_size=1, gather_idx=gather, R=1, force_forward=False)


def corrupt_targets(clean, span, mask_id, generator, stage=None):
    lo, hi = span
    if clean.ndim != 2 or clean.shape[0] != 1 or not 0 < lo < hi == clean.shape[1]:
        raise ValueError('invalid target span')
    if stage is None:
        stage = int(torch.randint(1, STAGES + 1, (), generator=generator, device=clean.device))
    if not 1 <= stage <= STAGES:
        raise ValueError('stage must be in 1..8')
    selected = torch.rand((hi - lo,), generator=generator, device=clean.device) < stage / STAGES
    corrupted = clean.clone()
    corrupted[0, lo:hi] = torch.where(selected, mask_id, clean[0, lo:hi])
    return corrupted, selected, stage


def exact_absorbing_loss(logits, gold_bits, masked, stage):
    """Uniform stage sampling: (8/(t*N))*sum_masked CE, including empty masks."""
    if logits.shape != (gold_bits.numel(), 2) or masked.shape != gold_bits.shape or gold_bits.numel() == 0:
        raise ValueError('expected [N,2] logits and [N] labels/mask')
    if not 1 <= stage <= STAGES:
        raise ValueError('stage must be in 1..8')
    per_position = F.cross_entropy(logits.float(), gold_bits.long(), reduction='none')
    return (per_position * masked.to(per_position.dtype)).sum() * (STAGES / (stage * gold_bits.numel()))


def validate_example(example, binary_ids, mask_id):
    ids, span, bits = example['input_ids'], tuple(example['target_span']), example['gold_bits']
    lo, hi = span
    if not 0 < lo < hi == len(ids) or len(bits) != hi - lo:
        raise ValueError('task example has invalid target suffix')
    if any(type(b) is not int or b not in (0, 1) for b in bits):
        raise ValueError('task labels must be binary integers')
    if list(ids[lo:hi]) != [binary_ids[b] for b in bits] or mask_id in ids:
        raise ValueError('task clean target IDs do not match binary labels')
    return list(ids), span, list(bits)


class NativeTokenizer:
    def __init__(self, native, vocab_path, implementation_path):
        self.native = native
        self.vocab_path = Path(vocab_path)
        self.implementation_path = Path(implementation_path)
        encoded = [self.encode(str(bit)) for bit in (0, 1)]
        if any(len(ids) != 1 for ids in encoded):
            raise ValueError('binary characters must each be a single native token')
        self.binary_ids = tuple(ids[0] for ids in encoded)
        self.mask_id = 65535
        if len(set(self.binary_ids)) != 2 or self.mask_id in self.binary_ids:
            raise ValueError('invalid binary/mask token IDs')
        for bit, token in enumerate(self.binary_ids):
            if self.native.decode([[token]])[0] != str(bit):
                raise ValueError('native bit token does not roundtrip')

    def encode(self, text):
        return list(self.native.encode(text)[0])


def load_tokenizer(checkpoint):
    """Load the checkpoint's native trie, avoiding HF added-token rewriting."""
    checkpoint = Path(checkpoint)
    sys.path.insert(0, str(checkpoint))
    import hf_rwkv_tokenizer as implementation
    vocab = checkpoint / 'rwkv_vocab_v20230424.txt'
    tokenizer = NativeTokenizer(implementation.RWKV_TOKENIZER(str(vocab)), vocab, implementation.__file__)
    return tokenizer, tokenizer.binary_ids, tokenizer.mask_id


def tokenize_example(example, tokenizer):
    """Prefix text + explicitly separated single-bit IDs; no gold-length cue."""
    prefix = tokenizer.encode(example['prompt'])
    bits = list(example['bits'])
    result = {**example, 'input_ids': prefix + [tokenizer.binary_ids[b] for b in bits],
              'target_span': (len(prefix), len(prefix) + len(bits)), 'gold_bits': bits}
    validate_example(result, tokenizer.binary_ids, tokenizer.mask_id)
    return result


def build_model(model_root, checkpoint, binary_ids, mask_id, init_seed):
    sys.path.insert(0, str(model_root))
    from longrwkv.model import LongRWKV, load_config
    torch.manual_seed(init_seed)
    config = load_config(checkpoint / 'config.json')
    backbone = LongRWKV.from_hf_pretrained(checkpoint, config, arm='A1', dtype=torch.float32)
    backbone.gradient_checkpointing = True
    identity = backbone.assert_parameter_identity()
    model = BinaryDenoiser(backbone, binary_ids, mask_id)
    if any(not p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('all model parameters must be trainable FP32 masters')
    return model, identity, config.to_record()


def save_checkpoint(path, model, optimizer, *, step, provenance, generator_state, save_optimizer):
    payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'step': step, 'provenance': provenance, 'corruption_generator_state': generator_state,
               'optimizer': optimizer.state_dict() if save_optimizer else None}
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return file_sha(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['smoke', 'train'], required=True)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--seed', type=int, choices=[17, 29, 43], default=17)
    ap.add_argument('--init-seed', type=int, default=0)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--model-root', type=Path, required=True)
    ap.add_argument('--task-module', default='lrwkv_evidence.train04.tasks')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke-receipt', type=Path)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--save-optimizer', action='store_true')
    args = ap.parse_args(argv)
    steps = args.steps or (20 if args.mode == 'smoke' else 500)
    if steps != (20 if args.mode == 'smoke' else 500):
        ap.error('the bounded protocol fixes smoke=20 and train=500 optimizer steps')
    if args.grad_accum != 4:
        ap.error('the fixed protocol uses grad-accum4 / global batch32')
    world, rank, local = (int(os.environ.get(k, d)) for k, d in [('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')])
    if world != 8:
        raise ValueError('this training protocol requires eight DDP ranks')
    import torch.distributed as dist
    torch.set_num_threads(4); torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    device = torch.device('cuda', local)
    task_module = importlib.import_module(args.task_module)
    tokenizer, binary_ids, mask_id = load_tokenizer(args.checkpoint)
    model, identity, config = build_model(args.model_root, args.checkpoint, binary_ids, mask_id, args.init_seed)
    initial_hash = state_digest(model)
    initial_hashes = [None] * world
    dist.all_gather_object(initial_hashes, initial_hash)
    if len(set(initial_hashes)) != 1:
        raise ValueError('DDP workers did not construct the same initial parameter state')
    source_hashes = {str(p): file_sha(p) for p in [Path(__file__), Path(task_module.__file__), *sorted((args.model_root / 'longrwkv').glob('*.py'))]}
    contract = {'arm': 'A1', 'stages': 8, 'objective': 'uniform t in1..8; independent targetmask p=t/8; maskedCEsum*8/(t*N)',
                'binary_ids': binary_ids, 'mask_id': mask_id, 'master_dtype': 'float32', 'autocast_dtype': 'bfloat16',
                'prefix': 'clamped', 'hidden_loop': False, 'gate_mod': False, 'initial_seed': args.init_seed,
                'base_sha256': file_sha(args.checkpoint / 'model.safetensors'), 'model_config': config,
                'vocab_sha256': file_sha(tokenizer.vocab_path),
                'tokenizer_source_sha256': file_sha(tokenizer.implementation_path),
                'grad_accum': 4, 'global_batch': 32, 'lr': args.lr, 'weight_decay': args.weight_decay,
                'source_content_sha256': sorted(source_hashes.values())}
    contract_hash = canonical_sha(contract)
    if args.mode == 'train':
        if not args.smoke_receipt:
            raise ValueError('training requires a successful20-step smoke receipt')
        smoke = json.loads(args.smoke_receipt.read_text())
        if smoke.get('qualified') is not True or smoke.get('steps') != 20 or smoke.get('contract_sha256') != contract_hash:
            raise ValueError('missing/incompatible smoke qualification')
    provenance = {'mode': args.mode, 'seed': args.seed, 'init_seed': args.init_seed, 'steps': steps,
        'world_size': world, 'per_rank_batch': 1, 'grad_accum': args.grad_accum, 'lr': args.lr,
        'weight_decay': args.weight_decay, 'optimizer': 'AdamW(beta=.9,.95,eps1e-8)',
        'contract': contract, 'contract_sha256': contract_hash, 'source_sha256': source_hashes,
        'initial_state_sha256': initial_hash, 'identity': identity, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'cpu_threads': 4}
    args.out.mkdir(parents=True, exist_ok=True)
    occupied = torch.tensor([int((args.out / 'provenance.json').exists()) if rank == 0 else 0], device=device)
    dist.broadcast(occupied, src=0)
    if int(occupied.item()):
        raise ValueError('output already contains a run; use a fresh directory')
    model = model.to(device).train()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device).manual_seed(args.seed * 1000 + rank)
    if rank == 0:
        atomic_json(args.out / 'provenance.json', provenance)
    dist.barrier()
    total_examples = total_tokens = total_targets = total_masked = total_empty = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    gradient_checks = None
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        statistics = torch.zeros(7, dtype=torch.float64, device=device)
        for micro in range(args.grad_accum):
            index = ((step - 1) * args.grad_accum + micro) * world + rank
            example = tokenize_example(task_module.make_example(split='train', seed=args.seed, index=index, n=8), tokenizer)
            ids, span, bits = validate_example(example, binary_ids, mask_id)
            clean = torch.tensor([ids], dtype=torch.long, device=device)
            gold = torch.tensor(bits, dtype=torch.long, device=device)
            corrupted, masked, stage = corrupt_targets(clean, span, mask_id, generator)
            sync = ddp.no_sync() if micro + 1 < args.grad_accum else nullcontext()
            with sync:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = ddp(corrupted, span, stage)
                    loss = exact_absorbing_loss(logits, gold, masked, stage)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite training loss')
                (loss / args.grad_accum).backward()
            statistics += torch.tensor([float(loss.detach()), 1, len(ids), len(bits), int(masked.sum()),
                                         int(not bool(masked.any())), stage], dtype=torch.float64, device=device)
        norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0, error_if_nonfinite=True)
        if step == 1:
            missing = [name for name, p in model.named_parameters() if p.grad is None]
            wrong_dtype = [name for name, p in model.named_parameters() if p.dtype != torch.float32 or (p.grad is not None and p.grad.dtype != torch.float32)]
            if missing or wrong_dtype:
                raise ValueError(f'gradient qualification failed missing={missing[:5]} dtype={wrong_dtype[:5]}')
            gradient_checks = {'all_parameters_trainable': True, 'all_master_and_grad_fp32': True,
                               'all_parameters_have_gradient': True, 'first_grad_norm': float(norm)}
        optimizer.step()
        if step == 1:
            bad_optimizer_dtypes = [str(t.dtype) for state in optimizer.state.values() for t in state.values()
                                   if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != torch.float32]
            if bad_optimizer_dtypes:
                raise ValueError(f'optimizer state is not FP32: {bad_optimizer_dtypes[:5]}')
            gradient_checks['all_optimizer_float32'] = True
        dist.all_reduce(statistics)
        values = statistics.tolist()
        total_examples += int(values[1]); total_tokens += int(values[2]); total_targets += int(values[3])
        total_masked += int(values[4]); total_empty += int(values[5])
        if rank == 0:
            record = {'step': step, 'mean_loss': values[0] / values[1], 'examples': int(values[1]),
                'input_tokens': int(values[2]), 'target_tokens': int(values[3]), 'masked_targets': int(values[4]),
                'empty_masks': int(values[5]), 'mean_stage': values[6] / values[1], 'rank0_grad_norm': float(norm),
                'elapsed_seconds': time.perf_counter() - started}
            with (args.out / 'train.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush()
            print(json.dumps(record), flush=True)
    dist.barrier()
    if rank == 0:
        final_state_hash = state_digest(model)
        if final_state_hash == initial_hash:
            raise ValueError('training did not change model state')
        weights_sha = save_checkpoint(args.out / 'final.pt', model, optimizer, step=steps,
            provenance=provenance, generator_state=generator.get_state(), save_optimizer=args.save_optimizer)
        result = {'qualified': True, 'mode': args.mode, 'steps': steps, 'seed': args.seed,
            'contract_sha256': contract_hash, 'initial_state_sha256': initial_hash, 'final_state_sha256': final_state_hash,
            'checkpoint_sha256': weights_sha, 'gradient_checks': gradient_checks,
            'examples': total_examples, 'input_tokens': total_tokens, 'target_tokens': total_targets,
            'masked_targets': total_masked, 'empty_mask_examples': total_empty,
            'denoiser_forward_calls': total_examples, 'elapsed_seconds': time.perf_counter() - started,
            'peak_allocated_bytes_rank0': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes_rank0': torch.cuda.max_memory_reserved(),
            'optimizer_saved': args.save_optimizer, 'resume_implemented': False}
        atomic_json(args.out / 'completion.json', result)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
