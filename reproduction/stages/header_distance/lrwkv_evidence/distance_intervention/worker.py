"""One discarded qualification, then fixed terminal matched adaptations."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from . import training as R
from lrwkv_evidence.train04 import worker as W
from lrwkv_evidence.train04dev import core as C
from lrwkv_evidence.train04mix import core as M


@torch.inference_mode()
def qualify(packed, tok, device):
    packed.eval()
    checks = []
    for step in (1, 2, 3):
        rows = R.records(step, 0, tok, 'balanced', R.SEEDS[0])[:2]
        b = R.batch(rows, device)
        normal = packed(b)
        singles = torch.cat([packed(R.batch([r], device)) for r in rows])
        poisoned = dict(b)
        poisoned['input_ids'] = b['input_ids'].clone()
        poisoned['input_ids'][0, :int(b['doc_starts'][0, 1])] = tok.binary_ids[1]
        changed = packed(poisoned)
        row = dict(position=rows[0]['position'], serial_difference=float((normal-singles).abs().max()),
                   other_document_difference=float((normal[1:]-changed[1:]).abs().max()),
                   own_document_difference=float((normal[0]-changed[0]).abs().max()))
        if row['serial_difference'] > 1e-5 or row['other_document_difference'] > 1e-5 or row['own_document_difference'] < 1e-6:
            raise ValueError(f'document isolation failed: {row}')
        checks.append(row)
    packed.train()
    return checks


def main():
    ap = argparse.ArgumentParser()
    for name in ('base', 'model-root', 'out', 'initial-checkpoint'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--initial-sha256', required=True)
    ap.add_argument('--arm', choices=R.ARMS, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, required=True)
    ap.add_argument('--qualification', action='store_true')
    ap.add_argument('--qualification-out', type=Path)
    args = ap.parse_args()
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 8 or os.environ.get('TRITON_F32_DEFAULT') != 'ieee':
        raise ValueError('eight IEEE ranks required')
    torch.set_num_threads(4); torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest'); torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl'); device = torch.device('cuda', local)
    root = Path(__file__).resolve().parents[2]
    sources = json.loads((root/'distance_sources.json').read_text())
    for rel, digest in sources.items():
        if W.file_sha(root/rel) != digest: raise ValueError('immutable source mismatch: '+rel)
    source_sha = W.canonical_sha(sources)
    if W.file_sha(args.initial_checkpoint) != args.initial_sha256: raise ValueError('initial checksum mismatch')
    design = json.loads((root/'distance_design.json').read_text())
    if args.initial_sha256 != design['initial_checkpoint_sha256']: raise ValueError('wrong starting lineage')
    if not args.qualification:
        q = json.loads((args.qualification_out/'completion.json').read_text())
        if not q.get('qualification') or not q.get('execution_complete') or q['step'] != 6 or q['source_sha256'] != source_sha:
            raise ValueError('qualification must match frozen runtime')
        execution = json.loads(Path(os.environ['DISTANCE_EXECUTION']).read_text())
        if execution['source_sha256'] != source_sha or args.steps != execution['terminal_steps'] or args.seed not in R.SEEDS:
            raise ValueError('wrong fixed execution selector')
    elif args.steps != 6 or args.arm != 'balanced' or args.seed != R.SEEDS[0]:
        raise ValueError('wrong qualification selector')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'train.jsonl').exists() or (args.out/'completion.json').exists(): raise ValueError('refuse reused output')
    tok, bits, mask = W.load_tokenizer(args.base)
    model, identity, _ = W.build_model(args.model_root, args.base, bits, mask, 0)
    model.backbone.lm_head = C.StableBinaryHead(model.backbone.lm_head)
    initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if initial['step'] != 2500 or initial['contract']['seed'] != 71 or initial['contract']['phase'] != 'independent':
        raise ValueError('wrong auxiliary conditioner')
    model.load_state_dict(initial['model'], strict=True); del initial
    model = model.to(device); packed = C.SerialDenoiser(model)
    checks = qualify(packed, tok, device)
    W.atomic_json(args.out/f'qualification_rank{rank}.json', dict(source_sha256=source_sha, checks=checks))
    ddp = torch.nn.parallel.DistributedDataParallel(packed, device_ids=[local], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-5, betas=(.9,.95), eps=1e-8, weight_decay=.01)
    contract = dict(recipe=R.RECIPE, arm=args.arm, seed=args.seed, terminal_steps=args.steps,
                    initial_checkpoint_sha256=args.initial_sha256, base_sha256=W.file_sha(args.base/'model.safetensors'),
                    source_sha256=source_sha, design_sha256=W.file_sha(root/'distance_design.json'))
    digest = W.canonical_sha(contract)
    if rank == 0:
        W.atomic_json(args.out/'provenance.json', dict(contract=contract, contract_sha256=digest, identity=identity,
                                                     qualification=args.qualification, stop_step=args.steps))
    dist.barrier(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps+1):
        began = time.perf_counter(); rows = R.records(step, rank, tok, args.arm, args.seed)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.; entropy_sum = 0.; tokens = 0
        for j, row in enumerate(rows):
            b = R.batch([row], device)
            with ddp.no_sync() if j < len(rows)-1 else nullcontext():
                value = R.loss(ddp(b), b) / len(rows)
                if not bool(torch.isfinite(value)): raise ValueError('nonfinite loss')
                value.backward()
            loss_sum += float(value.detach()); entropy_sum += float(M.oracle_entropy(b))/len(rows)
            tokens += b['input_ids'].numel()
        norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1., error_if_nonfinite=True)
        if step == 1:
            missing = [name for name, p in model.named_parameters() if p.grad is None or p.dtype != torch.float32 or p.grad.dtype != torch.float32]
            if missing: raise ValueError('missing or wrong dtype gradients: '+str(missing[:5]))
        lr = 1e-5 * min(step/20, 1.)
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.step()
        values = torch.tensor([loss_sum, entropy_sum, tokens], dtype=torch.float64, device=device); dist.all_reduce(values)
        torch.cuda.synchronize()
        profile = torch.tensor([time.perf_counter()-began, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], dtype=torch.float64, device=device)
        dist.all_reduce(profile, op=dist.ReduceOp.MAX)
        if rank == 0:
            row = dict(step=step, loss=float(values[0]/8), oracle_entropy=float(values[1]/8),
                       excess_ce=float((values[0]-values[1])/8), global_input_tokens=int(values[2]),
                       grad_norm_rank0=float(norm), lr=lr, elapsed_seconds=time.perf_counter()-started,
                       max_rank_step_seconds=float(profile[0]), max_rank_peak_allocated_bytes=int(profile[1]),
                       max_rank_peak_reserved_bytes=int(profile[2]))
            with (args.out/'train.jsonl').open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(row), flush=True)
    dist.barrier()
    if rank == 0:
        tmp = args.out/'resume.pt.tmp'
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(), step=args.steps, contract=contract, contract_sha256=digest), tmp)
        os.replace(tmp, args.out/'resume.pt')
        W.atomic_json(args.out/'completion.json', dict(execution_complete=True, qualification=args.qualification,
                      step=args.steps, source_sha256=source_sha, contract_sha256=digest,
                      checkpoint_sha256=W.file_sha(args.out/'resume.pt'), elapsed_seconds=time.perf_counter()-started,
                      arm=args.arm, seed=args.seed))
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
