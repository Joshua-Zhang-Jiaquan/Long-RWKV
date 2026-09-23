"""The DDP trainer for every arm and stage.

Three deliberate choices, each of which has a recorded failure behind it:

**DDP, not FSDP.**  At 0.45B, weights plus Adam state are about 7.2 GB per rank,
so sharding buys nothing -- and FSDP's wrap-by-class over a module holding a
single shared mixer is exactly where a silent untying would hide.  Plain
``state_dict()`` also stays trivially correct, which the run record depends on.

**The global batch is a checked invariant.**  The campaign runs arms at 32 GPUs
with ``grad_accum`` dropped from 4 to 1 so the global batch stays the value the
sibling ablation validated.  A global batch validated globally and applied
per-rank is a mistake this lineage has already made once, so the product is
asserted at startup and refuses before the queue wait rather than after it.

**The hidden-loop multiplicity is sampled, ramped, and declared.**  Applying
pretrained layers twice is not identity at init, so training starts at R=1 and
ramps to the declared distribution; the distribution used is written into the run
record, and per-R losses are logged so the loop's liveness is visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .conditioning import vis_codes
from .corruption import sample_corruption
from .ids import MASK_ID, PAD_ID
from .losses import (causal_next_token_valid, next_token_ce_global,
                     selected_token_ce_global)
from .model import ARM_SPECS, LongRWKV, load_config
from .refusal import Refusal
from .reversal import flatten_microbatch

TARGET_GLOBAL_BATCH = 1_048_576
"""microbatch * world_size * grad_accum * seq, the validated recipe's value."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_SPECS))
    parser.add_argument("--stage", default="adapt", choices=["adapt", "rollout", "long"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pack-dir", default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="defaults to the value that holds the global batch")
    parser.add_argument("--steps", type=int, default=1908)
    # **The ablation's schedule, because the ablation is the comparability gate.**
    # The plan admits the primary matrix only if its first 500 steps track the
    # sibling ablation's a1 curve at equal tokens, and that curve was produced at
    # peak 3e-5 with cosine decay to 0.1x (``train_birwkv_diffusion.py:1005-1008``,
    # ``--lr`` default 3e-5).  A 1e-4 peak held flat is 3.3x the peak on the same
    # pretrained checkpoint with no anneal at all, so the two curves cannot be
    # compared even when both are correct -- and the measured round-2 curve did
    # exactly what too-high-and-flat predicts: it bounced at ~6.1 for 180 steps
    # while the ablation was at 5.67 and still descending.
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-final-fraction", type=float, default=0.1,
                        help="cosine floor as a fraction of peak; 1.0 disables decay")
    parser.add_argument("--lambda-causal", type=float, default=0.1)
    parser.add_argument("--causal-aux-positions", type=int, default=4096,
                        help="positions sampled for the causal auxiliary; 0 uses "
                             "every valid pair, which does not fit at microbatch 8")
    parser.add_argument("--lambda-rollout", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--span-prob", type=float, default=0.5)
    parser.add_argument("--loop-ramp-steps", type=int, default=300)
    parser.add_argument("--rollout-batch-fraction", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--allow-package-change", action="store_true",
                        help="permit --resume-from across a package digest change; "
                             "both digests are recorded")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--allow-global-batch-change", action="store_true",
                        help="acknowledge that this run is NOT the validated recipe")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="retain every block's activations; the memory "
                             "arithmetic says this does not fit at microbatch 8")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip the corpus; used by the scaling leg only")
    return parser.parse_args(argv)


def package_digest(root: Path) -> str:
    """A content hash over the package's sources, for the run registry."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    """The sha256 of one file, read in chunks.

    A checkpoint is gigabytes, so it is not slurped: ``read_bytes`` on a 3.6 GiB
    file would hold the whole thing in memory beside the model that just saved it.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def lr_at(step: int, args) -> float:
    """Linear warmup then cosine decay to ``lr_final_fraction`` of peak.

    **Transcribed from the run this campaign is compared against**, not invented:
    ``train_birwkv_diffusion.py:1005-1008``, whose ``a1`` curve is the plan's
    admission gate for the primary matrix.  Keeping the shape identical is what
    makes "tracks the ablation's a1 curve at equal tokens" a statement about the
    model rather than about two different optimizers.

    ``step`` is the count of *completed* steps, so the first update uses
    ``(step + 1) / warmup`` and the schedule never starts at exactly zero -- a
    zero first step wastes a step and shifts every later comparison by one.

    The previous body was warmup-only: ``lr * min(1.0, (step+1)/warmup)``, which
    holds the peak for every step after warmup.  At the old 1e-4 default that put
    3.3x the ablation's peak on a pretrained checkpoint with no anneal, and the
    measured curve bounced at ~6.1 nats for 180 steps instead of descending.
    """
    warmup = max(1, args.warmup_steps)
    if step < warmup:
        return args.lr * (step + 1) / warmup
    floor = args.lr_final_fraction
    span = max(1, args.steps - warmup)
    t = min(1.0, (step - warmup) / span)
    return args.lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t)))


def loop_multiplicity(step: int, ramp_steps: int, multiplicities, generator,
                      device) -> int:
    """R ~ Categorical over the declared multiplicities, after a ramp from R=1.

    The ramp exists because direct loop re-application moves the step-0 logits:
    training begins at the pretrained configuration and moves away from it, rather
    than starting from a perturbed one.
    """
    if step < ramp_steps:
        return 1
    index = int(torch.randint(len(multiplicities), (1,), generator=generator,
                              device=device).item())
    return multiplicities[index]


#: The two properties of the trained model that are memory decisions, not
#: preferences.  They live here, in one dict, because the scaling gate builds its
#: own model: the gate's whole purpose is to certify *this* recipe at 32 GPUs,
#: and a gate that builds a different model certifies a shape nobody will run.
#: Round 6's 2n and 4n legs both OOMed at 75.7 GiB on every rank while every
#: other check passed, because the gate took ``from_hf_pretrained``'s bf16
#: default and left ``gradient_checkpointing`` at its off-by-default value --
#: activations for 24 bidirectional blocks over a 32768-position flattened row.
TRAINING_MODEL_BUILD = {
    # fp32 masters: autocast sets the dtype of the ops, not of what AdamW
    # updates, and 3.0% of this checkpoint's parameters have 1e-4 updates below
    # half a bf16 ulp (see the comment at the call site in ``main``).
    "dtype": torch.float32,
    # Recomputation per block, which ``model.py``'s arithmetic requires rather
    # than prefers.
    "gradient_checkpointing": True,
}


def build_training_model(checkpoint, config, arm: str, *,
                         gradient_checkpointing: bool | None = None):
    """The model exactly as a training arm builds it, before ``.to`` and DDP.

    Both the trainer and ``qz.scaling_leg`` call this.  The gate certifies the
    recipe, so it must construct the recipe; the alternative is what round 6
    did -- measure a bf16, un-checkpointed model, OOM, and report a failure
    about a configuration the campaign never runs.
    """
    model = LongRWKV.from_hf_pretrained(
        checkpoint, config, arm=arm, dtype=TRAINING_MODEL_BUILD["dtype"])
    model.gradient_checkpointing = (
        TRAINING_MODEL_BUILD["gradient_checkpointing"]
        if gradient_checkpointing is None else bool(gradient_checkpointing))
    return model


def build_loader(args, rank: int, world: int):
    from .data.packed_dataset import PackedDCLMDataset, ShardContiguousSampler, collate_packed

    dataset = PackedDCLMDataset(args.pack_dir, max_length=args.max_length)
    # ``shard_rows`` is what turns the shuffle from a whole-slice permutation into
    # a shard-grouped one, and it is the difference between a 0.836 s and a 0.001 s
    # batch on this pack (measured, one rank, no contention).  ``interleave`` is
    # tied to the dataset's own cache depth rather than chosen: grouping more
    # shards than the cache holds is the thrash the grouping exists to prevent.
    sampler = ShardContiguousSampler(len(dataset), world_size=world, rank=rank,
                                     shuffle=True, seed=args.seed,
                                     shard_rows=dataset.shard_rows,
                                     interleave=dataset.cache_shards)
    return DataLoader(dataset, batch_size=args.microbatch, sampler=sampler,
                      drop_last=True, num_workers=4, collate_fn=collate_packed,
                      pin_memory=True, persistent_workers=False), dataset, sampler


def require_objective_flags(args) -> bool:
    """Return whether ``args.arm`` trains autoregressively, refusing a mismatch.

    **The objective comes from the arm, not from a flag.**  A0 is the paper's
    autoregressive reference and the baseline every diffusion arm is measured
    against; M6 is the forward-only *denoiser* that isolates direction from
    objective.  While both were spelled only as flag values, the two arms emitted
    byte-identical ``EXTRA_ARGS`` and A0 trained M6's loss for three seeds: masked
    CE at weight 1.0 with the causal term as a 0.1 auxiliary.  Reading
    ``ARM_SPECS[arm].autoregressive`` here means the label and the loss cannot
    disagree, and the refusals below mean a contradicting flag stops the run
    instead of producing a plausible row.

    A function rather than a block inside ``main``: ``main`` needs a checkpoint, a
    pack and a process group before it reaches the gate, so a block there is
    reachable only from a GPU job -- and a guard no CPU test can execute is how
    the objective went unchecked in the first place.
    """
    spec = ARM_SPECS[args.arm]
    if spec.autoregressive:
        if args.lambda_causal != 1.0:
            raise Refusal(
                f"arm {args.arm} declares the autoregressive objective, so "
                f"next-token CE is the whole loss and --lambda-causal must be 1.0, "
                f"not {args.lambda_causal}. A weight below 1.0 on an arm with no "
                f"masked term trains a scaled loss, and the record would still "
                f"read arm={args.arm}.")
        if args.lambda_rollout:
            raise Refusal(
                f"arm {args.arm} is autoregressive; the rollout auxiliary "
                f"supervises sampler calls on a masked canvas and has no meaning "
                f"here (got --lambda-rollout {args.lambda_rollout})")
        if args.causal_aux_positions:
            raise Refusal(
                f"arm {args.arm} is autoregressive, so every valid next-token pair "
                f"is the objective rather than a sampled auxiliary; pass "
                f"--causal-aux-positions 0 (got {args.causal_aux_positions}). The "
                f"subsample is unbiased for an auxiliary's mean and wrong as a "
                f"primary loss.")
    elif args.lambda_causal >= 1.0:
        raise Refusal(
            f"arm {args.arm} trains the masked objective, where the causal term is "
            f"an auxiliary; --lambda-causal {args.lambda_causal} weights it at or "
            f"above the primary loss. Only an arm declaring "
            f"objective=autoregressive may do that.")
    return spec.autoregressive


def loss_terms_for(args, autoregressive: bool) -> list[str]:
    """The loss this run will actually optimize, as the record's own field.

    Written from the same ``args`` the step body reads, so a record cannot name a
    term the loop did not apply.  ``arm`` alone did not distinguish A0 from M6 in
    any artifact; this does.
    """
    if autoregressive:
        return ["next_token_ce@1.0"]
    terms = ["selected_token_ce@1.0",
             f"causal_next_token_ce@{args.lambda_causal}"]
    if args.lambda_rollout:
        terms.append(f"rollout@{args.lambda_rollout}")
    return terms


def main(argv=None) -> int:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1

    # The objective comes from the arm, not from a flag; see
    # ``require_objective_flags``, which is where the reasoning and the four
    # refusals live so a CPU test can execute them.
    spec = ARM_SPECS[args.arm]
    autoregressive = require_objective_flags(args)

    seq = args.max_length
    grad_accum = args.grad_accum
    if grad_accum is None:
        grad_accum = max(1, round(TARGET_GLOBAL_BATCH / (args.microbatch * world * seq)))
    if grad_accum < 1:
        raise Refusal(
            f"grad_accum computes to {grad_accum}: {args.microbatch} x {world} x {seq} "
            f"already exceeds the target global batch {TARGET_GLOBAL_BATCH}")
    global_batch = args.microbatch * world * grad_accum * seq
    if global_batch != TARGET_GLOBAL_BATCH and not args.allow_global_batch_change:
        raise Refusal(
            f"global batch {global_batch} != the validated {TARGET_GLOBAL_BATCH} "
            f"(microbatch {args.microbatch} x world {world} x grad_accum {grad_accum} "
            f"x seq {seq}). Pass --allow-global-batch-change to train a different "
            f"recipe, and record it.")

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config or Path(args.checkpoint) / "config.json")
    # **fp32 master weights.**  ``from_hf_pretrained`` defaults to bf16, and this
    # call used to take that default, so the parameters, the gradients and both
    # AdamW moments were all bf16 -- the autocast below sets the compute dtype of
    # the ops, not the dtype of what the optimizer updates.  Measured on this
    # checkpoint: a 1e-4 update falls below half a bf16 ulp for 13,494,179 of the
    # 450,767,872 parameters (3.0%), and round-to-nearest discards it every step
    # rather than accumulating it, so those weights -- the largest-magnitude ones
    # -- are frozen for the whole run while exp_avg_sq is simultaneously held to
    # bf16's three significant digits.  The loss still descends, because the other
    # 97% still move; that is what makes it worth a comment rather than a crash.
    # This module's own budget ("about 7.2 GB per rank" for weights plus Adam
    # state) is the fp32 figure: 0.45B x 16 bytes.
    model = build_training_model(
        args.checkpoint, config, args.arm,
        gradient_checkpointing=not args.no_gradient_checkpointing)
    if rank == 0:
        identity = model.assert_parameter_identity()
        (outdir / "parameter_identity.json").write_text(
            json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    model = model.to(device)

    if distributed:
        # The causal-only arms (A0, M6) never call the backward direction, so the
        # three fusion parameters per layer -- 72 of them at 24 layers -- produce no
        # gradient, and DDP refuses on the second step: "Expected to have finished
        # reduction in the prior iteration before starting a new one".  Every rank
        # runs the same arm, so all 32 raise it together; the job dies fast rather
        # than desyncing, but it dies.  The search costs a per-step graph traversal,
        # so it is paid only by the arms that need it.
        find_unused = ARM_SPECS[args.arm].causal_only
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=find_unused,
                    broadcast_buffers=False)
    raw_model = model.module if distributed else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=args.weight_decay)
    scaler_state = {}
    start_step = 0
    if args.resume_from:
        payload = torch.load(args.resume_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scaler_state = payload.get("scaler", {})
        start_step = int(payload.get("step", 0))
        # **A resume under changed code is a different experiment.**  The pods run
        # a staged copy of this package, and ``auto_fault_tolerance`` is on with
        # three retries -- so a retry that lands after the package is restaged
        # continues one run's optimizer state under another run's code, and
        # rewrites ``package_digest`` to the new value on the way out.  The record
        # would then name code that never produced most of the steps.
        #
        # This is not hypothetical: the LR schedule changed between the primary
        # matrix's first submission and its relaunch, and warmup-only versus
        # cosine-decayed is exactly the kind of difference no loss curve labels.
        # Refuse, and say which digest is which, rather than blending them.
        staged = package_digest(Path(__file__).resolve().parents[1])
        # The digest lives inside the saved ``record``, which is what
        # ``save_checkpoint`` stores -- not at the payload's top level.
        prior = (payload.get("record") or {}).get("package_digest")
        if prior and prior != staged and not args.allow_package_change:
            raise Refusal(
                f"the checkpoint at {args.resume_from} was written by package "
                f"{prior[:12]} and this package is {staged[:12]}; resuming would "
                f"continue one experiment's optimizer under another's code. Pass "
                f"--allow-package-change to record both digests and proceed.")
        if prior and prior != staged:
            resumed_under = prior
        else:
            resumed_under = None
    else:
        resumed_under = None

    loader, dataset, sampler = build_loader(args, rank, world) if not args.synthetic else \
        (None, None, None)
    if rank == 0 and sampler is not None:
        (outdir / "sampler_disjointness.json").write_text(
            json.dumps(sampler.assert_disjoint(), indent=2) + "\n", encoding="utf-8")

    record = {
        "arm": args.arm,
        "arm_spec": ARM_SPECS[args.arm].__dict__,
        # **What this run actually optimized, in its own field.**  ``arm`` is a
        # label and ``arm_spec`` a dict of flags; neither said which loss ran, so
        # the three A0 runs that trained the masked objective carry nothing that
        # contradicts their label.  ``objective`` is read off the spec (so it
        # cannot drift from the branch the step body takes) and ``loss_terms``
        # names the terms with their weights, which is the claim the paper's
        # method section makes and the one a reader can check.
        "objective": spec.objective,
        "loss_terms": loss_terms_for(args, autoregressive),
        "loss_input": "clean" if autoregressive else "corrupted",
        "stage": args.stage,
        "seed": args.seed,
        "architecture": config.to_record(),
        "microbatch": args.microbatch,
        "grad_accum": grad_accum,
        "world_size": world,
        "nnodes": args.nnodes,
        "seq": seq,
        "global_batch": global_batch,
        "validated_global_batch": TARGET_GLOBAL_BATCH,
        "global_batch_matches_validated_recipe": global_batch == TARGET_GLOBAL_BATCH,
        "steps": args.steps,
        "lr": args.lr,
        # The schedule, not just its peak.  Two runs at the same ``lr`` with
        # different warmup or decay are different experiments, and the paper's
        # training-exposure table is a transcription of this record.
        "warmup_steps": args.warmup_steps,
        "lr_final_fraction": args.lr_final_fraction,
        "lr_schedule": "linear warmup then cosine decay to lr_final_fraction of peak",
        "lambda_causal": args.lambda_causal,
        "lambda_rollout": args.lambda_rollout,
        "block_size": args.block_size,
        "span_prob": args.span_prob,
        # Read by ``collect_configuration``, which had no source for either and
        # published them as null.  ``rollout_batch_fraction`` is a declared
        # training choice and ``weight_decay`` an optimizer one, so an absent key
        # is an unaudited run rather than a cosmetic gap.
        "rollout_batch_fraction": args.rollout_batch_fraction,
        "weight_decay": args.weight_decay,
        # Precision and recomputation are effective-configuration facts the
        # collector states as prose.  Recording them here is what makes that
        # sentence checkable against the run instead of asserted about it -- and
        # the parameter dtype is READ OFF THE MODEL, not restated: a literal
        # "float32" here would be the same kind of claim as the hard-coded
        # precision string in ``collect.py`` it exists to replace, and would
        # survive the load site silently reverting to the bf16 default.  The
        # embedding is the largest tensor and the one the frozen-update
        # measurement was taken on, so a mixed-dtype tree cannot read as fp32.
        "parameter_dtype": str(next(raw_model.parameters()).dtype).replace(
            "torch.", ""),
        "parameter_dtypes_distinct": sorted(
            {str(p.dtype).replace("torch.", "")
             for p in raw_model.parameters()}),
        "autocast_dtype": "bfloat16" if torch.cuda.is_available() else "off",
        "gradient_checkpointing": bool(raw_model.gradient_checkpointing),
        "causal_aux_positions_per_microbatch": args.causal_aux_positions,
        "loop_multiplicities": list(config.loop_multiplicities),
        "loop_ramp_steps": args.loop_ramp_steps,
        "loop_sampling_distribution": (["R=1 during ramp"] if args.loop_ramp_steps else [])
        + [f"uniform over {list(config.loop_multiplicities)} after ramp"],
        "recycle_range": [config.recycle_lo, config.recycle_hi],
        "checkpoint": str(args.checkpoint),
        "pack_dir": args.pack_dir,
        "package_digest": package_digest(Path(__file__).resolve().parents[1]),
        # Non-null only when a resume crossed a code change under
        # --allow-package-change.  A single digest field would name the code that
        # wrote the last step and quietly disown the code that wrote the rest.
        "resumed_under_package_digest": resumed_under,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if rank == 0:
        (outdir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n",
                                                encoding="utf-8")

    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    # **R must be the same on every rank, unlike the corruption.**  The draw above
    # is deliberately per-rank -- identical mask positions on all 32 ranks would be
    # a real regression -- but one generator served both, so each rank drew its own
    # multiplicity.  Every rank then waits on the slowest: P(some rank draws 4) at
    # 32 ranks is 0.999998, so essentially every post-ramp step costs 60 block
    # passes against an intended mean of 40, a third of the wall clock for 84% of
    # the run.  And ``per_r_losses`` records rank 0's R only, so the liveness gate
    # would describe one rank of 32.
    loop_generator = torch.Generator(device=device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    step = start_step
    started = time.time()
    tokens_seen = 0
    canvas_visits = 0
    causal_aux_positions = 0
    ar_pairs_seen = 0
    per_r_losses: dict[int, list[float]] = {r: [] for r in config.loop_multiplicities}

    iterator = iter(loader) if loader is not None else None

    def next_batch():
        nonlocal iterator
        if iterator is None:
            raise Refusal("no corpus: this run was started with --synthetic")
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    model.train()
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_ce = 0.0
        for micro in range(grad_accum):
            if args.synthetic:
                length = seq
                rows = torch.randint(1, 60000, (args.microbatch, length),
                                     device=device, dtype=torch.long)
                row_mask = torch.ones(args.microbatch, length, device=device,
                                      dtype=torch.bool)
                row_docs = torch.tensor([[0, length // 2]] * args.microbatch,
                                        device=device, dtype=torch.long)
            else:
                batch = next_batch()
                rows = batch["input_ids"].to(device)
                row_mask = batch["attention_mask"].to(device)
                row_docs = batch["doc_starts"].to(device)
                # Fold the loader's per-row counts into the PARENT's copy of
                # CorpusStats.  ``num_workers=4`` forks the dataset, so the
                # counters ``__getitem__`` bumps live in a child that exits
                # before the record is written; folding here is the only path
                # that is right under every worker count.  See the docstring of
                # ``longrwkv.data.packed_dataset``.
                if dataset is not None and "counts" in batch:
                    dataset.stats.fold(batch["counts"])

            # The whole microbatch, folded into the [1, B*T] row the varlen kernel
            # requires.  Taking ``rows[0:1]`` here instead -- which is what this
            # loop did -- trains on one row of ``microbatch`` while every recorded
            # token count multiplies the declared microbatch, so a 3B-token run
            # would report 3B and have seen 375M.
            ids, attention, doc_starts, ctx = flatten_microbatch(
                rows, row_mask, row_docs)

            clean = ids.clone()
            if autoregressive:
                # **The autoregressive arm never sees a corrupted canvas.**  It is
                # the paper's "autoregressive adaptation of the same base": clean
                # input, next-token CE over every valid pair, one forward, no
                # masked term and no auxiliary.  Three things this branch fixes,
                # each of which was independently enough to void the baseline:
                #
                #  * the loss was masked CE at weight 1.0 with next-token CE as a
                #    0.1 auxiliary, i.e. the M6 objective;
                #  * the causal term's own forward was fed ``corrupted``, so even
                #    the auxiliary never saw undamaged context -- a next-token
                #    predictor reading MASK where its own prefix should be;
                #  * the pairs were subsampled to 4096 of ~32k, which is unbiased
                #    for an auxiliary's mean and simply a different objective as a
                #    primary loss.
                #
                # ``block_t``/``codes`` are not built: an AR arm is ``causal_only``
                # and not ``gate_mod``, so the model requires neither, and there is
                # no noise coordinate to condition on when nothing was corrupted.
                valid = causal_next_token_valid(doc_starts, attention)
                pairs = valid[0].nonzero(as_tuple=False).flatten()
                R = 1
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                    if pairs.numel():
                        ar_logits = model(clean, ctx=ctx, R=R, force_forward=True,
                                          gather_idx=pairs,
                                          block_size=args.block_size)
                        total, diag = next_token_ce_global(
                            ar_logits, clean[0].index_select(0, pairs + 1))
                    else:
                        total = torch.zeros((), device=device, requires_grad=True)
                        diag = {"ar_ce": 0.0, "n_pairs": 0}
                ar_pairs_seen += int(pairs.numel())
                step_ce += diag["ar_ce"]
                per_r_losses.setdefault(R, []).append(diag["ar_ce"])
                (total / grad_accum).backward()
                step_loss += float(total.detach())
                tokens_seen += int(attention.sum())
                canvas_visits += int(ids.numel())
                continue

            corrupted, mask, _, block_t = sample_corruption(
                ids, attention, block_size=args.block_size, span_prob=args.span_prob,
                generator=generator, mask_id=MASK_ID, pad_id=PAD_ID)
            R = loop_multiplicity(step, args.loop_ramp_steps,
                                  config.loop_multiplicities, loop_generator, device) \
                if ARM_SPECS[args.arm].hidden_loop else 1

            codes = vis_codes(corrupted, attention, mask, MASK_ID)
            selected = mask[0].nonzero(as_tuple=False).flatten()

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                if selected.numel():
                    logits = model(corrupted, ctx=ctx, block_t=block_t, codes=codes,
                                   R=R, gather_idx=selected,
                                   block_size=args.block_size)
                else:
                    logits = None
                aux = 0.0
                if args.lambda_causal > 0.0:
                    # **The auxiliary is gathered, like the main loss.**  Calling
                    # the model with no ``gather_idx`` returns ``[1, N, V]``: at
                    # N = microbatch 8 x seq 4096 = 32768 and V = 65536 that is
                    # 4.0 GiB in bf16, the ``.float()`` view another 8.0, and
                    # ``cross_entropy``'s own log_softmax 8.0 more -- 20 GiB of
                    # transient logits held until backward, on top of the main
                    # loss's 10 GiB and both directions' activations.  The
                    # comment at ``model.py``'s gathered head says it plainly:
                    # the alternative does not fit.
                    #
                    # Sampling positions keeps the estimator unbiased for the
                    # same mean: the auxiliary is a uniform average over valid
                    # next-token pairs, so a uniform subsample has that average
                    # as its expectation.  What it costs is variance, which is
                    # the right thing to spend on an auxiliary carrying
                    # lambda = 0.1 -- and the count is recorded, so the run is
                    # auditable rather than quietly cheaper.
                    valid = causal_next_token_valid(doc_starts, attention)
                    pairs = valid[0].nonzero(as_tuple=False).flatten()
                    if pairs.numel():
                        cap = args.causal_aux_positions
                        if 0 < cap < pairs.numel():
                            pick = torch.randperm(pairs.numel(), generator=generator,
                                                  device=device)[:cap]
                            pairs = pairs.index_select(0, pick)
                        causal_aux_positions += int(pairs.numel())
                        causal_logits = model(corrupted, ctx=ctx, block_t=block_t,
                                              codes=codes, R=R, force_forward=True,
                                              gather_idx=pairs,
                                              block_size=args.block_size)
                        # ``pairs`` indexes the predicting position i; the target
                        # is i+1, and ``causal_next_token_valid`` has already
                        # ruled out every i whose successor begins a document or
                        # is padding.
                        aux = F.cross_entropy(
                            causal_logits.float(),
                            clean[0].index_select(0, pairs + 1).long())

            if logits is not None:
                loss, diag = selected_token_ce_global(logits, clean[0].index_select(0, selected))
                step_ce += diag["mask_ce"]
                per_r_losses.setdefault(R, []).append(diag["mask_ce"])
            else:
                loss, diag = torch.zeros((), device=device, requires_grad=True), {"mask_ce": 0.0}
            total = loss + args.lambda_causal * aux
            (total / grad_accum).backward()
            step_loss += float(total.detach())
            tokens_seen += int(attention.sum())
            canvas_visits += int(ids.numel())

        if torch.cuda.is_available():
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1

        if rank == 0 and step % args.log_every == 0:
            elapsed = time.time() - started
            per_r = {r: (sum(v[-20:]) / max(1, len(v[-20:]))) for r, v in per_r_losses.items() if v}
            # The CE key names the objective that produced it.  A shared
            # ``mask_ce`` key is how the one artifact a human actually watches --
            # the run log -- showed an autoregressive arm and a denoiser as the
            # same measurement, and ``qz/scaling_leg.py`` compares curves by this
            # key, so a mislabelled one would be compared against the wrong lane.
            ce_key = "ar_ce" if autoregressive else "mask_ce"
            print(json.dumps({
                "step": step, "loss": step_loss / grad_accum,
                ce_key: step_ce / grad_accum, "lr": lr,
                "tokens": tokens_seen, "tokens_per_s": tokens_seen / max(elapsed, 1e-9),
                f"per_r_{ce_key}": per_r,
            }), flush=True)

        if rank == 0 and args.save_every and step % args.save_every == 0:
            save_checkpoint(outdir, raw_model, optimizer, scaler_state, step, record,
                            args.keep_checkpoints)

    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["wall_seconds"] = time.time() - started
    record["gpu_hours"] = record["wall_seconds"] * world / 3600.0

    # Token counts are per-rank; the paper reports the run's total.  Summing across
    # the cohort is the difference between "3.0B tokens" and rank 0's 94M at world
    # 32, and the understatement would be invisible -- a plausible number in the
    # right units, written into the training-exposure table.  The counters are
    # derived from the tensors that were actually served (not from the declared
    # microbatch), so this sum is a measurement rather than a restatement of the
    # config.
    local_counts = torch.tensor([float(tokens_seen), float(canvas_visits)],
                                device=device)
    if distributed:
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
    record["nonpad_tokens"] = int(local_counts[0].item())
    record["canvas_visits"] = int(local_counts[1].item())
    record["nonpad_tokens_rank0"] = tokens_seen
    record["tokens_counted_across_ranks"] = world if distributed else 1
    # Same key discipline as the log line: the per-R CE of an autoregressive arm
    # is not a masked CE, and the liveness criterion the plan states ("mask_ce(R=2)
    # within 0.1 nat of mask_ce(R=1)") is about the hidden loop, which an AR arm
    # does not have.  A0's record read ``final_per_r_mask_ce {"1": 6.18, "2": null,
    # "4": null}``, which is simultaneously the wrong name for the number and an
    # invitation to read two absent multiplicities as a failed liveness gate.
    if autoregressive:
        record["final_ar_ce"] = (sum(per_r_losses.get(1, [])) /
                                 len(per_r_losses[1])) if per_r_losses.get(1) else None
        record["ar_pairs_seen_rank0"] = ar_pairs_seen
        record["hidden_loop"] = False
    else:
        record["final_per_r_mask_ce"] = {str(r): (sum(v) / len(v) if v else None)
                                         for r, v in per_r_losses.items()}
    if dataset is not None:
        # Rank-local by construction: each rank draws a disjoint slice, so these
        # describe THIS rank's exposure.  The name says so, because the
        # training-exposure table wants the cohort's total and a reader who found
        # ``corpus`` there would have divided the corpus by 32 without knowing it.
        record["corpus_rank0"] = dataset.stats.as_record()
        local = dataset.stats
        corpus_counts = torch.tensor(
            [float(local.tokens_served), float(local.nonpad_tokens_served),
             float(local.legacy_id_remapped), float(local.rows_served)],
            device=device)
        if distributed:
            dist.all_reduce(corpus_counts, op=dist.ReduceOp.SUM)
        served = int(corpus_counts[0].item())
        record["corpus"] = {
            "tokens_served": served,
            "nonpad_tokens_served": int(corpus_counts[1].item()),
            "legacy_id_remapped": int(corpus_counts[2].item()),
            # ``None`` when nothing was served, never 0.0 -- see CorpusStats.
            "remapped_fraction": (int(corpus_counts[2].item()) / served
                                  if served else None),
            "rows_served": int(corpus_counts[3].item()),
            "files_read": len(local.files),
            "ranks_summed": world if distributed else 1,
        }
    if rank == 0:
        if args.save_every:
            save_checkpoint(outdir, raw_model, optimizer, scaler_state, step, record,
                            args.keep_checkpoints)
        (outdir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n",
                                                encoding="utf-8")
        print(json.dumps({"done": True, "step": step,
                          "wall_seconds": record["wall_seconds"]}), flush=True)
    if distributed:
        dist.destroy_process_group()
    return 0


def save_checkpoint(outdir: Path, model, optimizer, scaler_state, step: int,
                    record: dict, keep: int) -> None:
    """Atomic write, then prune to the most recent ``keep`` checkpoints.

    Written to a temporary name and renamed, so a fault-tolerance restart never
    resumes from a half-written file.

    The checkpoint's digest is written beside it because the run registry's
    "SHA / SHA" column is code sha *and* checkpoint sha, and nothing in the
    package wrote the second one: ``collect.py`` read ``checkpoint_sha.json``,
    ``latex_tables.sha_pair`` required both halves to be non-empty strings, so
    every registry SHA cell would have stayed pending however many runs
    finished.  Hashing the file we just renamed -- not the state dict in memory
    -- means the digest identifies the bytes a later job would actually load.
    """
    target = outdir / f"step_{step:08d}.pt"
    temporary = target.with_suffix(".pt.partial")
    torch.save({"model": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scaler": scaler_state,
                "step": step, "record": record}, temporary)
    temporary.rename(target)
    (outdir / "checkpoint_sha.json").write_text(
        json.dumps({"checkpoint": target.name, "step": step,
                    "sha256": file_digest(target),
                    "bytes": target.stat().st_size,
                    # The digest covers the weights *and* the optimizer state,
                    # because that whole file is what ``--resume-from`` loads.
                    "covers": "torch.save payload: model, optimizer, scaler, "
                              "step, record"}, indent=2) + "\n",
        encoding="utf-8")
    checkpoints = sorted(outdir.glob("step_*.pt"))
    for stale in checkpoints[:-keep] if keep > 0 else []:
        stale.unlink(missing_ok=True)
    (outdir / "latest.json").write_text(
        json.dumps({"step": step, "checkpoint": target.name}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
