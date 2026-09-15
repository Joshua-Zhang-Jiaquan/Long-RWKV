"""Offline larger-batch sampler eval for the BiRWKV token-diffusion denoiser.

Settles the code/plan section 4.6 "iterative beats single-step" gate with real
sample sizes: grid of mask_ratio x denoise-steps over held-out sequences from
one or more packed-pkl dirs, emitting qz_capability_sampler_shard_v1 records
so merge_eval.py produces bootstrap CIs per (arm = ratio/steps combo).

One GPU per shard; the launcher strides shards across GPUs like the other
capability evals.

Usage:
  python -m eval.capability.sampler_eval_offline \
    --ckpt_dir <step_dir> --model_dir <hf-dir> \
    --token_dirs dirA,dirB --per_dir 512 --window 1024 \
    --shard 0 --num_shards 8 --output <shard.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_SCALE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))

MASK_RATIOS = (0.3, 0.5, 0.7, 0.9)
STEP_GRID = (1, 8, 16, 32, 64)
# The default grid stops at 0.9, but FREE GENERATION is the ~1.0 regime: a clean
# prompt with the whole completion masked. The 0.9 measurement came back
# em@32-em@1 = -0.0252 (CI-disjoint) where 0.3/0.5/0.7 were all positive, so the
# one regime that matters for code generation is also the one where iterative
# denoising is destructive -- and it was never measured directly. `--mask_ratios`
# exists to measure it instead of extrapolating. The DEFAULT is unchanged so
# already-merged arms stay comparable.


def _load_tail_sequences(token_dir: str, count: int, window: int) -> list[torch.Tensor]:
    """Take `count` windows from the TAIL of the pack (training holds out the tail)."""
    from data.fineweb4096_packed import FineWebPackedPickleDataset

    ds = FineWebPackedPickleDataset(data_dir=token_dir, max_length=4096,
                                    pad_token_id=0, cache_shards=1)
    n = len(ds)
    out = []
    for i in range(max(0, n - count), n):
        ids = ds[i]["input_ids"][:window]
        out.append(ids)
    return out


def _repetition(row: torch.Tensor, pad_id: int = 0) -> tuple[float, float]:
    """Return ``(max_run / length, distinct_fraction)`` for one generated row.

    The commits-per-step finding (Spearman rho = -1.000, R^2 = 0.891) was computed ad-hoc
    over 6-12 raw HumanEval dumps and **never existed as a harness metric**. Phase 1a needs
    it on held-out text with CIs, so it lives here now.

    ``max_run / length`` rather than a raw repetition count: the length-naive version
    reversed the sign of the self-correction and canvas readings once outputs of different
    lengths were compared, because a short degenerate output and a long one score
    incomparably. Normalising by realised length is what made those readings stable.
    """
    toks = row[row != pad_id]
    n = int(toks.numel())
    if n == 0:
        return float("nan"), float("nan")
    # longest run of an identical token
    best = run = 1
    for i in range(1, n):
        run = run + 1 if toks[i] == toks[i - 1] else 1
        best = max(best, run)
    distinct = float(torch.unique(toks).numel()) / n
    return best / n, distinct


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--token_dirs", required=True, help="comma-separated packed-pkl dirs")
    p.add_argument("--per_dir", type=int, default=512)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--self_correction", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--steps_grid", default="",
                   help="override STEP_GRID (comma-separated). The g-sweep costs ~one forward "
                        "per committed token at g=1 (21x the legacy grid at ratio 0.5), so a "
                        "commit-group sweep must restrict the step grid rather than run the "
                        "full default. Empty = the default grid.")
    p.add_argument("--commit_groups", default="0",
                   help="comma-separated commit_group_size values to sweep (v5.2 B2/Q1-U2). "
                        "0 = legacy single-forward multi-commit (product of marginals); g>=1 "
                        "commits at most g positions per forward and refreshes the canvas "
                        "between groups. Each value becomes its own arm suffix _g<G>.")
    p.add_argument("--mask_ratios", default="",
                   help="comma-separated mask ratios (default: the built-in grid). "
                        "Use 0.95,1.0 to measure the free-generation regime directly.")
    args = p.parse_args()

    ratios = MASK_RATIOS
    if args.mask_ratios:
        ratios = tuple(float(x) for x in args.mask_ratios.split(",") if x.strip())
        if not all(0.0 < r <= 1.0 for r in ratios):
            msg = f"mask ratios must lie in (0, 1]: {ratios}"
            raise ValueError(msg)

    from eval.capability.birwkv_diffusion_model import load_birwkv_diffusion
    from models.birwkv7_diffusion import MASK_TOKEN_ID, iterative_denoise, kendall_tau_commit_order

    device = torch.device("cuda")
    loaded = load_birwkv_diffusion(args.ckpt_dir, args.model_dir)
    model = loaded.model

    # gather sequences, tagged by source dir name
    seqs: list[tuple[str, torch.Tensor]] = []
    for d in args.token_dirs.split(","):
        d = d.strip()
        if not d:
            continue
        tag = Path(d).name
        for ids in _load_tail_sequences(d, args.per_dir, args.window):
            seqs.append((tag, ids))
    # Shard by index, KEEPING the global index. The global index is what makes
    # document_id unique across shards: with a shard-local `start + row` label,
    # every shard emitted ":0"..":31" for DIFFERENT sequences, and merge_eval --
    # which keys on document_id -- deduplicated 7 of every 8 records. The merge
    # reported 640 records where the shards held 5120, and it exited 0.
    seqs = [(gi, tag, v) for gi, (tag, v) in enumerate(seqs)
            if gi % args.num_shards == args.shard]
    print(f"[sampler-eval] shard {args.shard}/{args.num_shards}: {len(seqs)} sequences", flush=True)

    step_grid = (tuple(int(x) for x in str(args.steps_grid).split(",") if x.strip() != "")
                 or STEP_GRID)
    if any(x < 1 for x in step_grid):
        msg = f"--steps_grid entries must be >= 1, got {step_grid}"
        raise ValueError(msg)
    if 1 not in step_grid:
        # em@1 is the single-shot baseline every iterative-gain claim is measured against;
        # the Gate-0 launcher already refuses a grid without it.
        print(f"[sampler-eval] WARNING: steps_grid={step_grid} has no 1 -- "
              "em@k - em@1 cannot be computed from this shard alone", flush=True)
    groups = tuple(int(g) for g in str(args.commit_groups).split(",") if g.strip() != "")
    if not groups:
        groups = (0,)
    if any(g < 0 for g in groups):
        msg = f"--commit_groups must be >= 0, got {groups}"
        raise ValueError(msg)
    print(f"[sampler-eval] commit_groups={groups} (0 = legacy product-of-marginals)", flush=True)
    # DEGENERATE-CELL WARNING. The group loop splits the PER-STEP commit budget
    # k = round(ratio*window/steps), so any g >= k yields exactly ONE group per step and
    # reproduces the legacy path bit-for-bit -- same output, same forward count (verified
    # against the code, not inferred). Such a cell costs full GPU time to re-measure the
    # baseline and, worse, reads as a real arm in the merged table. The free-generation
    # sweep spent one of six cells this way (g=16 at steps=32, k=16) before this warned.
    for _r in ratios:
        for _st in step_grid:
            _k = max(1, int(_r * args.window) // max(1, _st))
            for _g in groups:
                if _g > 0 and _g >= _k:
                    print(f"[sampler-eval] WARNING degenerate cell r{int(_r*100)}_s{_st}"
                          f"_g{_g}: g >= per-step budget k={_k}, so this arm IS the legacy "
                          "path and will duplicate g=0. Drop it or lower g.", flush=True)

    records: list[dict] = []
    t0 = time.time()
    for ratio in ratios:
      for steps in step_grid:
        for grp in groups:
            # Arm name keeps the legacy shape EXACTLY when grp == 0, so previously merged
            # sampler results stay comparable by arm name rather than needing a remap.
            arm = (f"r{int(ratio * 100)}_s{steps}"
                   + ("_sc" if args.self_correction else "")
                   + (f"_g{grp}" if grp != 0 else ""))
            for start in range(0, len(seqs), args.batch):
                chunk = seqs[start:start + args.batch]
                gidx = [g for g, _, _ in chunk]
                tags = [t for _, t, _ in chunk]
                ids = torch.stack([v for _, _, v in chunk]).to(device)
                # Seed does NOT include grp: every commit-group arm must see the SAME
                # corruption as its g=0 baseline, or the sweep would confound the grouping
                # with a different mask. Same discipline as the alpha:=1.0 paired arm.
                gen = torch.Generator(device=device).manual_seed(
                    args.seed * 100000 + int(ratio * 100) * 1000 + steps)
                eligible = ids.ne(0)
                mask = (torch.rand(ids.shape, device=device, generator=gen) < ratio) & eligible
                corrupted = torch.where(mask, torch.full_like(ids, MASK_TOKEN_ID), ids)
                with torch.no_grad():
                    denoised, commit_step = iterative_denoise(
                        model, corrupted, mask, steps=steps,
                        self_correction=args.self_correction,
                        commit_group_size=grp,
                    )
                match = (denoised == ids) & mask
                for row in range(ids.shape[0]):
                    m = int(mask[row].sum())
                    em = float(match[row].sum()) / max(1, m)
                    tau = kendall_tau_commit_order(
                        commit_step[row:row + 1], mask[row:row + 1]) if steps > 1 else 0.0
                    residue = int((denoised[row] == MASK_TOKEN_ID).sum())
                    maxrun, distinct = _repetition(denoised[row])
                    records.append({
                        # GLOBAL sequence index, not the shard-local offset.
                        "document_id": f"{tags[row]}:{gidx[row]}",
                        "arm": arm,
                        "seed": args.seed,
                        "metrics": {"em": em, "tau": tau, "residue": float(residue),
                                    "max_run_frac": maxrun, "distinct_frac": distinct},
                        "isolation_mode": "none",
                        "failure": None,
                    })
            cps = "n/a" if grp == 0 else f"{grp}/fwd"
            print(f"[sampler-eval] {arm}: done (commits {cps}, {time.time() - t0:.0f}s)",
                  flush=True)

    payload = {
        "schema": "qz_capability_sampler_shard_v1",
        "task": "sampler_reconstruction",
        "task_kind": "sampler",
        "checkpoint": str(Path(args.ckpt_dir).resolve()),
        "checkpoint_step": loaded.step,
        "registry_hash": os.environ.get("CAPABILITY_REGISTRY_HASH", "unpinned"),
        "profile_sha256": os.environ.get("CAPABILITY_PROFILE_SHA256", "unpinned"),
        "condition_profile_hash": os.environ.get("CAPABILITY_CONDITION_PROFILE_HASH", "unpinned"),
        "seeds": [args.seed],
        "metric_schema": ["em", "tau", "residue", "max_run_frac", "distinct_frac"],
        "shard_index": args.shard,
        "num_shards": args.num_shards,
        "n_records": len(records),
        "records": records,
        "grid": {"mask_ratios": list(ratios), "steps": list(step_grid),
                 "commit_groups": list(groups),
                 "window": args.window, "self_correction": args.self_correction},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, str(out))
    print(f"[sampler-eval] wrote {out} ({len(records)} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
