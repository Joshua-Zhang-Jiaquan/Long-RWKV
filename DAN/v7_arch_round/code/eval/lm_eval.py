"""Language-model likelihood eval for the BiRWKV token-diffusion denoiser.

Phase-1 (north-star) scorer: measure, then improve, the model's core LM ability
before touching pg19/RACE (phase 2) or code (phase 3).

Two likelihoods, both reported per sequence in nats/token (plus ``ppl = exp``
and ``bits = nll / ln2``):

* ``causal`` — next-token CE on the forward-only stream (``force_forward=True``).
  This is the standard language-model perplexity and is directly comparable to
  the frozen RWKV7-Goose-2.9B base, which this scorer also emits as a control
  arm (``--ckpt_dir base`` loads the warm-start geometry with no training
  state-dict, so ``force_forward`` reproduces the pretrained causal RWKV-7).
* ``mc<R>``  — masked-CE (denoising pseudo-likelihood): mask ratio ``R`` of each
  sequence, one bidirectional forward pass, CE over the masked positions only.
  This is the training objective and the diffusion-side likelihood proxy.
* ``mc_elbo``— equally-weighted average over the ratio grid (a single
  convenience number; the per-ratio ``mc<R>`` arms are the honest evidence).

Emits ``qz_capability_lm_shard_v1`` records (one per document per arm) so
``merge_eval.py`` folds them into bootstrap-CI summaries per arm. Requires CUDA
(fla kernels); runs inside a qz job via ``launch_capability_eval.sh`` TASK=lm.

Usage::

    python -m eval.capability.lm_eval \
      --ckpt_dir <step_dir | base> --model_dir <hf-dir> \
      --corpus wikitext103 --corpus_dir <npz dir> \
      --ratios 0.15,0.3,0.5,0.7,0.9 --max_samples 2000 --batch 16 \
      --shard 0 --num_shards 8 --output <shard.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_SCALE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))

MASK_TOKEN_ID = 65535  # unused slot in the RWKV World vocab (EOS=65530)
PAD_TOKEN_ID = 0
_DEFAULT_RATIOS = (0.15, 0.3, 0.5, 0.7, 0.9)
_BASE_SENTINEL = "base"


def _load_corpus(
    corpus_dir: str,
    shard: int,
    num_shards: int,
    max_samples: int | None,
    window: int,
    pack: int = 0,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Load pre-tokenized ``.npz`` docs assigned to this shard.

    Each npz has ``input_ids [T] int32`` and ``attention_mask [T] bool``
    (the pg19 / wikitext103 / lambada preprocessing convention on GPFS).
    ``max_samples`` is a per-shard cap applied after striding.

    ``pack`` > 0 concatenates **consecutive** chunks into sequences of that many
    tokens before sharding — the phase-2a long-context arm. pg19's ``tokens/``
    dirs are one contiguous token stream cut into fixed 512-token chunks (no
    document-boundary marker at chunk edges), so joining neighbours reconstructs
    real long-range context rather than stitching unrelated text. Sharding runs
    over the *packed* units so every shard still gets whole sequences.
    """
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.npz")))
    np = __import__("numpy")

    def _read(fp: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(fp, allow_pickle=True)
        return (
            torch.from_numpy(data["input_ids"].astype("int64")),
            torch.from_numpy(data["attention_mask"].astype(bool)),
        )

    docs: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    if pack and pack > 0:
        # Group consecutive files into packs, then stride over packs.
        per_pack: int | None = None
        pack_index = 0
        cursor = 0
        while cursor < len(files):
            if per_pack is None:
                probe_ids, _ = _read(files[cursor])
                chunk_len = max(1, int(probe_ids.numel()))
                per_pack = max(1, pack // chunk_len)
            group = files[cursor : cursor + per_pack]
            cursor += per_pack
            if pack_index % num_shards != shard:
                pack_index += 1
                continue
            pack_index += 1
            if max_samples is not None and len(docs) >= max_samples:
                break
            id_parts, am_parts = [], []
            for fp in group:
                ids_i, am_i = _read(fp)
                id_parts.append(ids_i)
                am_parts.append(am_i)
            ids = torch.cat(id_parts)[:pack]
            am = torch.cat(am_parts)[:pack]
            stem = f"{Path(group[0]).stem}+{len(group)}x"
            docs.append((stem, ids, am))
        return docs

    for idx, fp in enumerate(files):
        if idx % num_shards != shard:
            continue
        if max_samples is not None and len(docs) >= max_samples:
            break
        ids, am = _read(fp)
        if window and ids.numel() > window:
            ids = ids[:window]
            am = am[:window]
        docs.append((Path(fp).stem, ids, am))
    return docs


def _pad_batch(
    chunk: list[tuple[str, torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.nn.utils.rnn.pad_sequence(
        [c[1] for c in chunk], batch_first=True, padding_value=PAD_TOKEN_ID
    ).to(device)
    am = torch.nn.utils.rnn.pad_sequence(
        [c[2] for c in chunk], batch_first=True, padding_value=False
    ).to(device)
    return ids, am


@torch.no_grad()
def _causal_nll(
    model: object, docs: list[tuple[str, torch.Tensor, torch.Tensor]], batch: int, device: torch.device
) -> list[float]:
    """Next-token CE (nats/token) on the forward-only stream."""
    out: list[float] = []
    for start in range(0, len(docs), batch):
        chunk = docs[start : start + batch]
        ids, am = _pad_batch(chunk, device)
        logits = model(ids, force_forward=True).float()  # [B, T, V]
        tgt = ids[:, 1:]
        valid = am[:, 1:] & tgt.ne(PAD_TOKEN_ID)
        # Gather at valid positions only. This arm does not -inf the pad logit, so
        # `ce * valid` would also work — but selecting keeps it structurally
        # immune to the `inf * 0 = NaN` trap that hit _masked_ce on
        # variable-length corpora, and skips wasted CE on padding.
        shifted = logits[:, :-1]
        per_seq = torch.zeros(ids.shape[0], device=device)
        cnt = valid.sum(dim=1)
        if int(cnt.sum()) > 0:
            b_idx, t_idx = valid.nonzero(as_tuple=True)
            sel_ce = F.cross_entropy(
                shifted[b_idx, t_idx], tgt[b_idx, t_idx], reduction="none"
            )
            per_seq = per_seq.index_add(0, b_idx, sel_ce)
        nll = torch.where(
            cnt > 0, per_seq / cnt.clamp_min(1), torch.full_like(per_seq, float("nan"))
        )
        out.extend(nll.cpu().tolist())
    return out


@torch.no_grad()
def _masked_ce(
    model: object,
    docs: list[tuple[str, torch.Tensor, torch.Tensor]],
    ratio: float,
    seed: int,
    batch: int,
    device: torch.device,
    force_forward: bool = False,  # noqa: FBT001, FBT002
) -> list[float]:
    """Denoising pseudo-likelihood: masked-CE over the masked positions at ``ratio``.

    ``force_forward=True`` disables the reverse stream, which is **algebraically exactly
    the ``alpha := 1.0`` ablation**: the block computes ``o = alpha*o_fwd + (1-alpha)*o_bwd``
    normally, but under ``force_forward`` it takes the ``o = o_fwd`` branch outright.

    This exists to fill the missing cell of a 2x2 that the pre-existing arms cannot fill.
    The ``causal`` arm is (next-token objective x forward-only) and the ``mc<R>`` arm is
    (masked objective x bidirectional), so they differ in **two** variables at once --
    exactly the confound that leaves v4 section 1.1's "~1 nat bidirectional necessity"
    unable to isolate the reverse stream's contribution. Scoring the masked objective
    with the reverse stream off holds the objective, corruption, seed and documents fixed
    and varies only the stream.

    Motivation (v5.2 Phase 0a, measured): the fusion gate alpha never left its 0.9820
    init at either 2.9B or 0.4B, so the reverse stream carries only ~1-3% of the fused
    output. Whether that 1-3% is worth its ~934M parameters is precisely what this arm
    decides: a large mask-CE jump means the small weight has a large effect, a negligible
    one means the reverse stream is near-inert as fused.
    """
    out: list[float] = []
    vocab = model.lm_head.out_features
    for start in range(0, len(docs), batch):
        chunk = docs[start : start + batch]
        ids, am = _pad_batch(chunk, device)
        gen = torch.Generator(device=device).manual_seed(
            seed * 100000 + int(round(ratio * 1000)) + start
        )
        eligible = am & ids.ne(PAD_TOKEN_ID)
        mask = (torch.rand(ids.shape, device=device, generator=gen) < ratio) & eligible
        corrupted = torch.where(mask, torch.full_like(ids, MASK_TOKEN_ID), ids)
        logits = model(corrupted, force_forward=force_forward).float()  # [B, T, V]
        # never "predict" mask/pad — the target is always a real token here.
        logits[..., MASK_TOKEN_ID] = float("-inf")
        logits[..., PAD_TOKEN_ID] = float("-inf")
        # Gather CE at the masked positions ONLY. Scoring the full padded grid
        # would make pad positions cost +inf (their target IS PAD, whose logit we
        # just set to -inf), and `inf * 0` in the masked sum is NaN — which is
        # exactly how every variable-length corpus (lambada) returned NaN while
        # the uniform-512 corpora looked fine.
        per_seq = torch.zeros(ids.shape[0], device=device)
        cnt = mask.sum(dim=1)
        if int(cnt.sum()) > 0:
            b_idx, t_idx = mask.nonzero(as_tuple=True)
            sel_ce = F.cross_entropy(
                logits[b_idx, t_idx], ids[b_idx, t_idx], reduction="none"
            )
            per_seq = per_seq.index_add(0, b_idx, sel_ce)
        # A sequence with no masked position has no pseudo-likelihood to report:
        # emit NaN so merge_eval's mean is not silently biased toward 0.
        nll = torch.where(
            cnt > 0, per_seq / cnt.clamp_min(1), torch.full_like(per_seq, float("nan"))
        )
        out.extend(nll.cpu().tolist())
    return out


def _metrics(nll: float) -> dict[str, float]:
    return {"nll": nll, "ppl": math.exp(nll), "bits": nll / math.log(2)}


def _autotune_batch(
    docs: list[tuple[str, torch.Tensor, torch.Tensor]],
    device: torch.device,
    target_frac: float,
    floor: int,
) -> int:
    """Pick the largest batch that fits ``target_frac`` of this GPU's memory.

    The scorers are single-forward-pass and memory scales linearly in
    ``batch * seqlen``, so the batch that fills the card is a division, not a
    search. Calibrated on an H100: bf16 weights are ~8.2 GiB and activations cost
    ~1.20 MiB per token at 512-token sequences.

    Without this, one hardcoded ``--batch`` under-fills every card it does not
    happen to match: BATCH=16 at L=512 peaked at 18 GiB of an 80 GiB H100 (22%),
    and the same value on a 141 GiB H200 would use 13%.
    """
    if device.type != "cuda" or not docs:
        return floor
    total = torch.cuda.get_device_properties(device).total_memory / 2**20  # MiB
    weights = torch.cuda.memory_allocated(device) / 2**20  # model already resident
    seq_len = max(int(d[1].numel()) for d in docs)
    act_per_token = 1.20  # MiB, measured
    budget = total * target_frac - weights
    if budget <= 0:
        return floor
    batch = int(budget / (act_per_token * max(1, seq_len)))
    batch = max(floor, min(batch, len(docs)))
    print(
        f"[lm-eval] autotune batch={batch} (seq_len={seq_len}, gpu_total={total:.0f} MiB, "
        f"weights={weights:.0f} MiB, target={target_frac:.0%})",
        flush=True,
    )
    return batch


def _retry_oom(fn, batch: int, label: str):
    """Run ``fn(batch)``, halving the batch on CUDA OOM instead of losing the job.

    Autotune aims at a memory *fraction* from a measured activation constant, so it
    should fit — but fragmentation, a longer-than-probed sequence, or a shared card
    can still push a large batch over. Halving down to 1 costs a little throughput;
    an uncaught OOM costs the whole sweep (and the ~140 s model load with it).

    Returns ``(result, batch_used)``. The batch is returned because it silently enters
    the corruption seed: ``_masked_ce`` derives its generator from ``start``, which steps
    by ``batch``. Two arms that are meant to be PAIRED therefore see the same corruption
    only if they ran at the same batch, so a caller comparing arms must pin the second
    arm to the first's realised batch rather than the requested one.
    """
    while True:
        try:
            return fn(batch), batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 1:
                raise
            batch = max(1, batch // 2)
            print(f"[lm-eval] CUDA OOM in {label}; retrying at batch={batch}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BiRWKV LM-likelihood eval (causal + diffusion-MC)")
    p.add_argument("--ckpt_dir", required=True,
                   help="step_*/ dir with model.pt, or the literal 'base' for the "
                        "frozen HF warm-start (causal-only control)")
    p.add_argument("--model_dir", required=True, help="HF dir supplying geometry + tokenizer")
    p.add_argument("--corpus", required=True, help="corpus label for document_id (e.g. wikitext103)")
    p.add_argument("--corpus_dir", required=True, help="dir of pre-tokenized .npz examples")
    p.add_argument("--ratios", default="0.15,0.3,0.5,0.7,0.9",
                   help="comma-sep mask ratios for the diffusion-MC arms")
    p.add_argument("--max_samples", type=int, default=None, help="per-shard doc cap")
    p.add_argument("--window", type=int, default=0, help="truncate each doc to first W tokens (0=full)")
    p.add_argument("--pack", type=int, default=0,
                   help="concatenate consecutive chunks into P-token sequences "
                        "(phase-2a long-context arm; 0=off)")
    p.add_argument("--batch", type=int, default=16,
                   help="micro-batch; 0 = autotune to --gpu_mem_frac of the card")
    p.add_argument("--gpu_mem_frac", type=float, default=0.88,
                   help="target GPU-memory fraction when --batch 0 (autotune)")
    p.add_argument("--alpha-one-ablation", action="store_true",
                   help="add a paired mc<R>_fwdonly arm per ratio with the reverse stream "
                        "disabled (algebraically alpha:=1.0). Fills the missing cell of the "
                        "objective x stream 2x2 so the reverse stream's contribution can be "
                        "isolated; see _masked_ce's docstring and v5.2 Phase 0a.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA unavailable; lm_eval needs a qz GPU job.", flush=True)

    ratios = tuple(float(r) for r in args.ratios.split(",") if r.strip())
    is_base = args.ckpt_dir == _BASE_SENTINEL

    # --- Load the model (base = warm-start geometry, no training state dict). ---
    # Gated: concurrent 8-way loads exhaust the pod's host RAM (see load_gate.py).
    # The ckpt path gates inside load_birwkv_diffusion, so only the base branch
    # needs an explicit slot here — never nest the two.
    from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion

    t_load = time.time()
    if is_base:
        from eval.capability.load_gate import load_slot

        with load_slot(label="lm-eval"):
            model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(
                args.model_dir, dtype=torch.bfloat16
            ).to(device).eval()
        step = 0
        ckpt_label = _BASE_SENTINEL
    else:
        from eval.capability.birwkv_diffusion_model import load_birwkv_diffusion

        loaded = load_birwkv_diffusion(args.ckpt_dir, args.model_dir, device=device)
        model = loaded.model
        step = loaded.step
        ckpt_label = str(Path(args.ckpt_dir).resolve())
    print(f"[lm-eval] model ready in {time.time() - t_load:.0f}s", flush=True)

    docs = _load_corpus(args.corpus_dir, args.shard, args.num_shards,
                        args.max_samples, args.window, args.pack)
    print(f"[lm-eval] shard {args.shard}/{args.num_shards}: {len(docs)} docs "
          f"(ckpt={ckpt_label}, base={is_base})", flush=True)

    batch = args.batch if args.batch > 0 else _autotune_batch(
        docs, device, args.gpu_mem_frac, floor=8
    )

    records: list[dict] = []
    t0 = time.time()

    # Causal arm (always).
    causal, _causal_batch = _retry_oom(
        lambda b: _causal_nll(model, docs, b, device), batch, 'causal')
    for (stem, _, _), nll in zip(docs, causal):
        records.append({
            "document_id": f"{args.corpus}:{stem}",
            "arm": "causal",
            "seed": args.seed,
            "metrics": _metrics(nll),
            "isolation_mode": "none",
            "failure": None,
        })
    print(f"[lm-eval] causal done ({time.time() - t0:.0f}s)", flush=True)

    # Diffusion-MC arms (skip for the frozen base — bidir == causal there).
    if not is_base:
        ratio_nlls: dict[float, list[float]] = {}
        for ratio in ratios:
            ratio_nlls[ratio], mc_batch = _retry_oom(
                lambda b, r=ratio: _masked_ce(model, docs, r, args.seed, b, device),
                batch, f'mc{int(round(ratio * 100)):03d}')
            arm = f"mc{int(round(ratio * 100)):03d}"
            for (stem, _, _), nll in zip(docs, ratio_nlls[ratio]):
                records.append({
                    "document_id": f"{args.corpus}:{stem}",
                    "arm": arm,
                    "seed": args.seed,
                    "metrics": _metrics(nll),
                    "isolation_mode": "none",
                    "failure": None,
                })
            # PAIRED alpha:=1.0 ablation (v5.2 Phase 0a falsifier). Identical ratio,
            # seed, batch offsets and documents -- the SAME generator stream, so the
            # corruption mask is bit-identical -- with only the reverse stream removed.
            # `isolation_mode` marks it so merge_eval and any reader can tell the arms
            # apart without parsing the name.
            if args.alpha_one_ablation:
                # Pin to the bidir arm's REALISED batch (mc_batch), not the requested
                # one: batch enters the corruption seed, so an OOM downgrade on one arm
                # only would silently unpair them. force_forward uses strictly less
                # memory (no reverse stream), so this batch is always feasible here.
                fwd_nlls, fwd_batch = _retry_oom(
                    lambda b, r=ratio: _masked_ce(model, docs, r, args.seed, b, device,
                                                  force_forward=True),
                    mc_batch, f'mc{int(round(ratio * 100)):03d}fwd')
                if fwd_batch != mc_batch:
                    msg = (f"alpha:=1.0 pairing BROKEN at ratio {ratio}: bidir ran at "
                           f"batch={mc_batch}, fwdonly at {fwd_batch}. Batch enters the "
                           "corruption seed, so these arms scored different masks and are "
                           "not comparable. Refusing to emit the ablation arm.")
                    raise RuntimeError(msg)
                for (stem, _, _), nll in zip(docs, fwd_nlls):
                    records.append({
                        "document_id": f"{args.corpus}:{stem}",
                        "arm": f"{arm}_fwdonly",
                        "seed": args.seed,
                        "metrics": _metrics(nll),
                        "isolation_mode": "alpha_one",
                        "failure": None,
                    })
                print(f"[lm-eval] {arm}_fwdonly done (alpha:=1.0 ablation)", flush=True)
            print(f"[lm-eval] {arm} done ({time.time() - t0:.0f}s)", flush=True)

        for (stem, _, _), i in zip(docs, range(len(docs))):
            elbo = sum(ratio_nlls[r][i] for r in ratios) / len(ratios)
            records.append({
                "document_id": f"{args.corpus}:{stem}",
                "arm": "mc_elbo",
                "seed": args.seed,
                "metrics": _metrics(elbo),
                "isolation_mode": "none",
                "failure": None,
            })
        print(f"[lm-eval] mc_elbo done ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "schema": "qz_capability_lm_shard_v1",
        "task": f"lm-{args.corpus}",
        "task_kind": "lm",
        "checkpoint": ckpt_label,
        "checkpoint_step": step,
        "registry_hash": os.environ.get("CAPABILITY_REGISTRY_HASH", "unpinned"),
        "profile_sha256": os.environ.get("CAPABILITY_PROFILE_SHA256", "unpinned"),
        "condition_profile_hash": os.environ.get("CAPABILITY_CONDITION_PROFILE_HASH", "unpinned"),
        "seeds": [args.seed],
        "metric_schema": ["nll", "ppl", "bits"],
        "shard_index": args.shard,
        "num_shards": args.num_shards,
        "n_records": len(records),
        "records": records,
        "corpus": args.corpus,
        "corpus_dir": args.corpus_dir,
        "ratios": list(ratios),
        "is_base": is_base,
        "batch": batch,
        "window": args.window,
        "pack": args.pack,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, str(out))
    print(f"[lm-eval] wrote {out} ({len(records)} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
