"""Measure the trained decay distribution that the long-context bound depends on.

The influence bound (``theory_bounds.py``) is a function of ``exp(w)``, where
``w = -DECAY_SCALE * sigmoid(w_lora(x_w))`` is **input-dependent**.  The
checkpoint stores only the LoRA weights, so the distribution of ``w`` cannot be
read off disk: it has to be captured from a forward pass.  This module captures
it and reduces it to the two numbers the theorem needs.

Why the bias alone is not enough
--------------------------------
Evaluating the LoRA at zero input gives the *bias* of the decay, i.e. the centre
of the distribution.  A first pass over the current checkpoint's biases gives a
slowest channel of ``exp(w) = 0.99986`` -- a single-token half-life of roughly
2,000-5,000 tokens per layer.  That number is suggestive and must not be quoted
as the horizon: the real ``w`` varies per token, and the horizon follows the
distribution, not a single slice of it.  The bound of Lemma 1 (contraction for
*every* input) is what is proved; the horizon is a measurement, and this module
is where it is made.

The two figures
---------------
1. **Per layer and direction**, the slowest decay observed:
   ``kappa_L = max_t max_d exp(w_{t,d})``, and its half-life
   ``ln 2 / ln(1/kappa_L)``.
2. **Per FORWARD, for the recycled range.**  The weight-tied loop re-applies
   layers ``[lo, hi)`` once per forward, so those layers' states are updated
   *twice* per forward.  A token's influence therefore decays by the product of
   the two passes at that layer: ``kappa_forward = max_pass1 * max_pass2``, and
   the half-life halves in relative terms.  The loop makes the recycled layers
   forget *faster per forward*, which is a direct consequence of the mechanism
   and the opposite of what "more depth" might suggest -- worth stating.

Requires CUDA: the model cannot be constructed without the fla kernels
(``models.birwkv7_diffusion`` raises "0 active drivers" on a CPU-only host).
The reduction is deliberately separated from the capture so that it is fully
testable on CPU with synthetic captures.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from scale.experiments.nonlatent_iclr import theory_bounds as tb

#: Layers the weight-tied loop re-applies, from the training spec
#: (``--loop-range 16:32``) and confirmed in ``architecture_contract.json``.
LOOP_RANGE: tuple[int, int] = (16, 32)

SCHEMA = "nonlatent_decay_survey_v1"

#: The module whose output is the pre-sigmoid decay logits.  The fla layer
#: computes ``w = -DECAY_SCALE * sigmoid(w_lora(x_w))``, so a forward hook on
#: ``w_lora`` captures exactly the argument the bound is a function of.
HOOK_SUFFIX = "w_lora"


class DecaySurveyError(RuntimeError):
    """The survey could not produce the numbers the bound needs."""


def hook_target_names(model: Any, *, suffix: str = HOOK_SUFFIX) -> list[str]:
    """Names of the decay modules, in layer order.

    Named by traversal rather than assumed, so a model whose layer list is
    nested differently (streams, a wrapped backbone) reports what it actually
    has instead of what the caller expected.  Only ``layers.N.attn_{fwd,bwd}``
    modules are kept: the survey is about the recurrent decay, and picking up an
    unrelated module that happens to end in ``w_lora`` would silently mix
    quantities.
    """
    names: list[str] = []
    for name, module in model.named_modules():
        if not name.endswith(suffix):
            continue
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "layers":
            continue
        if not parts[2].startswith("attn_"):
            continue
        names.append(name)

    def layer_index(qualified: str) -> int:
        try:
            return int(qualified.split(".")[1])
        except (IndexError, ValueError) as exc:
            msg = f"{qualified!r} does not carry a numeric layer index"
            raise DecaySurveyError(msg) from exc

    # Sorted on (layer, full name) rather than the layer alone: within a layer
    # the two directions would otherwise come out in module-registration order,
    # which is an accident of how the block was built rather than a property of
    # the model, and a receipt whose row order depends on construction order is
    # hard to diff between two runs.
    return sorted(names, key=lambda qualified: (layer_index(qualified), qualified))


def register_decay_hooks(model: Any, sink: dict[str, list[torch.Tensor]],
                         *, device: str = "cuda") -> list[Any]:
    """Record ``max_d exp(w)`` per token for every decay module.

    The sink maps a module name to a LIST of per-token arrays, one entry per
    firing.  The list is what makes the recycled-depth figure possible: a layer
    inside the loop range fires twice per forward and the two passes have to be
    kept apart, because the second pass sees a different hidden state and hence
    a different ``w``.

    The tensor is moved off the device and reduced immediately -- keeping the
    full ``[B, T, hidden]`` logit tensor for 32 layers x 2 directions would be
    gigabytes for one document, and the reduction is all the bound needs.
    """
    handles = []
    for name in hook_target_names(model):
        module = model.get_submodule(name)

        def _hook(_module, _inputs, output, _name=name, _device=device):
            if not isinstance(output, torch.Tensor):
                return
            logits = output.detach().to(torch.float32)
            decay = torch.exp(-tb.DECAY_SCALE * torch.sigmoid(logits))
            # per-token, per-position slowest channel
            sink.setdefault(_name, []).append(decay.amax(dim=-1).to("cpu"))

        handles.append(module.register_forward_hook(_hook))
    if not handles:
        msg = (
            "no decay modules matched; the checkpoint is not the rwkv7 "
            "bidirectional denoiser this survey is written for"
        )
        raise DecaySurveyError(msg)
    return handles


def summarize(captured: Mapping[str, Iterable[torch.Tensor]], *,
              recycled_range: tuple[int, int] = LOOP_RANGE) -> dict[str, object]:
    """Reduce a capture to per-module kappa, half-lives, and per-forward rates.

    ``captured`` maps a module name to the per-firing per-token slowest-decay
    arrays.  Firing order is preserved, so for a recycled layer the first entry
    is the base pass and the second the loop pass; their product is the
    per-forward contraction at that layer.

    A module with no firings is reported as ``firings: 0`` rather than being
    omitted -- a layer that never fired is a different finding from a layer that
    fired and decayed fast.
    """
    lo, hi = recycled_range
    modules: dict[str, object] = {}
    for name in sorted(captured, key=lambda n: (int(n.split(".")[1]), n)):
        firings = [torch.as_tensor(entry).reshape(-1) for entry in captured[name]]
        if not firings:
            modules[name] = {"firings": 0}
            continue
        per_firing = [float(entry.max()) for entry in firings]
        # The per-forward rate is the product of the passes: a write survives a
        # forward only if it survives every application of that layer's state
        # update within it.
        per_forward = math.prod(per_firing)
        entry: dict[str, object] = {
            "firings": len(firings),
            "tokens_observed": int(sum(int(f.numel()) for f in firings)),
            "kappa_per_application": max(per_firing),
            "kappa_per_forward": per_forward,
            "median_decay": float(torch.cat(firings).median()),
        }
        layer = int(name.split(".")[1])
        entry["in_recycled_range"] = bool(lo <= layer < hi)
        for label, kappa in (("per_application", max(per_firing)),
                             ("per_forward", per_forward)):
            try:
                entry[f"half_life_tokens_{label}"] = tb.half_life(kappa)
            except tb.BoundRefusal:
                # kappa >= 1 certifies nothing; recording the raw rate is the
                # honest report, and omitting the key would hide the layer.
                entry[f"half_life_tokens_{label}"] = None
                entry[f"certifies_contraction_{label}"] = False
            else:
                entry[f"certifies_contraction_{label}"] = True
        modules[name] = entry

    # The slowest single application anywhere in the model is the model-level
    # constant: the bound is only as strong as the layer that forgets slowest.
    rates = [float(entry["kappa_per_application"])  # type: ignore[index]
             for entry in modules.values() if entry.get("firings")]
    forward_rates = [float(entry["kappa_per_forward"])  # type: ignore[index]
                     for entry in modules.values() if entry.get("firings")]
    summary: dict[str, object] = {
        "modules": modules,
        "n_modules": len(modules),
        "slowest_application": max(rates) if rates else None,
        "slowest_per_forward": max(forward_rates) if forward_rates else None,
        "recycled_range": [lo, hi],
        "note": (
            "kappa is a MEASUREMENT of the trained weights on the documents "
            "surveyed, not a proved constant. Lemma 1 proves contraction for "
            "every input; how close the rate sits to 1 is what the horizon "
            "depends on, and that is this number."
        ),
    }
    if rates:
        slowest = max(rates)
        try:
            summary["model_half_life_tokens"] = tb.half_life(slowest)
        except tb.BoundRefusal as exc:
            summary["model_half_life_tokens"] = None
            summary["model_half_life_refusal"] = str(exc)
    return summary


def write_survey(document: dict[str, object], path: Path) -> Path:
    """Write the survey receipt; refuses to overwrite an existing one."""
    path = Path(path)
    if path.exists():
        msg = f"{path} already exists; a survey receipt is written once"
        raise DecaySurveyError(msg)
    document = {"schema": SCHEMA, **document}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _documents(token_dir: str, count: int, length: int, seed: int) -> torch.Tensor:
    """A fixed probe batch: real packed ids, chosen deterministically.

    Real tokens rather than random ids, because ``w`` depends on the input and
    random ids are off-distribution in a way that would bias the decay toward
    whatever the embedding does with noise.
    """
    import pickle  # noqa: PLC0415

    directory = Path(token_dir)
    shards = sorted(directory.glob("*.pkl"))
    if not shards:
        msg = f"no packed *.pkl shards under {token_dir}"
        raise DecaySurveyError(msg)
    generator = torch.Generator().manual_seed(seed)
    rows: list[torch.Tensor] = []
    for shard in shards:
        with shard.open("rb") as handle:
            payload = pickle.load(handle)
        for row in payload:
            ids = row["input_ids"] if isinstance(row, dict) else row
            tensor = torch.as_tensor(list(ids), dtype=torch.long)
            if tensor.numel() < length:
                continue
            rows.append(tensor[:length])
            if len(rows) >= count:
                return torch.stack(rows)
        del generator  # the loop is deterministic; no sampling is used
    if not rows:
        msg = f"{token_dir} yielded no document of at least {length} tokens"
        raise DecaySurveyError(msg)
    return torch.stack(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True,
                        help="checkpoint directory holding model.pt")
    parser.add_argument("--model-dir", required=True,
                        help="HF directory supplying geometry and tokenizer")
    parser.add_argument("--token-dir", required=True,
                        help="packed-token shard directory for the probe batch")
    parser.add_argument("--out", required=True, help="survey receipt to write")
    parser.add_argument("--documents", type=int, default=64)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--loop-range", type=int, nargs=2, default=list(LOOP_RANGE))
    parser.add_argument("--loop-reps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import sys  # noqa: PLC0415
    code_root = Path(__file__).resolve().parents[3] / "DAN" / "v7_arch_round" / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion  # noqa: PLC0415

    loop_range = (int(args.loop_range[0]), int(args.loop_range[1]))
    model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(
        args.model_dir, dtype=torch.bfloat16,
        loop_range=loop_range, loop_reps=int(args.loop_reps))
    state = torch.load(Path(args.ckpt_dir) / "model.pt", map_location="cpu",
                       weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # A survey run on the wrong checkpoint would report a decay distribution
    # that belongs to no model in the paper, so the mismatch is reported rather
    # than tolerated silently.
    serious = [key for key in unexpected if "fuse_" not in key]
    if serious:
        msg = f"checkpoint carries {len(serious)} unexpected tensors, e.g. {serious[:3]}"
        raise DecaySurveyError(msg)
    model = model.to(args.device).eval()

    captured: dict[str, list[torch.Tensor]] = {}
    handles = register_decay_hooks(model, captured, device=args.device)
    batch = _documents(args.token_dir, int(args.documents), int(args.length),
                       int(args.seed))
    try:
        with torch.inference_mode():
            for row in batch:
                _ = model(row.unsqueeze(0).to(args.device), force_forward=False)
    finally:
        for handle in handles:
            handle.remove()

    document = {
        "checkpoint": str(args.ckpt_dir),
        "model_dir": str(args.model_dir),
        "token_dir": str(args.token_dir),
        "documents": int(args.documents),
        "length": int(args.length),
        "seed": int(args.seed),
        "loop_reps": int(args.loop_reps),
        "missing_keys": len(missing),
        "summary": summarize(captured, recycled_range=loop_range),
    }
    path = write_survey(document, Path(args.out))
    print(json.dumps({"decay_survey": str(path),
                      "slowest_application": document["summary"]["slowest_application"],  # type: ignore[index]
                      "model_half_life_tokens": document["summary"].get("model_half_life_tokens")}))  # type: ignore[union-attr]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
