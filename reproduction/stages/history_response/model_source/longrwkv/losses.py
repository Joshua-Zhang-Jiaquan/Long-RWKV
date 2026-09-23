"""Objectives: selected-token CE, the causal auxiliary, and the rollout auxiliary.

The normalization is the part that has to be right rather than merely plausible.
The selected-token loss is aggregated over **all** valid selected tokens in the
microbatch, so a rank that selected zero tokens must still contribute the global
ratio rather than a zero.  ``global_ratio_loss`` is written so that DDP's own
mean-over-ranks lands exactly on ``global_sum / global_cnt``:

    rank r's loss = local_sum_r * world / global_cnt
    DDP gradient  = (1/world) * sum_r d(loss_r)
                  = sum_r d(local_sum_r) / global_cnt
                  = d(global_sum / global_cnt)

The residual CE at every selected position is ``-log p(y_i | canvas)``; the head is
applied to gathered positions only (see the model's ``gather_idx``), which is why
these functions take ``[Q, V]`` logits rather than a full ``[T, V]`` array.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .corruption import BUCKET_NAMES
from .refusal import Refusal

try:  # torch.distributed is absent in some CPU-only test environments
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None


def global_ratio_loss(local_sum: torch.Tensor,
                      local_cnt: torch.Tensor,
                      world_size: int,
                      global_sum: torch.Tensor | None = None,
                      global_cnt: torch.Tensor | None = None) -> torch.Tensor:
    """This rank's contribution to a globally token-normalized mean.

    ``global_sum``/``global_cnt`` may be supplied directly so a CPU test can
    reproduce a multi-rank split without a process group.
    """
    if world_size < 1:
        raise Refusal(f"world_size must be positive, got {world_size}")
    if global_sum is None or global_cnt is None:
        raise Refusal("global_sum and global_cnt are required")
    return local_sum * (world_size / global_cnt.clamp_min(1.0))


def _reduce_pair(local_sum: torch.Tensor,
                 local_cnt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """All-reduce ``(sum, count)`` across the default process group if one exists."""
    if dist is not None and dist.is_available() and dist.is_initialized():
        stats = torch.stack([local_sum.detach(), local_cnt.detach().to(local_sum.dtype)])
        dist.all_reduce(stats)
        return stats[0], stats[1], dist.get_world_size()
    return local_sum.detach(), local_cnt.detach(), 1


def selected_token_ce(logits: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
    """Per-position CE in fp32 over gathered positions. ``[Q, V]``, ``[Q]`` -> ``[Q]``."""
    if logits.dim() != 2:
        raise Refusal(f"expected gathered logits [Q, V], got {tuple(logits.shape)}")
    if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
        raise Refusal(
            f"targets {tuple(targets.shape)} do not match logits {tuple(logits.shape)}")
    return F.cross_entropy(logits.float(), targets.long(), reduction="none")


def selected_token_ce_global(logits: torch.Tensor,
                             targets: torch.Tensor,
                             ) -> tuple[torch.Tensor, dict[str, float]]:
    """Globally normalized selected-token CE plus its diagnostics.

    An empty selection is legal and yields a zero loss with ``n_selected = 0``; it
    does not perturb the global count.
    """
    ce = selected_token_ce(logits, targets)
    local_sum = ce.sum()
    local_cnt = torch.tensor(float(ce.numel()), dtype=local_sum.dtype,
                             device=local_sum.device)
    global_sum, global_cnt, world = _reduce_pair(local_sum, local_cnt)
    loss = global_ratio_loss(local_sum, local_cnt, world, global_sum, global_cnt)
    mean_ce = float((global_sum / global_cnt.clamp_min(1.0)).item())
    return loss, {
        "mask_ce": mean_ce,
        "n_selected": int(ce.numel()),
        "global_n_selected": int(global_cnt.item()),
        # ``detach`` before ``item``: the diagnostic is a number for the record,
        # not a node in the graph, and reading a requires_grad tensor as a scalar
        # earns a UserWarning on every microbatch that would bury real ones.
        "local_sum": float(local_sum.detach().item()),
    }


def bucket_diagnostics(ce: torch.Tensor,
                       bucket_of_position: torch.Tensor,
                       block_size: int,
                       n_blocks: int) -> dict[str, float]:
    """Per-bucket mean CE. ``ce`` is per-position and already zero outside the selection."""
    diag: dict[str, float] = {}
    seq = bucket_of_position.shape[-1]
    for index, name in enumerate(BUCKET_NAMES):
        selected = torch.zeros_like(ce, dtype=torch.bool)
        for blk in range(n_blocks):
            start, end = blk * block_size, min((blk + 1) * block_size, seq)
            if start >= seq:
                break
            selected[:, start:end] |= (bucket_of_position[:, blk] == index).unsqueeze(1)
        picked = ce.masked_select(selected)
        diag[f"ce_{name}"] = float(picked.mean().item()) if picked.numel() else float("nan")
        diag[f"n_{name}"] = int(picked.numel())
    return diag


def causal_next_token_valid(doc_starts: torch.Tensor,
                            attention_mask: torch.Tensor) -> torch.Tensor:
    """Validity mask for next-token pairs, cut at document boundaries.

    Position ``i`` may predict ``i+1`` only when both are live **and** ``i+1`` does
    not begin a new document.  Without the second condition the auxiliary teaches
    the model to predict the first token of an unrelated document, which is
    exactly the leak the packed format makes easy to introduce.
    """
    if doc_starts.shape[0] != attention_mask.shape[0]:
        raise Refusal("doc_starts and attention_mask must share a batch dimension")
    live = attention_mask.bool()
    valid = live[:, :-1] & live[:, 1:]
    batch, _ = valid.shape
    for b in range(batch):
        for start in doc_starts[b].tolist():
            start = int(start)
            if 0 < start <= valid.shape[1]:
                valid[b, start - 1] = False
    return valid


def next_token_ce_global(logits: torch.Tensor,
                         targets: torch.Tensor,
                         ) -> tuple[torch.Tensor, dict[str, float]]:
    """Globally normalized next-token CE: the autoregressive arm's whole loss.

    Numerically this is :func:`selected_token_ce_global` over a different position
    set, and that is the point of giving it its own name and its own diagnostic
    key.  A0 is the paper's autoregressive reference; while its number was
    reported as ``mask_ce`` there was no field anywhere -- log line, run record,
    or results table -- in which an autoregressive run looked different from a
    masked one, and A0 trained the masked loss for three seeds without any
    artifact contradicting its label.  ``ar_ce`` is that field.

    The normalization is the same global ratio: a rank whose microbatch happened
    to contain no valid next-token pair still contributes the global mean rather
    than a zero.  See :func:`global_ratio_loss` for why DDP's mean-over-ranks
    lands exactly on ``global_sum / global_cnt``.
    """
    ce = selected_token_ce(logits, targets)
    local_sum = ce.sum()
    local_cnt = torch.tensor(float(ce.numel()), dtype=local_sum.dtype,
                             device=local_sum.device)
    global_sum, global_cnt, world = _reduce_pair(local_sum, local_cnt)
    loss = global_ratio_loss(local_sum, local_cnt, world, global_sum, global_cnt)
    mean_ce = float((global_sum / global_cnt.clamp_min(1.0)).item())
    return loss, {
        "ar_ce": mean_ce,
        "n_pairs": int(ce.numel()),
        "global_n_pairs": int(global_cnt.item()),
        "local_sum": float(local_sum.detach().item()),
    }


def rollout_aux_weights(n_calls: int, weights: tuple[float, ...] | None = None
                        ) -> torch.Tensor:
    """Per-call weights for the short-rollout auxiliary, averaged exactly.

    Default is uniform over the declared calls.  The weights are recorded in the
    run record: they are a declared training choice, and an unrecorded weight
    makes the auxiliary unauditable.
    """
    if n_calls < 1:
        raise Refusal(f"n_calls must be positive, got {n_calls}")
    if weights is None:
        weights = tuple(1.0 for _ in range(n_calls))
    if len(weights) != n_calls:
        raise Refusal(f"{len(weights)} weights for {n_calls} calls")
    total = float(sum(weights))
    if total <= 0:
        raise Refusal(f"rollout weights must sum to a positive value, got {total}")
    return torch.tensor([w / total for w in weights], dtype=torch.float32)
