"""Absorbing-mask corruption and the one shared definition of the noise coordinate.

``block_t_from_canvas`` is the single implementation of

    t_b = sum_{i in b} e_i m_i / max(1, sum_{i in b} e_i)

and it is called by training, by evaluation, and by the sampler.  That is
deliberate: when the sampler recomputes the noise coordinate from the live canvas
it must use the *same* function the model was trained with, or the model is
conditioned on a number that disagrees with the canvas it can see.  A block with
no eligible tokens gets ``t_b = 0`` by the ``max(1, .)``, and padding and
immutable prompt positions never enter the denominator.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .refusal import Refusal

#: (low, high, mixture weight) -- the three-bucket noise law.
NOISE_BUCKETS: tuple[tuple[float, float, float], ...] = (
    (0.05, 0.30, 0.30),
    (0.30, 0.70, 0.40),
    (0.70, 1.00, 0.30),
)
BUCKET_NAMES = ("low", "med", "high")

DEFAULT_BLOCK_SIZE = 256


def block_t_from_canvas(eligible: torch.Tensor,
                        is_mask: torch.Tensor,
                        block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    """Realised per-block masked fraction over eligible positions. ``[B, T] -> [B, n_blocks]``.

    Uses the *realised* mask, never the sampled ratio: the span variant quantises
    its length, eligibility removes padding, and the generation lanes overwrite the
    mask wholesale, so a draw would describe a corruption that no longer exists.
    """
    if eligible.shape != is_mask.shape:
        raise Refusal("eligible and is_mask must have the same shape")
    if block_size < 1:
        raise Refusal(f"block_size must be positive, got {block_size}")
    batch, width = eligible.shape
    n_blocks = (width + block_size - 1) // block_size
    pad_to = n_blocks * block_size
    elig = eligible.float()
    masked = is_mask.float() * elig
    if pad_to != width:
        pad = pad_to - width
        elig = F.pad(elig, (0, pad))
        masked = F.pad(masked, (0, pad))
    return masked.view(batch, n_blocks, block_size).sum(-1) / \
        elig.view(batch, n_blocks, block_size).sum(-1).clamp_min(1.0)


def sample_corruption(input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      block_size: int = DEFAULT_BLOCK_SIZE,
                      span_prob: float = 0.5,
                      generator: torch.Generator | None = None,
                      mask_id: int = 65535,
                      pad_id: int = 0,
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-block noise-bucket corruption with a contiguous-span variant.

    Returns ``(corrupted_ids, mask, bucket, block_t)``.  Only eligible positions
    are ever masked -- padding and, in training, no position is immutable, since
    the whole row is the target.

    The noise law and the span variant are the prior lineage's
    (``train_birwkv_diffusion.py:134-271``), minus its generation lanes, which the
    present arms do not use.
    """
    if input_ids.shape != attention_mask.shape:
        raise Refusal("input_ids and attention_mask must have the same shape")
    batch, seq = input_ids.shape
    device = input_ids.device
    n_blocks = (seq + block_size - 1) // block_size

    weights = torch.tensor([b[2] for b in NOISE_BUCKETS], device=device)
    bucket = torch.multinomial(
        weights.expand(batch * n_blocks, -1), 1, generator=generator,
    ).view(batch, n_blocks)
    low = torch.tensor([b[0] for b in NOISE_BUCKETS], device=device)[bucket]
    high = torch.tensor([b[1] for b in NOISE_BUCKETS], device=device)[bucket]
    ratio = low + (high - low) * torch.rand(
        batch, n_blocks, device=device, generator=generator)

    eligible = attention_mask.bool() & input_ids.ne(pad_id)
    mask = torch.zeros_like(eligible)
    use_span = torch.rand(batch, n_blocks, device=device, generator=generator) < span_prob
    rand = torch.rand(batch, seq, device=device, generator=generator)

    for blk in range(n_blocks):
        start, end = blk * block_size, min((blk + 1) * block_size, seq)
        width = end - start
        if width <= 0:
            continue
        rate = ratio[:, blk].unsqueeze(1)
        bmask = rand[:, start:end] < rate
        span_len = (rate.squeeze(1) * width).long().clamp(1, width)
        offset = (
            torch.rand(batch, device=device, generator=generator)
            * (width - span_len + 1).float()
        ).long()
        positions = torch.arange(width, device=device).unsqueeze(0)
        span = (positions >= offset.unsqueeze(1)) & \
               (positions < (offset + span_len).unsqueeze(1))
        bmask = torch.where(use_span[:, blk].unsqueeze(1), span, bmask)
        mask[:, start:end] = bmask & eligible[:, start:end]

    corrupted = torch.where(mask, torch.full_like(input_ids, mask_id), input_ids)
    block_t = block_t_from_canvas(eligible, mask, block_size)
    return corrupted, mask, bucket, block_t
