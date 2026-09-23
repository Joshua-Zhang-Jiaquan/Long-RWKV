"""Noise conditioning: the position features, the gate modulator, and the input term.

The paper's conditioning enters through the mixer's **native gate preactivations**:

    w = chi_w(g_w(h) + b^retention),    a = chi_a(g_a(h) + b^write)

so the modulator returns a pair of *preactivation* biases and the mixer adds them
before its own nonlinearity.  ``chi_w``/``chi_a``, their domains, and all native
normalization stay where they are -- the appendix requires that explicitly.

Parity at initialization is exact, not approximate: every output projection is
zero-initialized in **both weight and bias**, so the biases are the zero tensor
and a modulator-equipped model is bit-for-bit its unconditioned self at step 0.
That property is what lets A3-vs-A2 isolate gate-level modulation rather than
confound it with a changed model at initialization, so it is asserted in the
qualification suite rather than assumed.

A2 is this same model with ``gate_mod=None``: input-level conditioning is present
in A1, A2 and A3, so A3-A2 isolates the gate path alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .refusal import Refusal

# --- the four-way per-position visibility code ----------------------------
VIS_PAD = 0
VIS_IMMUTABLE = 1
VIS_FILLED = 2
VIS_MASKED = 3
NUM_VIS = 4

NUM_FREQS = 8
PHI_DIM = 1 + 2 * NUM_FREQS + NUM_VIS  # 21


def vis_codes(input_ids: torch.Tensor,
              attention_mask: torch.Tensor,
              target_mask: torch.Tensor,
              mask_id: int) -> torch.Tensor:
    """The per-position visibility code, in the paper's four-way convention.

    Order matters: a padding position is padding even if some caller also left it
    in ``target_mask``, so the tests short-circuit pad first.
    """
    live = attention_mask.bool()
    target = target_mask.bool() & live
    is_mask = input_ids.eq(mask_id)
    codes = torch.full_like(input_ids, VIS_PAD)
    codes = torch.where(live, torch.full_like(codes, VIS_IMMUTABLE), codes)
    filled = target & ~is_mask
    codes = torch.where(filled, torch.full_like(codes, VIS_FILLED), codes)
    masked = target & is_mask
    codes = torch.where(masked, torch.full_like(codes, VIS_MASKED), codes)
    return codes


def phi_features(block_t: torch.Tensor, codes: torch.Tensor,
                 block_size: int | None = None) -> torch.Tensor:
    """Per-position conditioning features: ``[t_b, sin/cos(2^j pi t_b), onehot(v_i)]``.

    ``block_t`` is ``[B, n_blocks]``; the value is broadcast to every position of
    its block, so the same block's positions all see the same noise coordinate.

    Fourier features rather than the raw scalar, because mask fraction interacts
    non-monotonically with difficulty -- a single linear map of ``t`` can only
    express a monotone ramp.

    **``block_size`` must travel with ``block_t``; it cannot be recovered from the
    shapes.**  ``block_t_from_canvas`` tiles with ``n_blocks = ceil(T / block_size)``
    and zero-pads the ragged tail, so the last block is short whenever
    ``block_size`` does not divide ``T``.  This function used to infer
    ``T // n_blocks`` and refuse when the product missed ``T`` -- which is not a
    check on the caller but a second, incompatible definition of the grid:

      * It refused **79.1%** of widths in 1..4199, and the refusal is silent at the
        benchmark level because ``longbench.score_task_rows`` catches ``Refusal``
        per row and moves on.  Training never saw it because every pack row is
        4096 = 16 x 256, and the qualification suite passes 16/32/64/256 into
        *both* halves, so producer and consumer always agreed there.
      * Measured over those widths against a producer at 256, the old consumer
        **refused 3320**, accepted **16** at the right grid, and accepted **863**
        at a *different* nominal grid -- of which **608** are positionally wrong,
        giving some position another block's noise value.  (The other 255 are
        single-block rows, T <= 256, where the nominal size cannot matter.)  The
        smallest genuinely corrupted width is 258: two blocks, division yields
        129, and position 129 reads block 1's value where the producer put it in
        block 0.
      * Where the division refuses it is also wrong about the grid: at T=3104 with
        13 blocks it yields 238, and ``238 x 13 = 3094 != 3104``.
      * The size is genuinely ambiguous from shapes alone: 13 blocks over 3104
        positions is consistent with any ``block_size`` in 239..258, and the
        producer's own answer (256) is not the one the division returns.

    So the grid is passed, and a mismatch is a Refusal about a *caller* that
    disagrees with the producer rather than about a width that is not a multiple.
    ``None`` keeps the derivation, for the callers whose width is a clean multiple
    and which predate the parameter; it is exactly the old behaviour and is still
    checked.
    """
    if block_t.dim() != 2 or codes.dim() != 2:
        raise Refusal("block_t and codes must both be [B, T_blocks] / [B, T]")
    batch, width = codes.shape
    n_blocks = block_t.shape[1]
    if not n_blocks:
        raise Refusal("block_t has no blocks, so no position has a noise value")
    if block_size is None:
        # The legacy path: derive, and keep the old refusal, which is correct
        # whenever the width IS a multiple -- the only case it was ever right for.
        block_size = width // n_blocks
        if block_size * n_blocks != width:
            raise Refusal(
                f"{n_blocks} blocks do not tile {width} positions evenly, and no "
                f"block_size was passed; the grid cannot be recovered from the "
                f"shapes (n_blocks x block_size >= width admits a range), so the "
                f"caller must pass the block_size it built block_t with")
    else:
        if block_size < 1:
            raise Refusal(f"block_size must be positive, got {block_size}")
        # The producer's own relation, asserted rather than assumed: a caller that
        # built block_t at another grid would otherwise silently shift every
        # position's noise value by up to a block.
        if (width + block_size - 1) // block_size != n_blocks:
            raise Refusal(
                f"block_t has {n_blocks} blocks but block_size={block_size} over "
                f"{width} positions tiles into "
                f"{(width + block_size - 1) // block_size}; block_t was built at a "
                f"different grid than this call claims")
    # ``repeat_interleave`` then truncate: the producer zero-pads its ragged tail to
    # n_blocks * block_size, so the broadcast is that wide and the final positions
    # beyond ``width`` belong to no token.  Slicing is the inverse of the producer's
    # padding, which is why it is a slice and not a second grid.
    t = block_t.to(torch.float32).repeat_interleave(block_size, dim=1)
    if t.shape[1] < width:
        raise Refusal(f"broadcast block_t to {t.shape[1]} positions, expected {width}")
    t = t[:, :width].unsqueeze(-1)
    freqs = 2.0 ** torch.arange(NUM_FREQS, dtype=torch.float32, device=t.device)
    ang = t * freqs.view(1, 1, -1) * math.pi
    onehot = torch.nn.functional.one_hot(codes.long(), NUM_VIS).to(torch.float32)
    return torch.cat((t, torch.sin(ang), torch.cos(ang), onehot), dim=-1)


class GateModulator(nn.Module):
    """Produces the per-position retention and write preactivation biases.

    ``U^q(tanh(V^q phi))`` for each gate.  Both ``U`` projections are zero-init in
    weight *and* bias, so the module contributes exactly zero at initialization.
    """

    def __init__(self, key_dim: int, rank: int = 32, phi_dim: int = PHI_DIM) -> None:
        super().__init__()
        if rank < 1:
            raise Refusal(f"modulator rank must be positive, got {rank}")
        self.key_dim = key_dim
        self.rank = rank
        self.phi_dim = phi_dim
        self.v_proj = nn.Linear(phi_dim, rank, bias=True)
        self.u_retention = nn.Linear(rank, key_dim, bias=True)
        self.u_write = nn.Linear(rank, key_dim, bias=True)
        for projection in (self.u_retention, self.u_write):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``phi [B, T, phi_dim] -> (b_retention [B, T, key_dim], b_write [B, T, key_dim])``."""
        if phi.shape[-1] != self.phi_dim:
            raise Refusal(
                f"expected phi of width {self.phi_dim}, got {phi.shape[-1]}")
        # Derive the parameter dtype rather than assuming one.  ``phi_features``
        # returns fp32 for precision while the module is bf16 under autocast, so a
        # raw matmul is a hard "mat1 and mat2 must have the same dtype" error on the
        # GPU -- nine qualification checks died on exactly this, because the input
        # conditioner converts and this path did not.
        dtype = self.v_proj.weight.dtype
        hidden = torch.tanh(self.v_proj(phi.to(dtype)))
        return self.u_retention(hidden), self.u_write(hidden)


class InputNoiseConditioner(nn.Module):
    """Additive input-level noise conditioning on the token embeddings.

    Separate from ``GateModulator`` on purpose: this is the path A1 and A2 also
    have, so it cannot be what A3-A2 measures.  Bounded by ``out_scale * tanh``,
    because an unbounded conditioning arm in this lineage diverged early.
    """

    def __init__(self, hidden_size: int, phi_dim: int = PHI_DIM,
                 out_scale: float = 0.5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.phi_dim = phi_dim
        self.out_scale = out_scale
        self.trunk = nn.Sequential(nn.Linear(phi_dim, hidden_size), nn.GELU())
        self.out = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.shape[-1] != self.phi_dim:
            raise Refusal(
                f"expected phi of width {self.phi_dim}, got {phi.shape[-1]}")
        # Derive the parameter dtype rather than hardcoding one: under bf16
        # autocast the weights are bf16 while phi arrives fp32, and a dtype
        # mismatch there is a hard error inside the first step, after the queue
        # wait, rather than a silent upcast.
        dtype = self.trunk[0].weight.dtype
        return self.out_scale * torch.tanh(self.out(self.trunk(phi.to(dtype))))
