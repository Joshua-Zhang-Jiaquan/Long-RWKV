"""Optional capacity mechanisms for the non-latent v7 architecture round.

Two opt-in arms, both wrapping the LAYER LOOP of
:class:`~models.birwkv7_diffusion.BiRWKV7ForMaskedDiffusion` (never the block
internals), both EXACT identity at init so a warm-started checkpoint is
numerically unchanged at step 0:

- :class:`ResidualStreamMixer` — n-stream hyper-connections (mHC arm, W-A2).
  The residual state becomes ``H ∈ [B, T, n, C]``; each layer the block
  processes ONE learned combination of the streams (1× block compute — the
  whole point of hyper-connections vs deeper/wider stacks), and the streams
  recombine around the block's update::

      x      = Σ_j w_pre[l,j] · H_l[j]                (input combination, [B,T,C])
      o      = block(x)                               (single call, 1x compute)
      delta  = o − x                                  (the block's own update)
      H_{l+1}[i] = Σ_j A_res[l,i,j] · H_l[j] + a_post[l,i] · delta

  Init: ``w_pre = e_0`` (one-hot), ``A_res = I``, ``a_post = 1`` — all via the
  bounded-deviation parameterization ``param = ref + mix_scale·tanh(raw)`` with
  ``raw ← 0`` — so the update collapses exactly to the legacy residual add and
  the mean readout is exact. Streams differentiate once trained (``w_pre``
  re-weights the combination, ``A_res`` rows diverge) — the ensemble of residual
  trajectories at ~zero extra compute is the capacity gain.

  DESIGN NOTE (the degeneracy this replaces): an earlier draft mixed the
  streams with ROW-STOCHASTIC matrices and fed the block the folded
  ``[B·n, T, C]`` batch. Both were wrong: a row-sum-1 matrix acting on
  replicated identical streams is the identity map FOREVER (streams never
  diverge — caught by the response control: a 1.2e-06 "change" that was pure
  fp noise), and folding multiplies block FLOPs by n. The combination form has
  neither defect.

- :class:`BackboneLoopControl` — weight-tied loop over a contiguous layer range
  (loop arm, W-A3; Ouro-style iterated depth, arXiv 2510.25741). Extra passes
  refine the residual state through a tanh-bounded per-pass scalar gate
  ``h ← h + g_p·(F(h) − h)`` with ``g_p = loop_scale·tanh(gates_raw)``,
  ``gates_raw ← 0`` ⇒ ``g_p = 0`` exactly.

Pure torch — no ``fla`` import — so the CPU regression suites import this
module directly on the login node (house rule, cf. ``latent_plan.py``).

FSDP placement: both modules are attached to the model root and consumed once
per forward (per-layer slices inside the loop), the ``latent_cond`` /
``block_t_cond`` precedent — keeping them OUT of
``transformer_layer_cls={BiRWKV7Block}`` avoids a per-block gather.

Dtype rule (the B3 MixedPrecision crash): every factor is computed in fp32 and
cast to the residual state's RUNTIME dtype (``h.dtype``) before use — never
hardcode; the trunk is bf16 under FSDP MixedPrecision.
"""

from __future__ import annotations

import torch
from torch import nn

from models.state_hijacking_dit_torch_types import TypedTorchModule


# ----------------------------------------------------------------------
# Stream helpers (module-level, pure torch)
# ----------------------------------------------------------------------
def expand_to_streams(h: torch.Tensor, n: int) -> torch.Tensor:
    """``h [B,T,C] -> H [B,T,n,C]`` with all n streams = h (warm-start replication)."""
    return h.unsqueeze(2).expand(-1, -1, n, -1).contiguous()


def readout(H: torch.Tensor) -> torch.Tensor:
    """``H [B,T,n,C] -> h [B,T,C]`` by mean over streams.

    Exact for identical streams: a mean of n identical fp values is that value
    bit-exactly for n ∈ {2, 4} (multiply/divide by powers of two are exact).
    """
    return H.mean(dim=2)


# ----------------------------------------------------------------------
# Mechanism 1: n-stream hyper-connections (mHC arm)
# ----------------------------------------------------------------------
class ResidualStreamMixer(TypedTorchModule):
    """Per-layer n-stream hyper-connection factors (mHC arm).

    Bounded-deviation parameterization (the connection manifold stand-in: every
    factor stays within ``mix_scale`` of its identity reference)::

        w_pre[l]  = e_0       + mix_scale · tanh(w_pre_raw[l])    [n]
        A_res[l]  = I_n       + mix_scale · tanh(m_res_raw[l])    [n, n]
        a_post[l] = 1         + mix_scale · tanh(p_post_raw[l])   [n]

    with every raw tensor zero-init, so at init the layer computes exactly::

        x = H[0];  o = block(x);  H'[i] = H[i] + (o − x)          (== legacy)

    Gradients flow to ALL THREE raw groups at init (none is bilinear-zero):
    ``∂x/∂w_pre = H[0] ≠ 0``, ``∂H'/∂A_res = H ≠ 0``, and
    ``∂H'/∂a_post = delta = block(x) − x ≠ 0`` (the block's own update is
    nonzero even at init). Raw tensors are 2-D/3-D (never 0-dim) so FSDP
    FULL_SHARD can shard them — the ``gate_logit`` ``[1]``-shape precedent.
    """

    def __init__(self, num_layers: int, n_streams: int, mix_scale: float = 1.0) -> None:
        """Zero-init the raw factors for ``num_layers`` layers of ``n_streams``."""
        super().__init__()
        if n_streams not in (2, 4):
            msg = f"n_streams must be in {{2,4}}, got {n_streams}"
            raise ValueError(msg)
        self.num_layers = int(num_layers)
        self.n_streams = int(n_streams)
        self.mix_scale = float(mix_scale)
        # Zero-init raw parts -> every factor equals its identity reference exactly.
        self.w_pre_raw = nn.Parameter(torch.zeros(self.num_layers, n_streams))
        self.m_res_raw = nn.Parameter(torch.zeros(self.num_layers, n_streams, n_streams))
        self.p_post_raw = nn.Parameter(torch.zeros(self.num_layers, n_streams))

    def factors(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(w_pre, A_res, a_post)`` for layer ``l`` — fp32, shapes ``[n]``, ``[n,n]``, ``[n]``.

        Computed in fp32; the CALLER casts to the residual state's runtime dtype
        (dtype-trap rule: derive from ``h.dtype``, never hardcode).
        """
        dev = self.w_pre_raw.device
        eye = torch.eye(self.n_streams, device=dev, dtype=torch.float32)
        e0 = torch.zeros(self.n_streams, device=dev, dtype=torch.float32)
        e0[0] = 1.0
        w_pre = e0 + self.mix_scale * torch.tanh(self.w_pre_raw[layer_idx].to(torch.float32))
        a_res = eye + self.mix_scale * torch.tanh(self.m_res_raw[layer_idx].to(torch.float32))
        a_post = 1.0 + self.mix_scale * torch.tanh(self.p_post_raw[layer_idx].to(torch.float32))
        return w_pre, a_res, a_post

    def max_deviation(self) -> float:
        """Max |A_res − I| over all layers — bounded by ``mix_scale`` (G-M gate)."""
        with torch.no_grad():
            eye = torch.eye(self.n_streams, device=self.m_res_raw.device, dtype=torch.float32)
            a = eye + self.mix_scale * torch.tanh(self.m_res_raw.to(torch.float32))
            return float((a - eye).abs().max())


# ----------------------------------------------------------------------
# Mechanism 2: weight-tied backbone loop (loop arm)
# ----------------------------------------------------------------------
class BackboneLoopControl(TypedTorchModule):
    """Weight-tied sub-range loop gate (Ouro-inspired backbone loop arm).

    Identity-init equation::

        g_p = loop_scale · tanh(gates_raw[p]),   gates_raw ← 0  ⇒  g_p = 0 exactly
        h   ← h + g_p · (F_{[lo,hi)}(h) − h)

    ``g_p`` is a PER-PASS scalar (one gate per extra rep, shared across the
    looped layers), tanh-bounded to ``[−loop_scale, loop_scale]`` — the house
    bounded-residual discipline. Gradients flow immediately:
    ``∂h/∂gates_raw[p] = loop_scale·(1−tanh²(0))·(F(h)−h) ≠ 0`` — no bilinear
    zero, no one-step delay.

    ``lo``/``hi`` are PERSISTENT buffers so they round-trip in ``model.pt``;
    all keys land under the ``loop.`` prefix (param + buffers) so
    ``_merge_latent_keys`` whitelists the whole arm with one prefix. The
    trained rep count is recoverable from ``gates_raw.shape[0]``.
    """

    def __init__(self, lo: int, hi: int, reps: int, loop_scale: float = 1.0) -> None:
        """Gate ``reps`` extra weight-tied passes of layers ``[lo, hi)``."""
        super().__init__()
        if reps not in (1, 2):
            msg = f"loop_reps must be in {{1,2}}, got {reps}"
            raise ValueError(msg)
        if not (0 <= int(lo) < int(hi)):
            msg = f"loop range [{lo},{hi}) is empty or negative"
            raise ValueError(msg)
        self.loop_scale = float(loop_scale)
        self.register_buffer("lo", torch.tensor(int(lo), dtype=torch.int64))
        self.register_buffer("hi", torch.tensor(int(hi), dtype=torch.int64))
        self.gates_raw = nn.Parameter(torch.zeros(int(reps)))

    def gates(self) -> torch.Tensor:
        """Per-pass gate ``[r]``, tanh-bounded, exactly 0 at init."""
        return self.loop_scale * torch.tanh(self.gates_raw)

    @property
    def reps(self) -> int:
        """Trained extra-pass count (from the gate vector length)."""
        return int(self.gates_raw.numel())
