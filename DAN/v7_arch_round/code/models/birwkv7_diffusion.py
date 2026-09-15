"""BiRWKV-7 token-level masked-diffusion denoiser.

The BiRWKV IS the denoiser: corrupted tokens go in, clean-token logits come out
of this model's own lm_head. There is no frozen renderer and no post-diffusion
causal generator (code/plan section 1.1 / 4.6 contract).

Design
------
Each layer holds TWO complete fla ``RWKV7Attention`` modules (forward + reverse)
plus the pretrained ``RWKV7FeedForward`` and norms. The reverse module runs on
the flipped sequence. Outputs fuse through a learned sigmoid gate initialised
strongly toward the forward direction, so at init (or with ``force_forward``)
the network is numerically ~identical to the pretrained causal RWKV-7 — which
is what makes the HF warm-start parity test possible.

Warm-start: both attention copies, the FFN, all norms, the embedding and the
lm_head load from the HF ``RWKV7ForCausalLM`` safetensors (fla-format keys,
e.g. ``model.layers.{i}.attn.r_proj.weight``). The reverse copy is a clone of
the forward weights (code/plan section 4.2 recipe).

Requires the ``fla`` package with CUDA (relay2:v2 image). Cannot import on a
CPU-only box: fla's triton kernels need an active GPU driver.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import torch
from fla.layers.rwkv7 import RWKV7Attention
from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.rwkv7.modeling_rwkv7 import RWKV7FeedForward
from torch import nn
from torch.utils import checkpoint as _torch_checkpoint
from typing_extensions import override

from models.residual_streams import (
    BackboneLoopControl,
    ResidualStreamMixer,
    expand_to_streams,
    readout,
)
from models.state_hijacking_dit_torch_types import TypedTorchModule

MASK_TOKEN_ID = 65535  # unused slot in the RWKV World vocab (EOS=65530, max data id 65530)
PAD_TOKEN_ID = 0
_MIN_KENDALL_POINTS = 2
_Z_SLOTTED_NDIM = 3

_F = TypeVar("_F", bound=Callable[..., object])


def _no_grad(func: _F) -> _F:
    """Type-preserving ``torch.no_grad`` decorator (torch's own TypeVar is Any-bound)."""
    return cast(_F, torch.no_grad()(func))


# ----------------------------------------------------------------------
# Latent conditioning (v4_plan section 5.3 Gate-D0 arms)
#
# Both arms are *optional*: with ``z_slots=None`` the forward pass below
# takes byte-identical code paths to the pre-latent model, which is what
# keeps the measured Gate-0 evidence (v4_plan section 1.1) valid and makes
# the D0.3 regression check meaningful.
# ----------------------------------------------------------------------
def _reduce_slots(z_slots: torch.Tensor) -> torch.Tensor:
    """Pool ``[B, H, d_z]`` typed slots to ``[B, d_z]``; pass ``[B, d_z]`` through.

    **Lossy — kept only for the deliberately-global soft-prefix arm.** Measured at
    the M0 configuration (H=8, d_z=32), this discards 87.5% of the field by
    dimension count, and a held-out linear probe recovers just 6.8% of the
    per-slot structure from the pooled vector: **93.2% is irrecoverable**. That
    defect fully explains the D0 dissociation (a predictor 16x more accurate on
    the latent target produced no denoising benefit, because the field was
    destroyed before use). See memory `latentnet-d0-dissociation`.

    Position-addressable readouts must use :func:`_slots_to_positions` instead.
    """
    if z_slots.dim() == _Z_SLOTTED_NDIM:
        return z_slots.mean(dim=1)
    return z_slots


def _slots_to_positions(z_slots: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Position-addressable slot readout; delegates to ``models.latent_plan``.

    The implementation lives in `latent_plan` because that module is pure torch
    and importable without CUDA, which lets the CPU regression suite assert the
    readout is lossless. Re-exported here so the conditioners read naturally.
    """
    from models.latent_plan import slots_to_positions

    return slots_to_positions(z_slots, seq_len)


class LatentFiLMConditioner(TypedTorchModule):
    """Arm A1: ``z`` -> per-layer ``(gamma, beta)`` FiLM on each block's output.

    Ported from ``Agent/diffrwkv_agent/latent_comm/fidelity_injector.py``
    (``PerLayerFiLM``), adapted from recurrent-state modulation to the
    full-sequence stream this denoiser runs on. Two properties carry over
    unchanged and both are load-bearing:

    * **exact identity at init** — the output projection is zero-init, so
      ``gamma = beta = 0`` and every block computes ``h * 1 + 0``. A model
      with a freshly attached conditioner is numerically equal to its
      warm-start checkpoint, so training can only move away from it.
    * **bounded gamma** — ``out_scale * tanh(.)`` keeps the multiplicative
      term in ``[1 - out_scale, 1 + out_scale]``. ``pertoken.py`` documents
      the divergence that follows from leaving this unbounded.

    **Position-addressable since 2026-08-16.** The modulation is now computed
    per position, with slot *h* conditioning the chunk of positions it
    summarises, rather than from one whole-sequence pooled vector. The pooled
    version discarded 93.2% of the slot field (held-out probe) and was the
    measured cause of the Gate-D0 dissociation — a predictor 16x more accurate on
    the latent target produced zero denoising benefit because its output was
    destroyed at the readout. See memory `latentnet-d0-dissociation`.
    """

    if TYPE_CHECKING:
        __call__: Callable[[torch.Tensor, int], torch.Tensor]

    kind: str = "film"

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        num_layers: int,
        trunk_dim: int = 256,
        out_scale: float = 0.5,
    ) -> None:
        """Build the shared trunk and the zero-init per-layer FiLM head."""
        super().__init__()
        self.latent_dim: int = latent_dim
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.out_scale: float = out_scale

        self.trunk: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim, trunk_dim),
            nn.GELU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.GELU(),
        )
        # 2 = (gamma, beta) per layer.
        self.head: nn.Linear = nn.Linear(trunk_dim, num_layers * 2 * hidden_size)
        _ = nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            _ = nn.init.zeros_(self.head.bias)

    @override
    def forward(self, z_slots: torch.Tensor, seq_len: int = 0) -> torch.Tensor:
        """Return the shared ``[B, T, trunk_dim]`` projection for per-position FiLM.

        The earlier version pooled all slots into one ``[B, d_z]`` vector and
        emitted ``[L, B, 1, 2*hidden]`` — one modulation broadcast over every
        position. That threw away 93.2% of the slot field (held-out probe) and is
        the measured cause of the D0 dissociation. Now slot *h* conditions the
        positions it summarises, so the modulation varies along the sequence.

        ``seq_len`` is passed explicitly rather than stashed on the module: the
        conditioner is called once per forward and the length is known there, so
        threading it as an argument keeps the module stateless (and safe under
        FSDP, where hidden mutable attributes are a reliable source of surprise).

        Cost note: the full per-layer output would be ``L x B x T x 2*hidden``,
        which at the 2.9B geometry (L=32, hidden=2560) is far too large to
        materialise. Only the trunk activations are returned; :meth:`film_for_layer`
        applies this layer's slice of the head on demand.
        """
        z_pos = _slots_to_positions(z_slots, seq_len).to(self.head.weight.dtype)
        return cast("torch.Tensor", self.trunk(z_pos))  # [B, T, trunk_dim]

    def film_for_layer(self, params: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, ...]:
        """Project trunk activations to this layer's bounded ``(gamma, beta)``.

        ``params`` is the ``[B, T, trunk_dim]`` tensor from :meth:`forward`. Only
        this layer's ``2 * hidden`` slice of the head is applied, so the full
        ``L x B x T x 2*hidden`` tensor is never materialised.
        """
        lo = layer_idx * 2 * self.hidden_size
        hi = lo + 2 * self.hidden_size
        w = self.head.weight[lo:hi]  # [2*hidden, trunk_dim]
        b = self.head.bias[lo:hi] if self.head.bias is not None else None
        out = torch.nn.functional.linear(params, w, b)  # [B, T, 2*hidden]
        gamma_raw, beta = out.chunk(2, dim=-1)
        gamma = self.out_scale * torch.tanh(gamma_raw)
        return gamma, beta


class LatentCrossAttnConditioner(TypedTorchModule):
    """Arm A5: content-addressed cross-attention readout of the latent field.

    Why this arm exists. FiLM (A1) broadcasts a *position-indexed* modulation:
    each position t is modulated by whatever slot ``slots_to_positions`` assigns
    it, so the only way a token can use plan content is if the plan happens to be
    routed to its position. The measured consequence was that the conditioner
    learned to react to ``Z``'s magnitude rather than its content -- generic cost
    ``h`` ran 12-17x the content benefit ``s`` at every amplitude, and D0 failed
    with every other explanation eliminated (see v4_plan section 5.5).

    Cross-attention removes the routing assumption: every position *queries* all
    ``H`` slots and takes what it needs, so which slot matters is learned from
    content rather than fixed by index. That is the one structural degree of
    freedom A1 lacks.

    Identity at init is exact and for the same reason as A1: ``out_proj`` is
    zero-init, so the returned delta is identically zero and the block sees the
    unmodified ``h``. Gate D0.3's regression check depends on this.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        num_heads: int = 8,
        out_scale: float = 1.0,
        bound_delta: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.out_scale = out_scale
        # Default TRUE: the unbounded variant is what damaged the LM (see
        # delta_for_layer). Kept switchable so the original run stays reproducible.
        self.bound_delta = bound_delta
        if hidden_size % num_heads != 0:
            msg = f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
            raise ValueError(msg)
        self.head_dim = hidden_size // num_heads
        # Queries come from the canvas, keys/values from the latent slots.
        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(latent_dim)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(latent_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(latent_dim, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Zero-init the OUTPUT projection only. Zeroing q/k/v instead would make
        # their gradients vanish too (the product is bilinear), so the arm could
        # never start learning -- the mirror of the A1 design note.
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, z_slots: torch.Tensor, seq_len: int = 0) -> torch.Tensor:
        """Return the slot keys/values packed for :meth:`delta_for_layer`.

        ``seq_len`` is accepted for signature-compatibility with the FiLM arm's
        dispatch and is unused: cross-attention needs no position routing, which
        is precisely the property being tested.
        """
        del seq_len
        z = z_slots.unsqueeze(1) if z_slots.dim() == 2 else z_slots  # [B, H, d_z]
        return self.kv_norm(z.to(self.k_proj.weight.dtype))

    def delta_for_layer(self, kv: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Cross-attend ``h`` (queries) over the latent slots; return an additive delta."""
        b, t, _ = h.shape
        n_slots = kv.shape[1]
        q = self.q_proj(self.q_norm(h.to(self.q_proj.weight.dtype)))
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n_slots, self.num_heads, self.head_dim).transpose(1, 2)
        ctx = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        ctx = ctx.transpose(1, 2).reshape(b, t, self.hidden_size)
        delta = self.out_proj(ctx)
        if self.bound_delta:
            # BOUNDED like FiLM's gamma. Without this the delta is an UNBOUNDED
            # additive residual applied after all 32 blocks, and it measurably wrecked
            # the LM: mask_ce spiked 4.43 -> 4.83 by step 40 and was still 0.19 nats
            # above the FiLM arm at step 140, with the UNCONDITIONED value degraded too
            # (shared backbone dragged). FiLM never did this because its modulation is
            # out_scale*tanh(...), bounded to +/-out_scale.
            #
            # An arm added for COMPARISON must match the incumbent's safety properties,
            # not just its interface. The 9 CPU checks proved identity AT INIT; none
            # proved boundedness AFTER the zero-init gate opens. An init check is not a
            # stability check.
            delta = torch.tanh(delta)
        return cast("torch.Tensor", delta * self.out_scale)


class LatentSoftPrefixConditioner(TypedTorchModule):
    """Arm A3: ``z`` -> ``k`` soft prefix tokens prepended to the embeddings.

    Ported from ``Agent/diffrwkv_agent/latent_chain/softprefix.py``
    (``SoftPrefixInjector``). Near-zero init, **not** exact identity: even
    all-zero prefix embeddings still advance the RWKV scan for every real
    position that follows, so the init-time output differs slightly from the
    unconditioned model. That is a property of prefixing, not a bug — but it
    means A3 cannot be used for the D0.3 bit-parity check the way A1 can, and
    its own baseline must be an equal-length zero-prefix control rather than
    the bare checkpoint.
    """

    kind: str = "softprefix"

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        num_prefix: int = 8,
        trunk_dim: int = 256,
        init_std: float = 1e-4,
    ) -> None:
        """Build the trunk and the near-zero-init prefix projection."""
        super().__init__()
        self.latent_dim: int = latent_dim
        self.hidden_size: int = hidden_size
        self.num_prefix: int = num_prefix

        self.trunk: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim, trunk_dim),
            nn.GELU(),
        )
        self.head: nn.Linear = nn.Linear(trunk_dim, num_prefix * hidden_size)
        _ = nn.init.normal_(self.head.weight, std=init_std)
        if self.head.bias is not None:
            _ = nn.init.zeros_(self.head.bias)

    @override
    def forward(self, z_slots: torch.Tensor) -> torch.Tensor:
        """Return soft prefix embeddings as ``[B, num_prefix, hidden]``."""
        z = _reduce_slots(z_slots).to(self.head.weight.dtype)
        raw: torch.Tensor = self.head(self.trunk(z))
        return raw.view(raw.shape[0], self.num_prefix, self.hidden_size)



class BlockTimestepConditioner(TypedTorchModule):
    """v5.2 B3 / Q1 U3a: additive per-block noise-level (`t`) conditioning.

    ``BiRWKV7ForMaskedDiffusion.forward`` has never received a timestep. The absorbing-mask
    formulation lets the model *infer* noise level from the visible mask count, and Gate 0
    passes without it -- but ``v4_plan`` section 2.4 writes the token factorization as
    ``p(x_{pi_{k+1}} | x_{V_k}, m_k, tau_k, Z)``, with ``m_k`` (mask state) and ``tau_k``
    (time) in the conditioning set. Both are absent today.

    The trainer already computes a per-block realised mask fraction and, before 2026-08-18,
    threw it away. This module turns that ``[B, n_blocks]`` field into a per-position
    additive embedding, broadcast so each block's positions receive their own block's ``t``.

    Design follows ``LACES-DLM``'s ``ConditionedTokenEmbedding``
    (``masked_values + t_proj(t) + plan_proj(z)``) and the identity-at-init discipline every
    conditioning arm in this file uses:

    * **exact identity at init** -- ``out`` is zero-init, so the added term is exactly 0 and
      a model with this module attached is numerically equal to its warm-start checkpoint.
      That is what keeps the measured Gate-0 evidence valid, and it is checked, not assumed.
    * **bounded** -- ``out_scale * tanh(.)`` keeps the additive term in
      ``[-out_scale, +out_scale]``. The unbounded cross-attention arm diverged
      (``mask_ce`` 4.43 -> 4.49 by step 40) and that lesson transfers directly.

    Fourier features rather than the raw scalar: a single linear map of ``t`` can only
    express a monotone ramp, while mask fraction interacts non-monotonically with difficulty
    (the measured bucket CEs are not linear in ratio).
    """

    if TYPE_CHECKING:
        __call__: Callable[[torch.Tensor, int, int], torch.Tensor]

    def __init__(
        self,
        hidden_size: int,
        num_freqs: int = 8,
        out_scale: float = 0.5,
    ) -> None:
        """Build the Fourier-feature trunk and the zero-init output projection."""
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_freqs: int = num_freqs
        self.out_scale: float = out_scale
        # log-spaced frequencies over [1, 2^(num_freqs-1)] cycles across t in [0,1]
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(num_freqs, dtype=torch.float32), persistent=False
        )
        self.trunk: nn.Sequential = nn.Sequential(
            nn.Linear(2 * num_freqs + 1, hidden_size),
            nn.GELU(),
        )
        self.out: nn.Linear = nn.Linear(hidden_size, hidden_size)
        _ = nn.init.zeros_(self.out.weight)
        _ = nn.init.zeros_(self.out.bias)

    @override
    def forward(self, block_t: torch.Tensor, seq_len: int, block_size: int) -> torch.Tensor:
        """Map ``[B, n_blocks]`` mask fractions to a ``[B, seq_len, hidden]`` additive term.

        Each block's ``t`` is repeated across the positions of that block, then trimmed or
        zero-padded to ``seq_len``. Trailing positions beyond the last block (only possible
        when ``seq_len`` is not a multiple of ``block_size``) get zero, which is the
        identity contribution rather than an arbitrary borrowed value.
        """
        # Fourier features in fp32 for precision, then cast to the TRUNK'S OWN dtype before
        # the Linear. Under FSDP MixedPrecision the module's weights are bf16 while
        # `block_t` arrives fp32, and `mat1 and mat2 must have the same dtype` is a hard
        # error, not a silent upcast. This crashed all 8 ranks of the first U3 arm at step 1.
        # The CPU suite could not have caught it: it runs pure fp32, where the mismatch does
        # not exist. Deriving the dtype from a parameter rather than hardcoding bf16 keeps
        # the module correct in fp32 unit tests AND under bf16 training.
        w_dtype = self.trunk[0].weight.dtype
        t = block_t.to(torch.float32).unsqueeze(-1)  # [B, n_blk, 1]
        ang = t * self.freqs.view(1, 1, -1) * math.pi
        feats = torch.cat((t, torch.sin(ang), torch.cos(ang)), dim=-1).to(w_dtype)
        per_block = self.out(self.trunk(feats))  # [B, n_blk, hidden]
        per_block = self.out_scale * torch.tanh(per_block)
        wide = per_block.repeat_interleave(block_size, dim=1)  # [B, n_blk*block_size, H]
        if wide.shape[1] >= seq_len:
            return wide[:, :seq_len]
        pad = torch.zeros(
            wide.shape[0], seq_len - wide.shape[1], wide.shape[2],
            device=wide.device, dtype=wide.dtype,
        )
        return torch.cat((wide, pad), dim=1)


class BiRWKV7Block(TypedTorchModule):
    """RWKV7Block with dual-direction attention and gated fusion.

    Key names mirror fla's RWKV7Block (attn_norm / ffn / ffn_norm / pre_norm)
    so pretrained weights load with a pure key remap; the two attention copies
    live at ``attn_fwd`` / ``attn_bwd``.
    """

    if TYPE_CHECKING:
        __call__: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, bool, tuple[torch.Tensor, ...] | None],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]

    def __init__(self, config: RWKV7Config, layer_idx: int, gate_bias_init: float = 4.0) -> None:
        """Build one dual-direction block from the fla config."""
        super().__init__()
        self.config: RWKV7Config = config
        self.layer_idx: int = layer_idx
        hidden = config.hidden_size

        if config.norm_first and layer_idx == 0:
            self.pre_norm: nn.LayerNorm = nn.LayerNorm(
                hidden, bias=config.norm_bias, eps=config.norm_eps
            )
        self.attn_norm: nn.LayerNorm = nn.LayerNorm(
            hidden, bias=config.norm_bias, eps=config.norm_eps
        )

        def _make_attn() -> RWKV7Attention:
            return RWKV7Attention(
                mode=config.attn_mode,
                hidden_size=hidden,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                decay_low_rank_dim=config.decay_low_rank_dim,
                gate_low_rank_dim=config.gate_low_rank_dim,
                a_low_rank_dim=config.a_low_rank_dim,
                v_low_rank_dim=config.v_low_rank_dim,
                norm_eps=config.norm_eps,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
                value_dim=config.value_dim[layer_idx],
                num_hidden_layers=config.num_hidden_layers,
            )

        self.attn_fwd: RWKV7Attention = _make_attn()
        self.attn_bwd: RWKV7Attention = _make_attn()

        # sigmoid fusion gate: alpha = sigmoid(W x + b); W zero-init, b = +gate_bias_init
        # -> alpha ~= sigmoid(4.0) ~= 0.982 forward at init (near-causal start).
        self.fuse_proj: nn.Linear = nn.Linear(hidden, hidden, bias=False)
        _ = nn.init.zeros_(self.fuse_proj.weight)
        self.fuse_bias: nn.Parameter = nn.Parameter(torch.full((hidden,), float(gate_bias_init)))

        self.ffn_norm: nn.LayerNorm = nn.LayerNorm(
            hidden, bias=config.norm_bias, eps=config.norm_eps
        )
        self.ffn: RWKV7FeedForward = RWKV7FeedForward(
            hidden_size=hidden,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=config.num_hidden_layers,
        )

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first_fwd: torch.Tensor,
        v_first_bwd: torch.Tensor,
        force_forward: bool = False,
        film: tuple[torch.Tensor, ...] | None = None,
        state_cache: object | None = None,
        use_cache: bool = False,  # noqa: FBT001, FBT002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse forward and reverse attention streams around the pretrained FFN.

        ``film`` is this layer's optional ``(gamma, beta)`` pair from
        :class:`LatentFiLMConditioner`. When ``None`` the body is exactly the
        pre-latent computation.

        ``state_cache`` / ``use_cache`` expose the RWKV-7 **recurrent state**, which
        this class previously discarded outright. One flag pair serves both
        directions, because of the order of operations inside fla's
        ``RWKV7Attention.forward``:

        1. ``last_state = get_layer_cache(...)`` — reads whatever is in the cache;
        2. ``kernel(..., initial_state=recurrent_state, output_final_state=use_cache)``;
        3. ``update_layer_cache(..., recurrent_state=<kernel output>)``.

        So a **pre-filled** cache is consumed at step 2 regardless of ``use_cache``
        (that is arm C, injection: `Z -> state`), while ``use_cache=True`` makes the
        kernel *return* the final state so step 3 stores it (that is the encoder
        direction, extraction: `state -> Z`).

        An injected state is NOT clobbered by step 3: ``FLALayer.update`` assigns
        ``recurrent_state`` only ``if recurrent_state is not None``, and with
        ``use_cache=False`` the kernel hands back ``None``. So a pre-filled cache
        survives the forward and stays reusable — verified by reading
        ``fla/models/utils.py``, not assumed.

        With ``state_cache=None`` and ``use_cache=False`` (both defaults) every
        branch below is bitwise the pre-state computation. Gate D0.3's regression
        check depends on that.
        """
        attn_kwargs: dict[str, object] = {}
        if state_cache is not None or use_cache:
            attn_kwargs = {"past_key_values": state_cache, "use_cache": use_cache}
        residual: torch.Tensor = (
            self.pre_norm(hidden_states) if hasattr(self, "pre_norm") else hidden_states
        )
        x: torch.Tensor = self.attn_norm(residual)

        o_fwd, _, _, v_first_fwd = cast(
            "tuple[torch.Tensor, None, object | None, torch.Tensor]",
            self.attn_fwd(hidden_states=x, v_first=v_first_fwd, **attn_kwargs),
        )
        if force_forward:
            o = o_fwd
        else:
            # reverse stream operates entirely in flipped coordinates; its
            # v_first stays flipped across layers.
            x_rev = torch.flip(x, dims=(1,))
            # The reverse stream is deliberately NOT given the state cache: it runs
            # in flipped coordinates, so its recurrent state is expressed in a
            # different basis than the forward stream's and mixing the two would be
            # meaningless. Arm C therefore injects/extracts the FORWARD stream only,
            # which is also the safer choice given that the reverse stream's flip is
            # still an unresolved suspect in the A3 soft-prefix disconnection.
            o_bwd, _, _, v_first_bwd = cast(
                "tuple[torch.Tensor, None, object | None, torch.Tensor]",
                self.attn_bwd(hidden_states=x_rev, v_first=v_first_bwd),
            )
            o_bwd = torch.flip(o_bwd, dims=(1,))
            gate_pre: torch.Tensor = self.fuse_proj(x)
            alpha = torch.sigmoid(gate_pre + self.fuse_bias.to(x.dtype))
            o = alpha * o_fwd + (1.0 - alpha) * o_bwd

        hidden_states = residual + o
        if film is not None:
            gamma, beta = film
            hidden_states = hidden_states * (1.0 + gamma.to(hidden_states.dtype)) + beta.to(
                hidden_states.dtype
            )
        residual = hidden_states
        ffn_in: torch.Tensor = self.ffn_norm(hidden_states)
        ffn_out, _ = cast("tuple[torch.Tensor, object | None]", self.ffn(ffn_in))
        hidden_states = residual + ffn_out
        return hidden_states, v_first_fwd, v_first_bwd

    def mean_forward_alpha(self) -> float:
        """Diagnostic: gate bias midpoint (input-independent part only)."""
        return float(torch.sigmoid(self.fuse_bias.detach().float()).mean())


class _GradCheckpointer(Protocol):
    """Typed surface of ``torch.utils.checkpoint.checkpoint`` for block calls."""

    def __call__(  # noqa: PLR0913
        self,
        block: BiRWKV7Block,
        hidden_states: torch.Tensor,
        v_first_fwd: torch.Tensor,
        v_first_bwd: torch.Tensor,
        force_forward: bool,  # noqa: FBT001
        film: tuple[torch.Tensor, ...] | None,
        state_cache: object | None,
        use_cache: bool,  # noqa: FBT001
        *,
        use_reentrant: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


_grad_checkpoint = cast(
    _GradCheckpointer,
    getattr(_torch_checkpoint, "checkpoint"),  # noqa: B009
)


class BiRWKV7ForMaskedDiffusion(TypedTorchModule):
    """Token denoiser: embeddings -> BiRWKV7 blocks -> norm -> lm_head."""

    if TYPE_CHECKING:
        __call__: Callable[
            [torch.Tensor, bool, torch.Tensor | None, object | None, bool,
             torch.Tensor | None, int],
            torch.Tensor,
        ]

    def __init__(
        self,
        config: RWKV7Config,
        gate_bias_init: float = 4.0,
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Assemble embeddings, dual-direction blocks, final norm, and lm_head."""
        super().__init__()
        self.config: RWKV7Config = config
        self.mask_token_id: int = MASK_TOKEN_ID
        self.pad_token_id: int = PAD_TOKEN_ID
        self.gradient_checkpointing: bool = gradient_checkpointing

        self.embeddings: nn.Embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        blocks = tuple(
            BiRWKV7Block(config, i, gate_bias_init) for i in range(config.num_hidden_layers)
        )
        self.layers: nn.ModuleList = nn.ModuleList(blocks)
        self._blocks: tuple[BiRWKV7Block, ...] = blocks
        self.norm: nn.LayerNorm = nn.LayerNorm(
            config.hidden_size, bias=config.norm_bias, eps=config.norm_eps
        )
        self.lm_head: nn.Linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Per-block timestep conditioning is opt-in; ``None`` keeps the pre-t model, and
        # even when attached it is zero-init so the model is numerically unchanged.
        self.block_t_cond: BlockTimestepConditioner | None = None
        # Latent conditioning is opt-in; ``None`` keeps the pre-latent model.
        self.latent_cond: (
            LatentFiLMConditioner | LatentSoftPrefixConditioner
            | LatentCrossAttnConditioner | None
        ) = None
        # v7 architecture round (W-A0): both capacity mechanisms are opt-in;
        # ``None`` keeps the single-stream single-pass model, and even when
        # attached they are exact-identity at init (models/residual_streams.py),
        # so the warm-started checkpoint is numerically unchanged at step 0.
        self.residual_streams: ResidualStreamMixer | None = None
        self.loop: BackboneLoopControl | None = None

    # ------------------------------------------------------------------
    # Latent conditioning attachment (v4_plan section 9, Plan-1 M0)
    # ------------------------------------------------------------------
    def attach_block_timestep_conditioner(
        self,
        num_freqs: int = 8,
        out_scale: float = 0.5,
        default_block_size: int = 0,
    ) -> BlockTimestepConditioner:
        """Attach the per-block timestep arm (v5.2 B3); returns the new module.

        Zero-init, so attaching it does not change the model's output until it trains. The
        returned module's parameters are NOT inside the FSDP transformer wrap set, which is
        deliberate: like the FiLM conditioner, it is consumed once per forward rather than
        per block, so wrapping it separately would force an extra gather.
        """
        mod = BlockTimestepConditioner(
            hidden_size=int(self.config.hidden_size),
            num_freqs=num_freqs,
            out_scale=out_scale,
        )
        self.block_t_cond = mod
        # Used by forward() when no explicit block_t is passed (the generation/eval path):
        # the trainer conditions on the REALISED masked fraction of the canvas, so deriving
        # it from the input itself is semantically identical to training -- the sampler
        # needs no plumbing and cannot get out of sync with the canvas.
        self.block_t_default_size: int = int(default_block_size)
        return mod

    def block_t_parameters(self) -> list[nn.Parameter]:
        """Parameters of the timestep arm, for a separate optimizer group."""
        if self.block_t_cond is None:
            return []
        return list(self.block_t_cond.parameters())

    def attach_latent_conditioner(
        self,
        kind: str,
        latent_dim: int,
        num_prefix: int = 8,
        out_scale: float = 0.5,
    ) -> LatentFiLMConditioner | LatentSoftPrefixConditioner | LatentCrossAttnConditioner:
        """Attach a Gate-D0 conditioning arm; returns the new module.

        ``kind="film"`` (arm A1) is exact-identity at init; ``kind="softprefix"``
        (arm A3) is near-identity. ``kind="none"`` is rejected here rather than
        silently attaching nothing — a no-op conditioning lane is the failure
        mode that cost ~43 GPU-hours in round-4a, so the caller must express
        "unconditioned" by not calling this at all.
        """
        hidden = self.config.hidden_size
        if kind == "film":
            cond: (
                LatentFiLMConditioner | LatentSoftPrefixConditioner
                | LatentCrossAttnConditioner
            ) = LatentFiLMConditioner(
                latent_dim=latent_dim,
                hidden_size=hidden,
                num_layers=self.config.num_hidden_layers,
                out_scale=out_scale,
            )
        elif kind == "softprefix":
            cond = LatentSoftPrefixConditioner(
                latent_dim=latent_dim,
                hidden_size=hidden,
                num_prefix=num_prefix,
            )
        elif kind == "crossattn":
            cond = LatentCrossAttnConditioner(
                latent_dim=latent_dim,
                hidden_size=hidden,
                out_scale=out_scale,
            )
        else:
            msg = (f"unknown latent conditioner kind {kind!r} "
                   "(expected 'film', 'softprefix' or 'crossattn')")
            raise ValueError(msg)
        self.latent_cond = cond
        return cond

    def latent_parameters(self) -> list[nn.Parameter]:
        """Parameters belonging to the conditioning arm (adapter-first unfreeze)."""
        if self.latent_cond is None:
            return []
        return list(self.latent_cond.parameters())

    # ------------------------------------------------------------------
    # v7 architecture round: capacity-mechanism attachment (W-A0)
    # ------------------------------------------------------------------
    def attach_residual_streams(self, n_streams: int, mix_scale: float = 1.0) -> ResidualStreamMixer:
        """Attach the n-stream hyper-connection arm (mHC, W-A2); returns the module.

        Zero-init, so attaching it does not change the model's output until it
        trains. Root-level module (the ``latent_cond``/``block_t_cond``
        precedent): consumed once per forward, sliced per layer inside the
        loop, so it stays OUT of the ``BiRWKV7Block`` FSDP wrap set and shards
        with the root flat param. Requires ``n_streams in {2, 4}``.
        """
        if n_streams not in (2, 4):
            msg = f"n_streams must be in {{2,4}}, got {n_streams}"
            raise ValueError(msg)
        if self.residual_streams is not None:
            msg = "residual_streams already attached"
            raise RuntimeError(msg)
        mod = ResidualStreamMixer(
            num_layers=int(self.config.num_hidden_layers),
            n_streams=int(n_streams),
            mix_scale=mix_scale,
        )
        self.residual_streams = mod
        return mod

    def attach_backbone_loop(
        self,
        loop_range: tuple[int, int],
        loop_reps: int,
        loop_scale: float = 1.0,
    ) -> BackboneLoopControl:
        """Attach the weight-tied loop arm (W-A3); returns the module.

        ``loop_reps`` extra weight-tied passes of layers ``[lo, hi)`` — the
        same block modules re-run (weight-tying by construction), gated by a
        tanh-bounded per-pass scalar that is exactly 0 at init. Root-level
        module, same FSDP reasoning as the mHC arm. All keys land under the
        ``loop.`` prefix for ``_merge_latent_keys`` whitelisting.
        """
        lo, hi = loop_range
        n_layers = int(self.config.num_hidden_layers)
        if not (0 <= lo < hi <= n_layers):
            msg = f"loop_range {loop_range} outside [0,{n_layers})"
            raise ValueError(msg)
        if loop_reps not in (1, 2):
            msg = f"loop_reps must be in {{1,2}}, got {loop_reps}"
            raise ValueError(msg)
        if self.loop is not None:
            msg = "backbone loop already attached"
            raise RuntimeError(msg)
        mod = BackboneLoopControl(lo, hi, loop_reps, loop_scale)
        self.loop = mod
        return mod

    def _run_layer_range(  # noqa: PLR0913
        self,
        lo: int,
        hi: int,
        h: torch.Tensor,
        v_first_fwd: torch.Tensor,
        v_first_bwd: torch.Tensor,
        force_forward: bool,  # noqa: FBT001
        film_params: torch.Tensor | None,
        xattn_kv: torch.Tensor | None,
        state_cache: object | None,
        use_cache: bool,  # noqa: FBT001, FBT002
        apply_xattn: bool = True,  # noqa: FBT001, FBT002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run blocks ``lo..hi-1`` once — single-stream legacy or n-stream mHC.

        With no ``residual_streams`` attached this is the byte-identical
        legacy loop (same call order, same ``_grad_checkpoint`` arguments).
        With streams attached, ``h`` is ``[B, T, n, C]`` and each layer applies
        the hyper-connection update (ONE block call on the combined input —
        1x block compute, the whole point of the mechanism)::

            x      = Σ_j w_pre[l,j] · H_l[j]         (input combination, [B,T,C])
            o      = block(x)                        (single call)
            delta  = o − x                           (the block's own update)
            H_{l+1}[i] = Σ_j A_res[l,i,j] · H_l[j] + a_post[l,i] · delta

        At init ``w_pre = e_0``, ``A_res = I``, ``a_post = 1`` and the update
        collapses to the plain residual add, bit-exactly. The tiny factor
        einsums run OUTSIDE the checkpoint (matching the FiLM/cross-attn
        precedent). See ``models/residual_streams.py`` for why the streams are
        NOT fed to the block as a folded ``[B*n, T, C]`` batch (n× FLOPs) and
        why the mixing is NOT row-stochastic (degenerate on replicated
        streams).
        """
        n = self.residual_streams.n_streams if self.residual_streams is not None else 1
        for layer_idx in range(lo, hi):
            # self.layers (the ModuleList), NOT the self._blocks tuple alias:
            # FSDP replaces the wrapped modules in the ModuleList's _modules,
            # so the raw tuple holds UNWRAPPED blocks whose params are un-gathered
            # shard views (pre_norm.weight [0] crash, smoke job-e9295fb7 arm 1).
            block = self.layers[layer_idx]
            film: tuple[torch.Tensor, ...] | None = None
            if film_params is not None and isinstance(self.latent_cond, LatentFiLMConditioner):
                film = self.latent_cond.film_for_layer(film_params, layer_idx)
            if n > 1:
                assert self.residual_streams is not None  # noqa: S101  # narrowed by n > 1
                w_pre, a_res, a_post = self.residual_streams.factors(layer_idx)
                w = w_pre.to(h.dtype).view(1, 1, -1, 1)
                x = (h * w).sum(dim=2)  # [B, T, C] combination; == H[0] exactly at init
                if self.gradient_checkpointing and self.training:
                    o, v_first_fwd, v_first_bwd = _grad_checkpoint(
                        block, x, v_first_fwd, v_first_bwd,
                        force_forward, film, state_cache, use_cache, use_reentrant=False,
                    )
                else:
                    o, v_first_fwd, v_first_bwd = block(
                        x, v_first_fwd, v_first_bwd, force_forward, film,
                        state_cache, use_cache,
                    )
                delta = o - x
                h = (torch.einsum("ij,btjc->btic", a_res.to(h.dtype), h)
                     + a_post.to(h.dtype).view(1, 1, -1, 1) * delta.unsqueeze(2))
            else:
                if self.gradient_checkpointing and self.training:
                    h, v_first_fwd, v_first_bwd = _grad_checkpoint(
                        block, h, v_first_fwd, v_first_bwd,
                        force_forward, film, state_cache, use_cache, use_reentrant=False,
                    )
                else:
                    h, v_first_fwd, v_first_bwd = block(
                        h, v_first_fwd, v_first_bwd, force_forward, film,
                        state_cache, use_cache,
                    )
            if apply_xattn and xattn_kv is not None:
                if n > 1:
                    # The cross-attn conditioner computes per-position queries
                    # from a [B, T, C] state; the mHC arm is a NON-LATENT arm
                    # and never runs with it. Fail closed rather than fold.
                    msg = "crossattn latent conditioning is unsupported with residual_streams"
                    raise ValueError(msg)
                # Additive residual AFTER the block, so the arm cannot disturb the
                # RWKV state recurrences (v_first_*) that carry the any-order
                # property Gate 0 certifies. Applied outside the checkpointed call
                # for the same reason FiLM is: the delta is a function of `h`,
                # which recompute would have to reproduce anyway.
                assert isinstance(self.latent_cond, LatentCrossAttnConditioner)  # noqa: S101
                h = h + self.latent_cond.delta_for_layer(xattn_kv, h).to(h.dtype)
        return h, v_first_fwd, v_first_bwd

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        force_forward: bool = False,
        z_slots: torch.Tensor | None = None,
        state_cache: object | None = None,
        use_cache: bool = False,  # noqa: FBT001, FBT002
        block_t: torch.Tensor | None = None,
        block_size: int = 0,
        output_hidden: bool = False,
        loop_reps_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return [B, T, vocab] logits.

        Padding must be handled by the caller's loss mask; RWKV state simply
        flows through pad positions.

        ``z_slots`` is the optional typed latent field, ``[B, H, d_z]`` or
        ``[B, d_z]``. With ``z_slots=None`` — or with no conditioner attached —
        every branch below reduces to the pre-latent computation, which is what
        the D0.3 regression check verifies.

        ``block_t`` is the optional per-block realised mask fraction, ``[B, n_blocks]``,
        applied additively to the embeddings via :class:`BlockTimestepConditioner` (v5.2 B3).
        It requires ``block_size`` so each block's ``t`` reaches exactly its own positions.
        With ``block_t=None`` -- or with no ``block_t_cond`` attached -- the embedding is
        untouched, so the same D0.3 invariant holds for this arm too.
        """
        cond = self.latent_cond if z_slots is not None else None

        h: torch.Tensor = self.embeddings(input_ids)

        # Applied on the RAW embeddings, before any soft prefix is concatenated: block_t is
        # indexed by token position, and a prepended prefix would shift every block's span.
        if self.block_t_cond is not None:
            if block_t is None:
                # Auto-derive from the canvas. Training fed the realised masked fraction of
                # the VISIBLE canvas, so this reproduces the training semantics exactly and
                # requires the sampler to pass nothing. Needs the default block size set at
                # attach time; without it, auto-derivation is ambiguous and must not guess.
                if getattr(self, "block_t_default_size", 0) <= 0:
                    msg = ("block_t_cond attached but no block_t passed and no "
                           "default_block_size set; pass block_t or attach with "
                           "default_block_size>0")
                    raise ValueError(msg)
                bs = self.block_t_default_size
                n_blk = (h.shape[1] + bs - 1) // bs
                maskpos = (input_ids == self.mask_token_id)
                pad = n_blk * bs - h.shape[1]
                if pad:
                    maskpos = torch.nn.functional.pad(maskpos, (0, pad))
                blk_mask = maskpos.view(input_ids.shape[0], n_blk, bs)
                blk_t = blk_mask.float().mean(dim=2)
                block_t, block_size = blk_t, bs
            if block_size <= 0:
                msg = "block_t requires block_size > 0 to map blocks onto positions"
                raise ValueError(msg)
            h = h + self.block_t_cond(block_t, h.shape[1], block_size).to(h.dtype)

        n_prefix = 0
        if cond is not None and isinstance(cond, LatentSoftPrefixConditioner):
            assert z_slots is not None  # noqa: S101  # narrowed by `cond is not None`
            prefix = cond(z_slots).to(h.dtype)
            h = torch.cat((prefix, h), dim=1)
            n_prefix = prefix.shape[1]

        film_params: torch.Tensor | None = None
        if cond is not None and isinstance(cond, LatentFiLMConditioner):
            assert z_slots is not None  # noqa: S101  # narrowed by `cond is not None`
            # Pass the CURRENT sequence length (post-prefix, if any) so slot h
            # modulates exactly the positions it summarises.
            film_params = cond(z_slots, h.shape[1])

        xattn_kv: torch.Tensor | None = None
        if cond is not None and isinstance(cond, LatentCrossAttnConditioner):
            assert z_slots is not None  # noqa: S101  # narrowed by `cond is not None`
            xattn_kv = cond(z_slots)

        v_first_fwd = torch.zeros_like(h)
        v_first_bwd = torch.zeros_like(h)
        # v7 W-A0: stream expansion (mHC) + weight-tied loop, both exact
        # identity at init. With neither mechanism attached the path below is
        # the legacy single-stream single-pass loop, byte-identical.
        n_streams = self.residual_streams.n_streams if self.residual_streams is not None else 1
        if n_streams > 1:
            # The recurrent state cache is a single-stream contract; the mHC arm
            # never runs with it (trainer/eval never pass both).
            if state_cache is not None:
                msg = "state_cache is unsupported with residual_streams attached"
                raise ValueError(msg)
            h = expand_to_streams(h, n_streams)
            # v_first_* stay [B, T, C]: the block processes ONE combination of
            # the streams per layer (1x compute), so the highway is single.
        h, v_first_fwd, v_first_bwd = self._run_layer_range(
            0, int(self.config.num_hidden_layers), h, v_first_fwd, v_first_bwd,
            force_forward, film_params, xattn_kv, state_cache, use_cache,
        )
        if self.loop is not None:
            # Weight-tied extra passes of [lo, hi): the SAME block modules
            # re-run (weight-tying by construction), gated by tanh-bounded
            # per-pass scalars that are exactly 0 at init. v_first is CARRIED
            # between passes (a re-pass must inherit the highway state the
            # first pass delivered -- more depth, not reset depth); post-loop
            # v_first updates are discarded (nothing downstream re-runs), so
            # h/logits are unchanged at init. ``loop_reps_override`` enables
            # depth-extrapolation probes at eval (reps > trained).
            lo_l = int(self.loop.lo)
            hi_l = int(self.loop.hi)
            reps = self.loop.reps if loop_reps_override is None else int(loop_reps_override)
            gates = self.loop.gates()
            for p in range(reps):
                g_p = gates[min(p, gates.numel() - 1)]  # extrapolation reuses the last trained gate
                h_pass, _, _ = self._run_layer_range(
                    lo_l, hi_l, h, v_first_fwd, v_first_bwd,
                    force_forward, film_params, xattn_kv, state_cache, use_cache,
                    apply_xattn=False,  # conditioning re-injection stays first-pass-only
                )
                h = h + g_p.to(h.dtype) * (h_pass - h)
        if n_streams > 1:
            h = readout(h)
        if n_prefix:
            # Drop the soft prefix so logits stay aligned with ``input_ids``.
            h = h[:, n_prefix:, :]
        normed_h: torch.Tensor = self.norm(h)
        logits: torch.Tensor = self.lm_head(normed_h)
        if output_hidden:
            # Candidate-H capture path (v5.3 §2): expose the final per-position
            # hidden activations (post-block, post-FiLM (FiLM is inside the block
            # call), pre-norm) so the hidden-pool block encoder has something to
            # pool. bf16 under FSDP; callers upcast at their interface (the T5
            # normalization site), matching the S-readout convention.
            return logits, h
        return logits

    def mean_forward_alpha(self) -> float:
        """Average the per-block forward-gate midpoint across all layers."""
        return sum(b.mean_forward_alpha() for b in self.layers) / len(self.layers)

    # ------------------------------------------------------------------
    # HF warm-start
    # ------------------------------------------------------------------
    @staticmethod
    def remap_hf_key(key: str) -> list[str]:
        """Map one HF RWKV7ForCausalLM key to our key(s).

        Attention weights map to BOTH direction copies (reverse = clone of
        forward).
        """
        if key.startswith("model.layers.") and ".attn." in key and ".attn_norm" not in key:
            head, tail = key.split(".attn.", 1)
            idx = head.removeprefix("model.layers.")
            return [f"layers.{idx}.attn_fwd.{tail}", f"layers.{idx}.attn_bwd.{tail}"]
        if key.startswith("model."):
            return [key.removeprefix("model.")]
        return [key]  # lm_head.weight

    @classmethod
    def from_hf_pretrained(
        cls,
        model_dir: str | Path,
        dtype: torch.dtype = torch.bfloat16,
        gate_bias_init: float = 4.0,
        gradient_checkpointing: bool = False,  # noqa: FBT001, FBT002
        n_streams: int = 1,
        mix_scale: float = 1.0,
        loop_range: tuple[int, int] | None = None,
        loop_reps: int = 0,
        loop_scale: float = 1.0,
    ) -> BiRWKV7ForMaskedDiffusion:
        """Warm-start from an HF RWKV7ForCausalLM directory via key remap.

        ``n_streams > 1`` attaches the mHC arm and ``loop_reps > 0`` (with
        ``loop_range``) attaches the weight-tied loop arm — both AFTER the
        strict HF load so the mapping-completeness check never sees the new
        keys, and both zero-init so the warm-started output is numerically
        unchanged at step 0. Defaults (off) preserve every existing caller.
        """
        from safetensors import torch as _safetensors_torch

        load_file = cast(
            "Callable[[str], dict[str, torch.Tensor]]",
            getattr(_safetensors_torch, "load_file"),  # noqa: B009
        )

        model_dir = Path(model_dir)
        cfg_json = cast("dict[str, object]", json.loads((model_dir / "config.json").read_text()))
        _ = cfg_json.pop("auto_map", None)
        _ = cfg_json.pop("architectures", None)
        # Build via kwargs: __init__ derives num_heads and the per-layer
        # value_dim list; post-hoc setattr would leave them stale (GroupNorm
        # divisibility crash at 2.9B geometry).
        probe = RWKV7Config()
        filtered = {k: v for k, v in cfg_json.items() if hasattr(probe, k)}
        config = RWKV7Config(**filtered)

        model = cls(config, gate_bias_init, gradient_checkpointing)

        index_file = model_dir / "model.safetensors.index.json"
        if index_file.exists():
            weight_map = cast(
                "dict[str, str]", json.loads(index_file.read_text())["weight_map"]
            )
            shard_files = sorted(set(weight_map.values()))
        else:
            shard_files = ["model.safetensors"]

        remapped: dict[str, torch.Tensor] = {}
        for shard in shard_files:
            for hf_key, tensor in load_file(str(model_dir / shard)).items():
                for our_key in cls.remap_hf_key(hf_key):
                    remapped[our_key] = tensor

        incompatible = model.load_state_dict(remapped, strict=False)
        missing = cast("list[str]", getattr(incompatible, "missing_keys"))  # noqa: B009
        unexpected = cast("list[str]", getattr(incompatible, "unexpected_keys"))  # noqa: B009
        # only the fusion gates may be missing; anything else is a mapping bug.
        bad_missing = [k for k in missing if "fuse_proj" not in k and "fuse_bias" not in k]
        if bad_missing or unexpected:
            msg = (
                f"HF warm-start mapping incomplete: missing={bad_missing[:8]} "
                f"unexpected={list(unexpected)[:8]}"
            )
            raise RuntimeError(msg)
        # v7 W-A0: attach the capacity mechanisms AFTER the strict HF load
        # (see docstring). Seeded AFTER, matching the trainer's attach-before-FSDP
        # ordering — the trainer calls the attach methods directly, so this
        # path exists for parity scripts and eval loaders.
        if n_streams > 1:
            _ = model.attach_residual_streams(n_streams, mix_scale)
        if loop_reps > 0:
            if loop_range is None:
                msg = "loop_reps > 0 requires loop_range"
                raise ValueError(msg)
            _ = model.attach_backbone_loop(loop_range, loop_reps, loop_scale)
        return model.to(dtype)


# ----------------------------------------------------------------------
# Iterative denoising sampler (port of state_prefill_rwkv denoise_block,
# global-sequence variant: the masked positions of the WHOLE input are
# iteratively committed by confidence).
# ----------------------------------------------------------------------
@_no_grad
def iterative_denoise(  # noqa: PLR0913
    model: BiRWKV7ForMaskedDiffusion,
    corrupted: torch.Tensor,
    masked: torch.Tensor,
    steps: int = 16,
    temperature: float = 0.0,
    self_correction: bool = False,  # noqa: FBT001, FBT002
    remask_threshold: float = 0.25,
    commit_group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-denoise ``corrupted`` (mask ids at ``masked`` positions).

    Linear commit schedule: each step commits the highest-confidence masked
    positions so that ~(step/steps) of the originally-masked set is clean.
    Returns (denoised ids, commit_step [B,T]; 0 for never-masked positions).

    ``commit_group_size`` (v5.2 B2 / Q1 U2) bounds how many positions may be committed
    from a **single** forward pass.

    * ``0`` (default) — legacy behaviour: every position this step needs is committed from
      one forward, i.e. their joint is approximated by the **product of independent
      per-position marginals**. ``v4_plan`` section 2.4 forbids exactly this for grouped
      commits ("never a product of independent per-position marginals").
    * ``g >= 1`` — commit at most ``g`` positions per forward, then **re-run the forward on
      the updated canvas** before choosing the next group. Each group therefore conditions
      on the tokens the previous group actually wrote, which is a correct sequential
      factorization. ``g = 1`` is exact mode (one token per forward); larger ``g`` trades
      exactness for wall-clock.

    Why this exists: the measured repetition degeneracy is ordered by
    **commits-per-step** with Spearman rho = -1.000, R^2 = 0.891 over 7 points, and a
    falsifier confirmed it across a 4x canvas range (the same canvas at 16 vs 64
    commits/step gave 77.4% vs 29.8% max-run/length). ``em@32 - em@1`` also goes negative
    at >=80% mask. All of that is the signature of committing many positions as if they
    were independent. Setting ``g`` small makes the *sampler* pay for correctness rather
    than the *model*, and needs no retraining.

    Cost: the number of forwards becomes ``sum_step ceil(k_step / g)`` instead of ``steps``.
    At ``g = 1`` this is one forward per committed token, which is why exact mode is
    reserved for measurement and small canvases.
    """
    if commit_group_size < 0:
        msg = f"commit_group_size must be >= 0, got {commit_group_size}"
        raise ValueError(msg)
    cur = corrupted.clone()
    still = masked.clone()
    commit_step = torch.zeros_like(cur)
    total = masked.sum(dim=1)  # [B]

    for step in range(1, steps + 1):
        if not still.any():
            break
        logits = model(cur, False).float()  # noqa: FBT003
        logits[..., model.mask_token_id] = float("-inf")
        logits[..., model.pad_token_id] = float("-inf")
        probs = logits.softmax(dim=-1)
        if temperature > 0.0:
            shaped = (logits / max(temperature, 1e-6)).softmax(dim=-1)
            pred = torch.multinomial(shaped.reshape(-1, shaped.shape[-1]), 1).view_as(cur)
        else:
            pred = probs.argmax(dim=-1)
        conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)

        update = still.clone()
        if self_correction and step < steps:
            cur_conf = probs.gather(-1, cur.unsqueeze(-1)).squeeze(-1)
            reopen = (~still) & masked & (cur_conf < remask_threshold)
            update |= reopen
        conf = conf.masked_fill(~update, float("-inf"))

        if step == steps:
            commit = update
        else:
            commit = torch.zeros_like(update)
            target_clean = (total.float() * step / steps).long()
            already_clean = (masked & ~update).sum(dim=1)
            need = (target_clean - already_clean).clamp_min(0)
            for row in range(cur.shape[0]):
                k = min(int(need[row]), int(update[row].sum()))
                if k > 0:
                    _, idx = torch.topk(conf[row], k=k)
                    commit[row, idx] = True

        # Reopened positions go back to MASK regardless of grouping: that is part of
        # this step's canvas edit, not a commit.
        reopened = update & ~commit & ~still
        cur = torch.where(reopened, torch.full_like(cur, model.mask_token_id), cur)

        if commit_group_size <= 0:
            # Legacy: one forward, all of `commit` written at once (product of marginals).
            cur = torch.where(commit, pred, cur)
            uncommitted = commit & (commit_step == 0)
            commit_step = torch.where(
                uncommitted, torch.full_like(commit_step, step), commit_step)
        else:
            # Grouped sequential commit: write at most `commit_group_size` positions,
            # then RE-RUN the forward so the next group sees what was actually written.
            # `commit` is the step's budget; `pending` shrinks as groups are written.
            pending = commit.clone()
            group_pred, group_conf = pred, conf
            while bool(pending.any()):
                sel = torch.zeros_like(pending)
                gconf = group_conf.masked_fill(~pending, float("-inf"))
                for row in range(cur.shape[0]):
                    n_row = int(pending[row].sum())
                    if n_row == 0:
                        continue
                    k = min(commit_group_size, n_row)
                    _, idx = torch.topk(gconf[row], k=k)
                    sel[row, idx] = True
                cur = torch.where(sel, group_pred, cur)
                newly = sel & (commit_step == 0)
                commit_step = torch.where(
                    newly, torch.full_like(commit_step, step), commit_step)
                pending = pending & ~sel
                if not bool(pending.any()):
                    break
                # Refresh predictions on the updated canvas. This is the whole point of
                # the arm: without it the remaining groups would still be scored against
                # the pre-commit canvas, which is the product-of-marginals behaviour the
                # flag exists to avoid.
                g_logits = model(cur, False).float()  # noqa: FBT003
                g_logits[..., model.mask_token_id] = float("-inf")
                g_logits[..., model.pad_token_id] = float("-inf")
                g_probs = g_logits.softmax(dim=-1)
                if temperature > 0.0:
                    shaped = (g_logits / max(temperature, 1e-6)).softmax(dim=-1)
                    group_pred = torch.multinomial(
                        shaped.reshape(-1, shaped.shape[-1]), 1).view_as(cur)
                else:
                    group_pred = g_probs.argmax(dim=-1)
                group_conf = g_probs.gather(-1, group_pred.unsqueeze(-1)).squeeze(-1)

        still = update & ~commit

    return cur, commit_step


def kendall_tau_commit_order(commit_step: torch.Tensor, masked: torch.Tensor) -> float:
    """Measure Kendall tau-b of commit order against left-to-right position.

    Computed over masked positions only (any-order-ness diagnostic; tau=1
    means strictly L2R).
    """
    taus: list[float] = []
    for row in range(commit_step.shape[0]):
        steps = commit_step[row][masked[row]].float()
        n = steps.numel()
        if n < _MIN_KENDALL_POINTS:
            continue
        pos = torch.arange(n, dtype=torch.float32, device=steps.device)
        d_steps = steps.unsqueeze(0) - steps.unsqueeze(1)
        d_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        iu = torch.triu_indices(n, n, offset=1)
        s = torch.sign(d_steps[iu[0], iu[1]]) * torch.sign(d_pos[iu[0], iu[1]])
        concordant = (s > 0).sum().item()
        discordant = (s < 0).sum().item()
        ties = (s == 0).sum().item()
        denom = math.sqrt((concordant + discordant + ties) ** 2)  # tau-a with tie damping
        if concordant + discordant > 0:
            taus.append((concordant - discordant) / denom)
    return sum(taus) / len(taus) if taus else 0.0
