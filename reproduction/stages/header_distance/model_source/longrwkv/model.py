"""The tied-direction bidirectional Long-RWKV denoiser.

Two departures from the prior implementation, both structural:

**One mixer, invoked twice.**  The earlier block built two ``RWKV7Attention``
modules (``attn_fwd`` and ``attn_bwd``) with independent parameters, so the
"bidirectional" model was a different, roughly twice-as-wide-in-the-mixer model.
Here a single mixer runs the forward scan and then the per-document reversed scan
with separate zero-initialized states.  ``assert_parameter_identity`` checks that
no parameter name contains ``attn_bwd`` and that every parameter object appears
once, because "two tensors with identical initial values" is not parameter
sharing and is exactly what a state-dict traversal has to rule out.

**Per-document reversal.**  The earlier model reversed with
``torch.flip(x, dims=(1,))``, which reorders the documents inside a packed row.
``reversal.build_seq_ctx`` supplies the segment layout and ``ctx.rev`` the
involutive index map, so each document is reversed inside its own boundaries and
padding never joins a document's recurrence.

The value highway follows the appendix's conservative convention: the base pass's
layer-0 ``v`` is carried across layers, and recycled passes receive that same
fixed highway input and discard the passes' own updates.

**Per-block gradient checkpointing, off by default and on in training.**  It is
not an optimization here: the flattened row is ``microbatch 8 x seq 4096 = 32768``
tokens, so one ``[N, D]`` bf16 tensor is 64 MiB and one ``[N, 4*D]`` feed-forward
tensor 256 MiB.  Retaining a block's internals for both directions across 24
layers is tens of GiB, and the hidden loop multiplies the layer count again:
``L_eff(4) = 24 + 3*12 = 60`` block passes.  The sibling 0.4B ablation whose
41-43 GB per card is this campaign's only measured memory figure ran
``--gradient-checkpointing`` at ``--loop-reps 1``; this trainer runs up to R=4, so
the same budget needs at least the same recomputation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _checkpoint

from fla.models.rwkv7.modeling_rwkv7 import RWKV7FeedForward

from .arms import (ARM_SPECS, AUTOREGRESSIVE, MASKED_DENOISING, OBJECTIVES,
                   ArmSpec, require_distinct_arms)
from .conditioning import PHI_DIM, GateModulator, InputNoiseConditioner, phi_features
from .fla_cond_attention import CondRWKV7Attention
from .ids import VOCAB_SIZE
from .keymap import NEW_MODULE_PREFIXES, is_new_module_parameter, remap_hf_key
from .refusal import Refusal
from .reversal import SeqCtx, build_seq_ctx


@dataclass
class LongRWKVConfig:
    """Effective architecture.  Recorded verbatim in the run registry."""

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    vocab_size: int = VOCAB_SIZE
    head_dim: int = 64
    intermediate_size: int = 4096
    hidden_act: str = "sqrelu"
    decay_low_rank_dim: int = 64
    a_low_rank_dim: int = 64
    gate_low_rank_dim: int = 128
    v_low_rank_dim: int = 32
    norm_eps: float = 1e-5
    norm_bias: bool = True
    fuse_norm: bool = False
    norm_first: bool = True
    hidden_ratio: float = 4.0
    # --- the new modules ---
    fusion_rank: int = 64
    fusion_bias_init: float = 4.0
    gate_mod_rank: int = 32
    phi_dim: int = PHI_DIM
    input_cond_scale: float = 0.5
    # --- the hidden loop ---
    recycle_lo: int = 12
    recycle_hi: int = 24
    loop_multiplicities: tuple[int, ...] = (1, 2, 4)

    def to_record(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


#: The arm table is re-exported from :mod:`longrwkv.arms` (see the import above) --
#: a module with no import that can fail -- because this module imports ``fla`` at
#: module scope and so cannot import on a host without a Triton driver.  While the
#: table lived here, every CPU test that asked what an arm is had to skip, and that
#: is how A0 and M6 came to be field-for-field identical.  Importers keep using
#: ``longrwkv.model.ARM_SPECS``; the definition is one file away.
__arms_reexported__ = ("ARM_SPECS", "ArmSpec", "AUTOREGRESSIVE",
                       "MASKED_DENOISING", "OBJECTIVES", "require_distinct_arms")


class TiedBiRWKV7Block(nn.Module):
    """One mixer, called forward and per-document-reversed, then direction-fused."""

    def __init__(self, config: LongRWKVConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        if config.norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(config.hidden_size, bias=config.norm_bias,
                                         eps=config.norm_eps)
        self.attn_norm = nn.LayerNorm(config.hidden_size, bias=config.norm_bias,
                                      eps=config.norm_eps)
        self.attn = CondRWKV7Attention(
            mode="chunk",
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            decay_low_rank_dim=config.decay_low_rank_dim,
            gate_low_rank_dim=config.gate_low_rank_dim,
            a_low_rank_dim=config.a_low_rank_dim,
            v_low_rank_dim=config.v_low_rank_dim,
            norm_eps=config.norm_eps,
            fuse_norm=config.fuse_norm,
            layer_idx=layer_idx,
            value_dim=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
        )
        # Low-rank directional fusion.  W_up is zero-initialized so at step 0 the
        # gate is sigmoid(b_g) -- a fixed directional prior, not a learned one --
        # which is what the appendix means by "zero output projection preserves
        # the chosen initial directional bias".
        self.fuse_down = nn.Linear(config.hidden_size, config.fusion_rank, bias=False)
        self.fuse_up = nn.Linear(config.fusion_rank, config.hidden_size, bias=False)
        nn.init.zeros_(self.fuse_up.weight)
        self.fuse_bias = nn.Parameter(
            torch.full((config.hidden_size,), config.fusion_bias_init))

        self.ffn_norm = nn.LayerNorm(config.hidden_size, bias=config.norm_bias,
                                     eps=config.norm_eps)
        self.ffn = RWKV7FeedForward(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_idx=layer_idx,
            num_hidden_layers=config.num_hidden_layers,
        )

    def forward(self, hidden_states: torch.Tensor,
                v_first_fwd: torch.Tensor | None,
                v_first_bwd: torch.Tensor | None,
                ctx: SeqCtx,
                gate_bias: tuple[torch.Tensor, torch.Tensor] | None = None,
                force_forward: bool = False,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.pre_norm(hidden_states) if hasattr(self, "pre_norm") \
            else hidden_states
        x = self.attn_norm(residual)

        out_fwd, _, _, v_first_fwd = self.attn(
            x, v_first=v_first_fwd, cu_seqlens=ctx.cu, gate_bias=gate_bias)

        if force_forward:
            fused = out_fwd
        else:
            # The same mixer, the same segment boundaries, the reversed token order.
            x_rev = x[:, ctx.rev]
            # The same segment-reversal index that reorders x reorders the
            # biases: they are token-ordered rows, so the permutation is the
            # first axis, not the sequence axis of a batch-shaped tensor.
            gate_bias_rev = None if gate_bias is None else \
                tuple(bias[ctx.rev] for bias in gate_bias)
            out_bwd, _, _, v_first_bwd = self.attn(
                x_rev, v_first=v_first_bwd, cu_seqlens=ctx.cu,
                gate_bias=gate_bias_rev)
            # rev is an involution, so this same map returns to the forward frame
            out_bwd = out_bwd[:, ctx.rev]
            gate = torch.sigmoid(
                self.fuse_up(torch.tanh(self.fuse_down(x))) + self.fuse_bias)
            fused = gate * out_fwd + (1.0 - gate) * out_bwd

        hidden_states = residual + fused
        hidden_states = hidden_states + self.ffn(
            self.ffn_norm(hidden_states), cu_seqlens=ctx.cu)[0]
        return hidden_states, v_first_fwd, v_first_bwd


class LongRWKV(nn.Module):
    """The denoiser: embeddings, tied bidirectional blocks, hidden loop, gathered head."""

    def __init__(self, config: LongRWKVConfig, arm: str = "A3") -> None:
        super().__init__()
        if arm not in ARM_SPECS:
            raise Refusal(f"unknown arm {arm!r}; expected one of {sorted(ARM_SPECS)}")
        self.config = config
        self.spec = ARM_SPECS[arm]
        self.arm = arm
        # Off by default so every eval and diagnostic path keeps its plain
        # forward; ``train.py`` turns it on, and ``_run_range`` also requires
        # ``self.training`` and an enabled grad, so a no_grad measurement never
        # pays the recomputation.
        self.gradient_checkpointing = False

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TiedBiRWKV7Block(config, i) for i in range(config.num_hidden_layers))
        self.norm = nn.LayerNorm(config.hidden_size, bias=config.norm_bias,
                                 eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # A2 and A3 both carry input-level conditioning; A3 adds the gate path.
        # Keeping them separable is what makes A3-minus-A2 a gate contrast.
        if self.spec.bidirectional:
            self.input_cond = InputNoiseConditioner(
                config.hidden_size, config.phi_dim, config.input_cond_scale)
        if self.spec.gate_mod:
            self.gate_mods = nn.ModuleList(
                GateModulator(config.hidden_size, config.gate_mod_rank, config.phi_dim)
                for _ in range(config.num_hidden_layers))

    # -- cost accounting ---------------------------------------------------
    def blocks_per_call(self, R: int) -> int:
        """Executed block passes for one full denoiser call at multiplicity ``R``.

        ``L_eff(R) = L + (R-1) * (hi - lo)``: the base pass runs every layer, and
        each extra pass runs only the recycle range.
        """
        if R < 1:
            raise Refusal(f"R must be >= 1, got {R}")
        lo, hi = self.config.recycle_lo, self.config.recycle_hi
        extra = (R - 1) * (hi - lo) if self.spec.hidden_loop else 0
        if not self.spec.hidden_loop and R != 1:
            raise Refusal(
                f"arm {self.arm} does not implement a hidden loop, so R={R} is not a "
                f"cost it can pay; use R=1 or select a looping arm")
        return self.config.num_hidden_layers + extra

    def effective_layers(self, R: int) -> int:
        return self.blocks_per_call(R)

    # -- the pass ----------------------------------------------------------
    def _run_range(self, lo: int, hi: int, hidden_states: torch.Tensor,
                   v_first_fwd, v_first_bwd, ctx: SeqCtx, gate_biases, force_forward: bool):
        for index in range(lo, hi):
            bias = None if gate_biases is None else gate_biases[index]
            block = self.layers[index]
            if self.gradient_checkpointing and self.training \
                    and torch.is_grad_enabled():
                # ``use_reentrant=False``: ``ctx`` is a dataclass and
                # ``force_forward`` a bool, and the reentrant implementation
                # accepts only tensor arguments.  It is also the variant that
                # tolerates a block returning three values.
                hidden_states, v_first_fwd, v_first_bwd = _checkpoint(
                    block, hidden_states, v_first_fwd, v_first_bwd, ctx, bias,
                    force_forward, use_reentrant=False)
            else:
                hidden_states, v_first_fwd, v_first_bwd = block(
                    hidden_states, v_first_fwd, v_first_bwd, ctx, bias, force_forward)
        return hidden_states, v_first_fwd, v_first_bwd

    def forward(self,
                input_ids: torch.Tensor,
                doc_starts: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                ctx: SeqCtx | None = None,
                block_t: torch.Tensor | None = None,
                codes: torch.Tensor | None = None,
                gather_idx: torch.Tensor | None = None,
                R: int = 1,
                force_forward: bool | None = None,
                block_size: int | None = None,
                ) -> torch.Tensor:
        """One denoiser pass.  ``input_ids`` is ``[B, T]``; only ``B == 1`` is supported.

        Returns ``[Q, V]`` logits when ``gather_idx`` is given (the target-only head)
        and ``[1, T, V]`` otherwise.

        ``block_size`` is the grid ``block_t`` was built on, and it must be passed
        whenever the width is not a multiple of it: the producer tiles with
        ``ceil(T / block_size)`` and the size is not recoverable from the two shapes
        (13 blocks over 3104 positions is consistent with any size in 239..258).
        Omitting it keeps the derivation, which is correct only for the clean
        multiples -- every 4096-wide training row, and nothing on the LongBench
        canvases, whose widths are ``len(prompt) + answer_length``.
        """
        if input_ids.dim() != 2:
            raise Refusal(f"input_ids must be [B, T], got {tuple(input_ids.shape)}")
        batch, width = input_ids.shape
        if batch != 1:
            raise Refusal(
                f"the varlen kernel contract requires B=1 (flattened), got batch {batch}")

        if force_forward is None:
            force_forward = self.spec.causal_only
        if ctx is None:
            if doc_starts is None and attention_mask is None:
                raise Refusal("one of ctx, doc_starts or attention_mask is required")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            if doc_starts is None:
                doc_starts = torch.zeros((batch, 1), dtype=torch.long,
                                         device=input_ids.device)
            ctx = build_seq_ctx(doc_starts, attention_mask.bool(), row_len=width)
        if ctx.batch != batch or ctx.row_len != width:
            raise Refusal(
                f"ctx describes {ctx.batch}x{ctx.row_len} but input is {batch}x{width}")

        hidden_states = self.embeddings(input_ids)

        gate_biases = None
        if self.spec.gate_mod or self.spec.bidirectional:
            if block_t is None or codes is None:
                raise Refusal(
                    f"arm {self.arm} conditions on the noise coordinate, so block_t "
                    f"and codes are required")
            phi = phi_features(block_t, codes, block_size)
            if self.spec.gate_mod:
                # The mixer's declared contract is a token-ordered
                # ``[N, key_dim]`` bias (``CondRWKV7Attention`` checks
                # ``shape[0] == B*T``), while the modulator emits the model's
                # batch-shaped ``[B, T, key_dim]``.  Passing the 3-D form made
                # the mixer read the batch axis as a token count -- 1 row,
                # expected N -- and refuse every call of every check.
                # B == 1 is already enforced, so the flat reshape is exactly
                # the token order.
                gate_biases = [
                    tuple(bias.reshape(-1, bias.shape[-1])
                           for bias in modulator(phi))
                    for modulator in self.gate_mods]
            # phi is [B, T, phi_dim] and the conditioner already returns [B, T, D];
            # a leading unsqueeze here made the sum 4-D, which every LayerNorm passed
            # through undisturbed and only the mixer's [B, T, D] unpack caught -- on
            # the GPU, three qualification rounds deep.
            hidden_states = hidden_states + self.input_cond(phi)

        hidden_states, v_first_fwd, v_first_bwd = self._run_range(
            0, self.config.num_hidden_layers, hidden_states, None, None, ctx,
            gate_biases, force_forward)

        if self.spec.hidden_loop and R > 1:
            lo, hi = self.config.recycle_lo, self.config.recycle_hi
            if not 0 <= lo < hi <= self.config.num_hidden_layers:
                raise Refusal(
                    f"recycle range ({lo}, {hi}) is not inside "
                    f"[0, {self.config.num_hidden_layers})")
            for _ in range(R - 1):
                # The base pass's highway input is held fixed and the passes' own
                # updates are discarded -- the appendix's conservative convention.
                hidden_states, _, _ = self._run_range(
                    lo, hi, hidden_states, v_first_fwd, v_first_bwd, ctx,
                    gate_biases, force_forward)
        elif R != 1:
            raise Refusal(f"R={R} requested but arm {self.arm} has no hidden loop")

        hidden_states = self.norm(hidden_states)

        if gather_idx is not None:
            # Gather first, then project: a full 65536-position fp32 logit array is
            # 16 GiB where 256 positions are 64 MiB.  This is a correctness-adjacent
            # cost decision, not an optimization -- the alternative does not fit.
            if gather_idx.dim() != 1:
                raise Refusal(
                    f"gather_idx must be 1-D flat positions, got {tuple(gather_idx.shape)}")
            gathered = hidden_states[0].index_select(0, gather_idx)
            return self.lm_head(gathered)
        return self.lm_head(hidden_states)

    # -- provenance and identity ------------------------------------------
    def assert_parameter_identity(self) -> dict:
        """Directional sharing, checked structurally rather than assumed.

        Two independently constructed mixers with identical initial values would
        pass a value comparison; they fail here because the parameter objects differ.
        """
        names = [name for name, _ in self.named_parameters()]
        duplicated_weight = [name for name in names if "attn_bwd" in name]
        if duplicated_weight:
            raise Refusal(
                f"{len(duplicated_weight)} parameters belong to a second mixer "
                f"(e.g. {duplicated_weight[0]!r}); the directions must share one")
        parameters = list(self.parameters())
        unique = {id(parameter) for parameter in parameters}
        if len(unique) != len(parameters):
            raise Refusal(
                f"{len(parameters) - len(unique)} parameter objects are registered "
                f"more than once; unique-parameter traversal would double-count")
        return {
            "arm": self.arm,
            "deployed_parameter_count": sum(p.numel() for p in parameters),
            "unique_parameter_objects": len(unique),
            "mixer_copies": 1,
        }

    @classmethod
    def from_hf_pretrained(cls, checkpoint: str | Path, config: LongRWKVConfig,
                           arm: str = "A3", dtype: torch.dtype = torch.bfloat16,
                           strict_missing_prefixes: tuple[str, ...] = NEW_MODULE_PREFIXES,
                           ) -> "LongRWKV":
        """Load the released 0.4B backbone into the tied model.

        The key remap targets **one** mixer.  The prior implementation's
        ``remap_hf_key`` wrote every attention tensor into both copies, which is
        precisely the parameter doubling this design removes.
        """
        from safetensors.torch import load_file

        checkpoint = Path(checkpoint)
        weights_path = checkpoint / "model.safetensors" if checkpoint.is_dir() \
            else checkpoint
        raw = load_file(str(weights_path))

        model = cls(config, arm=arm)
        remapped: dict[str, torch.Tensor] = {}
        attention_targets = 0
        for key, tensor in raw.items():
            new_key = remap_hf_key(key)
            # exactly one destination per tensor: no mirroring into a second mixer
            if ".attn." in new_key:
                attention_targets += 1
            if new_key in remapped:
                raise Refusal(
                    f"two checkpoint tensors map to {new_key!r}; the remap is not "
                    f"one-to-one and a module would be silently overwritten")
            remapped[new_key] = tensor.to(dtype)

        own = set(model.state_dict().keys())
        unexpected = sorted(set(remapped) - own)
        if unexpected:
            raise Refusal(
                f"{len(unexpected)} checkpoint tensors have no destination, e.g. "
                f"{unexpected[:3]}")

        missing = sorted(own - set(remapped))
        allowed = [name for name in missing
                   if is_new_module_parameter(name, strict_missing_prefixes)]
        disallowed = [name for name in missing if name not in allowed]
        if disallowed:
            raise Refusal(
                f"{len(disallowed)} model parameters have no checkpoint source, e.g. "
                f"{disallowed[:3]}; only the new modules may be missing")

        model.load_state_dict(remapped, strict=False)
        model.to(dtype)
        model.new_module_parameters = allowed
        model.attention_tensors_mapped = attention_targets
        return model


def load_config(path: str | Path) -> LongRWKVConfig:
    """Read the released backbone's config into this package's config object."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LongRWKVConfig(
        hidden_size=payload["hidden_size"],
        num_hidden_layers=payload["num_hidden_layers"],
        vocab_size=payload["vocab_size"],
        head_dim=payload["head_dim"],
        intermediate_size=payload["intermediate_size"],
        hidden_act=payload["hidden_act"],
        decay_low_rank_dim=payload["decay_low_rank_dim"],
        a_low_rank_dim=payload["a_low_rank_dim"],
        gate_low_rank_dim=payload["gate_low_rank_dim"],
        v_low_rank_dim=payload["v_low_rank_dim"],
        norm_eps=payload["norm_eps"],
        norm_bias=payload.get("norm_bias", True),
        fuse_norm=payload.get("fuse_norm", False),
        norm_first=payload.get("norm_first", True),
        hidden_ratio=payload.get("hidden_ratio", 4.0),
    )
