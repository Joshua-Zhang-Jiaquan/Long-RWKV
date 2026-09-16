"""Typed latent plan slots for the BiRWKV denoiser (v4_plan sections 3, 4.1, 9/M0).

Two modules, and the split between them is the whole point:

* :class:`PlanEncoder` sees **clean** tokens and produces the training target
  ``Z``. It is target-derived, therefore *oracle-adjacent*.
* :class:`PlanPredictor` sees only the **corrupted canvas** — visible tokens
  plus mask ids, exactly what the denoiser itself gets — and produces ``Ẑ``.

``v4_plan`` prohibition **P1** says every headline number involving the latent
space must be computed with ``Ẑ``. Keeping these two classes separate, with the
predictor structurally unable to read a clean target, is how that prohibition is
enforced in code rather than in a reviewer's memory. (Same discipline as
``LACES-DLM``'s ``LatentPlanPrior.encode_context()``.)

Slot typing
-----------
Slots carry a learned type embedding for ``tau`` and a positional code for
``p``, so a ``[B, H, d_z]`` field is typed rather than an anonymous tensor —
the gap flagged as BUILD-NEW in ``v4_plan`` section 3. ``r`` (role) and ``u``
(uncertainty) are carried but only ``u`` is predicted for now; role becomes
meaningful once more than one macro-neuron exists (Plan-3).

Scope note (deliberate deviation from v4_plan section 9/M0, recorded rather
than silently taken): the plan lists ``L_StateWrite`` and ``L_Cycle`` among
M0's losses. Both are defined in terms of writing into the *persistent
recurrent state* ``S`` and reading it back. Arm A1 (FiLM) modulates the
per-layer activation stream, not ``S``, so neither loss is well-defined for it.
They activate with arm A2 (``NativeStateRelay``), where a state write actually
happens. M0 therefore trains ``L_align`` + ``L_keep`` only, and the Gate-D0
verdict must say so.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.nn import functional
from typing_extensions import override

from models.state_hijacking_dit_torch_types import TypedTorchModule

# Slot types, in the order v4_plan section 3.1 lists them.
SLOT_TYPES: tuple[str, ...] = (
    "global",
    "task",
    "plan",
    "work",
    "message",
    "observation",
    "multimodal",
)

_SLOTTED_NDIM = 3


def slots_to_positions(z_slots: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Broadcast ``[B, H, d_z]`` slots to ``[B, T, d_z]``, slot *h* covering its chunk.

    This is the readout the slot structure exists for. The encoder builds `Z` by
    mean-pooling ``T/H``-token chunks, so slot *h* summarises positions
    ``[h*T/H, (h+1)*T/H)``; conditioning position *i* on the slot covering *i*
    gives the denoiser *locally relevant* content instead of one whole-sequence
    average, which is what a per-position denoising decision can use.

    **Why this lives here rather than beside the conditioner:** it is pure torch,
    and `birwkv7_diffusion` cannot be imported without CUDA (fla's triton
    kernels). Keeping it in this module lets the CPU regression suite assert the
    readout is lossless — the check whose absence cost an entire co-train, when a
    mean-pooled readout silently discarded 93.2% of the slot field. See memory
    `latentnet-d0-dissociation`.

    A ``[B, d_z]`` input (already pooled, or a genuinely global vector) is
    broadcast unchanged to every position.
    """
    if z_slots.dim() != _SLOTTED_NDIM:
        return z_slots.unsqueeze(1).expand(-1, seq_len, -1)
    bsz, n_slots, d_z = z_slots.shape
    # repeat_interleave then trim/pad, so any (T, H) pair works without assuming
    # divisibility — the encoder right-pads, so the tail slot legitimately covers
    # fewer real positions.
    per = -(-seq_len // n_slots)  # ceil
    expanded = z_slots.repeat_interleave(per, dim=1)
    if expanded.shape[1] >= seq_len:
        return expanded[:, :seq_len, :]
    pad = seq_len - expanded.shape[1]
    return torch.cat((expanded, expanded[:, -1:, :].expand(bsz, pad, d_z)), dim=1)


class _SlotTrunk(TypedTorchModule):
    """Shared body: token ids -> per-slot pooled features.

    Chunk-pool then MLP, following ``FrozenTargetEncoder``'s shape (pool ->
    2x(Linear+LN+GELU) -> head). The embedding is this module's own and small
    on purpose: coupling to the denoiser's 65k-row embedding table would make
    every latent gradient touch the backbone.
    """

    def __init__(
        self,
        vocab_size: int,
        num_slots: int,
        d_z: int,
        embed_dim: int = 256,
        hidden: int = 512,
    ) -> None:
        """Build the embedding, pooling MLP, and typed slot heads."""
        super().__init__()
        self.num_slots: int = num_slots
        self.d_z: int = d_z
        self.embed: nn.Embedding = nn.Embedding(vocab_size, embed_dim)
        self.body: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.mu_head: nn.Linear = nn.Linear(hidden, d_z)
        # Per-slot LayerNorm on the CONTENT coordinate. This is what stops the
        # regression target from being dominated by a constant: after norming,
        # emitting a fixed vector no longer minimises the MSE, because the target
        # has zero mean and unit scale per slot by construction.
        self.content_norm: nn.LayerNorm = nn.LayerNorm(d_z)
        # tau: one learned vector per slot type; p: one per slot index.
        # NOT part of the regression target -- see `typed()` / `typing_codes()`.
        self.type_embed: nn.Embedding = nn.Embedding(len(SLOT_TYPES), d_z)
        self.pos_embed: nn.Embedding = nn.Embedding(num_slots, d_z)
        # Slot i gets type i for the first len(SLOT_TYPES) slots, then cycles
        # through the "plan"/"work" pair, which is where extra capacity helps.
        assign = [
            i if i < len(SLOT_TYPES) else (2 + (i % 2)) for i in range(num_slots)
        ]
        self.register_buffer("slot_type_ids", torch.tensor(assign, dtype=torch.long))

    def pooled(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Chunk-pool ``[B, T]`` ids into ``[B, num_slots, hidden]`` features."""
        h: torch.Tensor = self.embed(input_ids)
        bsz, seq, dim = h.shape
        n = self.num_slots
        # Right-pad to a multiple of num_slots so chunk sizes stay uniform.
        if seq % n != 0:
            pad = n - (seq % n)
            h = torch.cat((h, h.new_zeros(bsz, pad, dim)), dim=1)
            seq = seq + pad
        chunks = h.view(bsz, n, seq // n, dim).mean(dim=2)
        return cast("torch.Tensor", self.body(chunks))

    def typed(self, feats: torch.Tensor) -> torch.Tensor:
        """Project pooled features to typed slot coordinates ``[B, H, d_z]``.

        **Content and typing are deliberately kept apart.** An earlier version
        returned ``mu_head(feats) + type_embed + pos_embed``, and measurement
        showed that target was degenerate: ~80% of ‖Z‖ was the constant
        type+position term, and cosine(Z(30% masked), Z(clean)) was 0.9597 —
        barely different from 0.9507 at *100%* masking. A regression onto such a
        target is minimised by emitting a fixed per-slot vector, which is exactly
        what the co-train did (`z_mse` pinned at 3.75, `d0_delta` frozen at
        −0.0023). See memory `latentnet-d0-target-defect`.

        So the regression target is **content only**, per-slot LayerNorm'd so no
        constant can win on norm. Type and position identity are still available
        to downstream consumers via :meth:`typing_codes`, and are added back only
        where slot identity is actually needed (routing, packet headers) — never
        inside the quantity the predictor is asked to match.
        """
        v: torch.Tensor = self.mu_head(feats)
        return cast("torch.Tensor", self.content_norm(v))

    def typing_codes(self, device: torch.device) -> torch.Tensor:
        """Return the ``[1, H, d_z]`` type+position code, for downstream use only.

        Kept out of the regression target on purpose (see :meth:`typed`).
        """
        type_ids = cast("torch.Tensor", self.slot_type_ids).to(device)
        tau = self.type_embed(type_ids).unsqueeze(0)
        pos = self.pos_embed(torch.arange(self.num_slots, device=device)).unsqueeze(0)
        return tau + pos


class PlanEncoder(TypedTorchModule):
    """Clean tokens -> target latent field ``Z`` (oracle-adjacent; see module docs)."""

    if TYPE_CHECKING:
        __call__: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        vocab_size: int,
        num_slots: int = 8,
        d_z: int = 32,
        embed_dim: int = 256,
        hidden: int = 512,
    ) -> None:
        """Build the encoder trunk."""
        super().__init__()
        self.trunk: _SlotTrunk = _SlotTrunk(vocab_size, num_slots, d_z, embed_dim, hidden)
        self.d_z: int = d_z
        self.num_slots: int = num_slots

    @override
    def forward(self, clean_ids: torch.Tensor) -> torch.Tensor:
        """Return the target latent field ``[B, H, d_z]``."""
        return self.trunk.typed(self.trunk.pooled(clean_ids))


class PlanPredictor(TypedTorchModule):
    """Corrupted canvas -> predicted latent field ``Ẑ`` plus per-slot uncertainty ``u``.

    Structurally leakage-safe: ``forward`` takes the corrupted ids only. There
    is no code path by which a clean target reaches this module, which is what
    makes P1 an architectural guarantee instead of a convention.
    """

    if TYPE_CHECKING:
        __call__: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

    def __init__(
        self,
        vocab_size: int,
        num_slots: int = 8,
        d_z: int = 32,
        embed_dim: int = 256,
        hidden: int = 512,
    ) -> None:
        """Build the predictor trunk and the uncertainty head."""
        super().__init__()
        self.trunk: _SlotTrunk = _SlotTrunk(vocab_size, num_slots, d_z, embed_dim, hidden)
        self.u_head: nn.Linear = nn.Linear(hidden, 1)
        self.d_z: int = d_z
        self.num_slots: int = num_slots

    @override
    def forward(self, corrupted_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z_hat [B, H, d_z], u [B, H])``."""
        feats = self.trunk.pooled(corrupted_ids)
        z_hat = self.trunk.typed(feats)
        u = torch.sigmoid(self.u_head(feats)).squeeze(-1)
        return z_hat, u


def latent_align_loss(
    z_hat: torch.Tensor,
    z_target: torch.Tensor,
    u: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regression of ``Ẑ`` onto a detached ``Z`` target, optionally ``u``-weighted.

    The target is detached: the predictor chases the encoder, never the reverse.
    Letting gradients flow both ways invites the pair to collapse onto a
    constant, which would satisfy the loss and teach nothing.

    With ``u`` supplied this becomes a heteroscedastic objective — a slot the
    predictor declares uncertain is penalised less on the error term and more
    on the confidence term, so ``u`` learns to mean something instead of
    drifting.
    """
    target = z_target.detach()
    per_slot = (z_hat.float() - target.float()).pow(2).mean(dim=-1)  # [B, H]
    if u is None:
        loss = per_slot.mean()
        diag = {"z_align_mse": float(loss.detach())}
        return loss, diag
    precision = 1.0 - u.float().clamp(0.0, 0.99)
    loss = (precision * per_slot - torch.log(precision.clamp_min(1e-4))).mean()
    diag = {
        "z_align_mse": float(per_slot.mean().detach()),
        "z_u_mean": float(u.float().mean().detach()),
    }
    return loss, diag


def distribution_keep_loss(
    cond_logits: torch.Tensor,
    base_logits: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``KL[p_base || p_conditioned]`` over valid positions (v4_plan section 10.5).

    The guard that keeps the section-1.1 evidence intact while conditioning
    trains: the conditioned model may sharpen its predictions but must not walk
    away from the language distribution the frozen lineage established.
    ``base_logits`` must come from a no-grad forward with ``z_slots=None``.
    """
    base_logp = functional.log_softmax(base_logits.float(), dim=-1).detach()
    cond_logp = functional.log_softmax(cond_logits.float(), dim=-1)
    kl = (base_logp.exp() * (base_logp - cond_logp)).sum(dim=-1)  # [B, T]
    denom = valid.sum().clamp_min(1)
    loss = (kl * valid).sum() / denom
    return loss, {"keep_kl": float(loss.detach())}


def derange(n: int, device: torch.device, gen: torch.Generator | None = None) -> torch.Tensor:
    """A permutation of ``range(n)`` with no fixed points (``n >= 2``).

    Shared with ``eval.capability.d0_eval._intervene`` so the training-time
    "wrong plan" and the eval-time ``shuffle`` arm mean the same thing. If they
    diverged, a contrastive loss could be optimising a different comparison than
    Gate D0.2 measures -- the exact class of mismatch that makes a gate number
    unfalsifiable.
    """
    if n < 2:
        return torch.arange(n, device=device)
    perm = torch.randperm(n, device=device, generator=gen)
    for i in range(n):
        if int(perm[i]) == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    return perm


def plan_contrastive_loss(
    cond_logits: torch.Tensor,
    wrong_logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reward the *own* plan over a *wrong* plan on the masked positions.

    Gate D0.2 asks a specific question: does swapping in another document's ``Z``
    make the prediction worse? The gain sweep measured that quantity
    (``s = shuffle - predz``) at **+0.0006 .. +0.0018 nats** while the generic
    cost of conditioning at all (``h = shuffle - uncond``) sat ~9x higher, and
    the two scaled apart (``h ~ g^0.70`` vs ``s ~ g^0.63``) so no amplitude makes
    amplification pay.

    The diagnosis behind this loss: nothing in the previous objective put gradient
    on ``s``. ``L_align`` shapes the *predictor* (``Zhat`` toward the target ``Z``),
    and the only signal telling the *conditioner* what to DO with a plan arrived
    through ``mask_ce``, diluted across ~42M FiLM weights -- which the conditioner
    can satisfy by reacting to ``Z``'s magnitude and ignoring its content.

    This term optimises ``s`` directly as a hinge on the per-token NLL difference:

        L = mean_masked( relu( margin + nll_own - nll_wrong ) )

    It is zero once the own plan already beats the wrong plan by ``margin`` nats,
    so it applies pressure only where the discrimination is missing rather than
    pushing the two apart without limit.

    ``wrong_logits`` must come from a forward whose ``z_slots`` were deranged
    (see :func:`derange`); ``mask`` selects the positions scored by the masked
    diffusion loss, since those are the only ones D0 measures.
    """
    # Gather the scored positions BEFORE the fp32 upcast. At B=8, T=4096,
    # V=65536 a full-tensor `.float()` is 8.6 GiB per branch -- 17 GiB for the two
    # here, on top of the extra wrong-plan forward's own graph. Selecting the ~30%
    # masked positions first cuts that by ~3x and makes the term affordable
    # alongside gradient checkpointing.
    vocab = cond_logits.shape[-1]
    sel = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        zero = cond_logits.sum() * 0.0
        return zero, {"z_contrast_gap": 0.0, "z_contrast_loss": 0.0}
    tgt = target_ids.reshape(-1)[sel]
    nll_own = functional.cross_entropy(
        cond_logits.reshape(-1, vocab)[sel].float(), tgt, reduction="none")
    nll_wrong = functional.cross_entropy(
        wrong_logits.reshape(-1, vocab)[sel].float(), tgt, reduction="none")
    gap = nll_wrong - nll_own  # >0 means the own plan is better: what we want
    loss = functional.relu(margin - gap).mean()
    with torch.no_grad():
        diag = {
            "z_contrast_gap": float(gap.mean().detach()),
            "z_contrast_loss": float(loss.detach()),
        }
    return loss, diag


def make_state_cache() -> object:
    """Return an empty fla ``Cache`` for recurrent-state capture or injection.

    Imported lazily: ``fla.models.utils`` pulls in the Triton kernels, so a
    module-level import would make this file un-importable on a CPU-only box and
    break the whole local test suite.
    """
    from fla.models.utils import Cache  # noqa: PLC0415

    return Cache()


def read_state_cache(cache: object, num_layers: int) -> list[torch.Tensor]:
    """Extract per-layer ``recurrent_state`` tensors from a populated fla cache.

    Returns a list of ``num_layers`` tensors, each ``[B, heads, head_dim, head_dim]``
    (``[B, 40, 64, 64]`` at the 2.9B geometry). Raises if any layer is missing --
    a silently short list would make arm C inject into only part of the stack and
    look like a weak result rather than a broken one.
    """
    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) < num_layers:
        msg = (
            f"state cache has {0 if layers is None else len(layers)} layers, "
            f"expected {num_layers} -- was use_cache=True actually passed?"
        )
        raise ValueError(msg)
    out: list[torch.Tensor] = []
    for idx in range(num_layers):
        state = getattr(layers[idx], "state", None)
        tensor = None if state is None else state.get("recurrent_state")
        if not isinstance(tensor, torch.Tensor):
            msg = f"layer {idx} has no recurrent_state; cache was not populated"
            raise ValueError(msg)
        out.append(tensor)
    return out


def state_norm_features(states: Sequence[torch.Tensor]) -> dict[str, float]:
    """Cheap liveness diagnostics for a captured/injected recurrent state.

    The A3 arm trained for hundreds of steps against a conditioning path that had
    provably zero effect, and only a dedicated response test caught it. So arm C
    logs the state's own magnitude every step: a state that is all-zero, constant
    across the batch, or non-finite means the capture silently failed, and the
    resulting "no benefit" would be a plumbing artifact rather than a result.
    """
    if not states:
        return {}
    flat = torch.stack([s.detach().float().flatten(1) for s in states], dim=0)
    per_row = flat.norm(dim=-1)                       # [L, B]
    batch_spread = per_row.std(dim=-1).mean() if per_row.shape[-1] > 1 else torch.zeros(())
    return {
        "state_absmean": float(flat.abs().mean()),
        "state_norm": float(per_row.mean()),
        # If this is ~0 the batch rows share one state => the shuffle control would
        # be vacuous, exactly the failure the D0.2 arms exist to detect.
        "state_batch_spread": float(batch_spread),
    }
