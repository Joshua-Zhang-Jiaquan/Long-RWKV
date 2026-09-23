"""Algorithm 1: monotone token-feedback denoising that selects *before* it samples.

The ordering is the whole point.  The earlier repository sampler sampled
candidate values first, ranked positions by the *sampled* token's confidence, and
then took the top-k.  The appendix rules that order out with a finite
counterexample: for two independent Bernoulli(0.9) coordinates, sample-then-select
gives ``P(11) = 0.99 * 0.9 = 0.891`` against a true ``0.81``, a total-variation
error of 0.081, because selecting on one's own draw makes the first chosen value
one whenever *either* candidate was one.

So here, for every call ``k``:

1. the noise coordinate is recomputed from the **current** canvas with the same
   ``block_t_from_canvas`` that training uses;
2. one forward pass gathers representations at the unfilled targets;
3. ``b_k = ceil(n_k / (K - k))`` positions are **selected** -- from probabilities
   only, with a fixed tie rule -- and nothing is drawn before this selection;
4. values are sampled at those selected positions and written.

``b_k`` guarantees at least one fill per call and all remaining on the last, so
there are exactly ``K = min(K_req, M)`` useful calls and no empty-stage calls.
``K_req`` is what was requested; the *realised* K is what the tables report.

Counters are exact by construction rather than estimated: ``calls`` counts full
model invocations, ``head_positions`` sums the gathered positions the vocabulary
head was applied to, ``token_visits`` sums canvas tokens fed back through the
network, and ``executed_blocks`` comes from the model's own declared block count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .conditioning import vis_codes
from .corruption import DEFAULT_BLOCK_SIZE, block_t_from_canvas
from .ids import FORBIDDEN_SAMPLING_IDS, MASK_ID
from .refusal import Refusal

MIN_ENTROPY = "min_entropy"
FIXED_ORDER = "fixed_order"
POLICIES = (MIN_ENTROPY, FIXED_ORDER)

MONOTONE = "monotone"
ARGMAX = "argmax"
REMASK = "remask"
MODES = (MONOTONE, ARGMAX, REMASK)


@dataclass
class SamplerConfig:
    """One declared operating point.  Everything here is recorded on the result."""

    k_req: int
    r: int = 1
    tau: float = 1.0
    policy: str = MIN_ENTROPY
    mode: str = MONOTONE
    block_size: int = DEFAULT_BLOCK_SIZE
    remask_fraction: float = 0.0
    vocab_restriction: tuple[int, ...] = FORBIDDEN_SAMPLING_IDS

    def __post_init__(self) -> None:
        if self.k_req < 1:
            raise Refusal(f"k_req must be >= 1, got {self.k_req}")
        if self.r < 1:
            raise Refusal(f"R must be >= 1, got {self.r}")
        if self.tau <= 0:
            raise Refusal(f"tau must be positive, got {self.tau}")
        if self.policy not in POLICIES:
            raise Refusal(f"unknown policy {self.policy!r}; expected one of {POLICIES}")
        if self.mode not in MODES:
            raise Refusal(f"unknown mode {self.mode!r}; expected one of {MODES}")
        if not 0.0 <= self.remask_fraction < 1.0:
            raise Refusal(
                f"remask_fraction must be in [0, 1), got {self.remask_fraction}")

    @property
    def policy_id(self) -> str:
        """Identifies the exact transition law, so no two laws share a row label."""
        return f"{self.mode}/{self.policy}/tau={self.tau:g}/R={self.r}"


@dataclass
class SampleResult:
    """The canvas plus every counter the paper's tables require."""

    canvas: torch.Tensor
    calls: int
    selected_positions: list[torch.Tensor]
    executed_blocks: int
    head_positions: int
    token_visits: int
    per_call: list[dict] = field(default_factory=list)
    reopened: list[torch.Tensor] = field(default_factory=list)
    policy_id: str = ""
    vocab_restriction: tuple[int, ...] = ()
    k_req: int = 0
    realized_k: int = 0
    target_capacity: int = 0
    failures: list[str] = field(default_factory=list)


ModelCall = Callable[..., torch.Tensor]
FeedbackHook = Callable[[torch.Tensor, int, dict], torch.Tensor]


def select_positions(unfilled: torch.Tensor,
                     entropy: torch.Tensor,
                     b_k: int,
                     policy: str) -> torch.Tensor:
    """Choose ``b_k`` rows of ``entropy`` to fill.  Returns row indices, ascending.

    ``unfilled`` is the ascending flat position list the logits rows correspond to.

    Both policies are deterministic functions of the *current probability vectors*
    alone.  The minimum-entropy order is a stable sort, and because ``unfilled`` is
    ascending, equal entropies resolve to the lower position -- a fixed tie rule,
    which is what the algorithm requires.
    """
    if b_k < 1:
        raise Refusal(f"b_k must be >= 1, got {b_k}")
    n = int(entropy.shape[0])
    if b_k > n:
        b_k = n
    if policy == FIXED_ORDER:
        return torch.arange(b_k, device=entropy.device)
    if policy == MIN_ENTROPY:
        order = torch.argsort(entropy, stable=True)
        return torch.sort(order[:b_k]).values
    if policy == "max_entropy":  # explicit alternative, same selection-before-sample order
        order = torch.argsort(entropy, stable=True, descending=True)
        return torch.sort(order[:b_k]).values
    raise Refusal(f"unknown policy {policy!r}; expected one of {POLICIES}")


def autoregressive_config(capacity: int, *, tau: float = 1.0,
                          block_size: int = DEFAULT_BLOCK_SIZE) -> SamplerConfig:
    """The operating point that makes this sampler decode left to right.

    No new sampler is needed, and that is worth stating plainly because the
    package briefly had no autoregressive decode path at all while claiming an
    autoregressive baseline.  With ``k_req = M`` and ``policy=fixed_order`` the
    schedule ``b_k = ceil(n_k / (K - k))`` is ``[1, 1, ..., 1]``, and
    ``select_positions`` under ``fixed_order`` takes the first row of an ascending
    ``unfilled`` list: one position per call, in position order, each conditioned
    on every token committed before it.  On a ``force_forward`` arm that *is*
    left-to-right decoding.

    What it is not is the default point.  At ``k_req = 1`` the same code commits
    the whole answer region in a single parallel call from an all-MASK region, so
    an "autoregressive reference" would predict tokens 2..n from MASK rather than
    from its own prefix.  Measured answer widths on the real trie: 3 tokens for
    ``associative_recall`` and ``latest_write``, and 32 to 512 for the LongBench
    tasks -- so the manuscript's single-token escape clause covers exactly one of
    the families it was read as covering.
    """
    if capacity < 1:
        raise Refusal(f"capacity must be >= 1 for an AR schedule, got {capacity}")
    return SamplerConfig(k_req=capacity, r=1, tau=tau, policy=FIXED_ORDER,
                         mode=MONOTONE, block_size=block_size)


def require_autoregressive_schedule(config: SamplerConfig, capacity: int) -> None:
    """Refuse an operating point that would decode an AR arm in parallel.

    Called from :func:`denoise` whenever the bound model's arm declares the
    autoregressive objective, so the check sits at the point of use and no caller
    can reach the sampler without passing it.
    """
    if config.policy != FIXED_ORDER:
        raise Refusal(
            f"an autoregressive arm decodes in position order, so policy must be "
            f"{FIXED_ORDER!r}, not {config.policy!r}: an entropy order commits a "
            f"later position before an earlier one, and the earlier one is then "
            f"predicted from a suffix the model never had at training time")
    if config.k_req < capacity:
        raise Refusal(
            f"an autoregressive arm at k_req={config.k_req} with M={capacity} "
            f"target positions fills ceil({capacity}/{config.k_req}) positions per "
            f"call from a masked region, which is parallel decoding under an "
            f"autoregressive label. Pass k_req >= M (see autoregressive_config).")
    if config.mode != MONOTONE:
        raise Refusal(
            f"an autoregressive arm never revises a committed token, so mode must "
            f"be {MONOTONE!r}, not {config.mode!r}")
    if config.r != 1:
        raise Refusal(
            f"an autoregressive arm has no hidden loop; R must be 1, got {config.r}")


def _entropy_from_logits(logits: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax and its entropy, in fp32, with non-finite rows refused."""
    scaled = logits.float() / tau
    probs = torch.softmax(scaled, dim=-1)
    log_probs = torch.log(probs.clamp_min(torch.finfo(torch.float32).tiny))
    entropy = -(probs * log_probs).sum(dim=-1)
    if not torch.isfinite(probs).all():
        raise Refusal("softmax produced non-finite probabilities")
    return probs, entropy


def denoise(model: ModelCall,
            canvas: torch.Tensor,
            target_mask: torch.Tensor,
            attention_mask: torch.Tensor,
            config: SamplerConfig,
            doc_starts: torch.Tensor | None = None,
            feedback_hook: FeedbackHook | None = None,
            generator: torch.Generator | None = None,
            mask_id: int = MASK_ID,
            ) -> SampleResult:
    """Run Algorithm 1 on one request.  ``canvas`` is ``[1, T]``.

    A single request, not a batch: the varlen kernel contract already flattens to
    ``B=1``, and ``K`` is defined per target row, so batching rows with different
    capacities would make ``K = min(K_req, M)`` ambiguous.
    """
    if canvas.dim() != 2 or canvas.shape[0] != 1:
        raise Refusal(f"canvas must be [1, T], got {tuple(canvas.shape)}")
    if target_mask.shape != canvas.shape or attention_mask.shape != canvas.shape:
        raise Refusal("target_mask and attention_mask must share canvas's shape")

    work = canvas.clone()
    live = attention_mask.bool()
    target = target_mask.bool() & live
    eligible = target
    immutable = live & ~target

    capacity = int(target.sum().item())
    if capacity < 1:
        raise Refusal("target capacity M is zero; nothing to fill")
    # **The arm decides whether this operating point is legal for it.**  Read off
    # the model rather than passed in, because the defect this guards against was
    # exactly a caller that did not know it mattered: ``Binding`` carries no arm,
    # so every arm was scored at the registry's single (k_req=1, r=1) point, and
    # an autoregressive baseline committed its whole answer region in one parallel
    # call from an all-MASK region.  A model with no ``spec`` (the rigged stand-ins
    # the sampler tests use) is unaffected.
    spec = getattr(model, "spec", None)
    if spec is not None and getattr(spec, "autoregressive", False):
        require_autoregressive_schedule(config, capacity)
    realized_k = min(config.k_req, capacity)

    forbidden = torch.tensor(config.vocab_restriction, dtype=torch.long,
                             device=canvas.device)
    if int(forbidden.min()) < 0:
        raise Refusal("vocab restriction contains a negative id")
    if mask_id not in config.vocab_restriction:
        raise Refusal(
            "mask_id must be in vocab_restriction: a sampled token that equals the "
            "mask id would silently shrink the target capacity")

    result = SampleResult(
        canvas=work.clone(), calls=0, selected_positions=[], executed_blocks=0,
        head_positions=0, token_visits=0, policy_id=config.policy_id,
        vocab_restriction=config.vocab_restriction, k_req=config.k_req,
        realized_k=realized_k, target_capacity=capacity,
    )
    blocks_per_call = int(model.blocks_per_call(config.r)) if hasattr(
        model, "blocks_per_call") else 0
    n_positions = int(canvas.shape[1])

    for k in range(realized_k):
        model_view = work.clone() if (k == 0 or feedback_hook is None) else \
            feedback_hook(work, k, {
                "config": config, "call": k, "mask_id": mask_id,
                "eligible": eligible, "immutable": immutable,
            }).clone()

        if config.mode == REMASK and k > 0 and config.remask_fraction > 0.0:
            reopened = _reopen_positions(
                model_view, eligible, immutable, config, model, generator,
                mask_id)
            result.reopened.append(reopened)
            if reopened.numel():
                model_view[0, reopened] = mask_id
                work[0, reopened] = mask_id

        is_mask = model_view.eq(mask_id)
        # index row 0 explicitly: nonzero() on a [1, T] tensor returns (row, col)
        # *pairs*, and flattening them interleaves rows with columns -- which
        # reported twice as many unfilled positions as existed and corrupted the
        # b_k schedule.  The canvas is [1, T] by contract, so columns are the
        # positions.
        unfilled = (eligible & is_mask)[0].nonzero(as_tuple=False).flatten()
        n_k = int(unfilled.numel())
        if n_k == 0:
            result.failures.append(f"call {k}: no unfilled eligible positions")
            break

        block_t = block_t_from_canvas(eligible, is_mask, config.block_size)
        codes = vis_codes(model_view, live, target, mask_id)
        # Forward the attention mask: without it the model treats trailing padding
        # as live tokens and gives the request a different segment layout than
        # training would, which is a silent change to what the canvas means.
        #
        # ``block_size`` goes with it, for the same class of reason.  The canvas
        # width here is ``len(prompt) + answer_length`` and is only a multiple of
        # 256 by luck: at a 3072-token prompt budget the LongBench widths are 3104,
        # 3136, 3200, and ``phi_features`` used to re-derive the grid as
        # ``T // n_blocks``, which refused 79% of widths outright and silently
        # shifted the noise coordinate on most of the rest.  Passing the config's
        # own ``block_size`` -- the same value ``block_t_from_canvas`` above was
        # given -- makes producer and consumer one decision instead of two.
        #
        # Inference only: every caller of the sampler scores or counts, none
        # backprops, so the graph a grad-enabled call would retain is pure
        # memory -- one full activation set per call, which at the long-context
        # lengths is the difference between fitting and not.
        with torch.no_grad():
            logits = model(model_view, block_t=block_t, codes=codes,
                           gather_idx=unfilled, R=config.r, doc_starts=doc_starts,
                           attention_mask=live, block_size=config.block_size)
        if logits.dim() != 2 or logits.shape[0] != n_k:
            raise Refusal(
                f"call {k}: model returned {tuple(logits.shape)} for {n_k} gathered "
                f"positions; the head must be applied to gathered rows only")

        masked_logits = logits.float().clone()
        masked_logits[:, forbidden] = float("-inf")
        probs, entropy = _entropy_from_logits(masked_logits, config.tau)

        b_k = -(-n_k // (realized_k - k))  # ceil(n_k / (K - k)), >= 1
        rows = select_positions(unfilled, entropy, b_k, config.policy)

        if config.mode == ARGMAX:
            values = probs[rows].argmax(dim=-1)
        else:
            values = torch.multinomial(
                probs[rows], 1, generator=generator).squeeze(-1)
        chosen = unfilled[rows]

        before = work[0, chosen].clone()
        work[0, chosen] = values.to(work.dtype)

        # The immutability guarantee is structural, but asserted: a policy that
        # rewrote a prompt token would invalidate every comparison built on it.
        if immutable.any():
            touched = work[0][immutable[0]] != canvas[0][immutable[0]]
            if bool(touched.any()):
                raise Refusal(
                    f"call {k}: {int(touched.sum())} immutable positions changed")

        result.calls += 1
        result.executed_blocks += blocks_per_call
        result.head_positions += n_k
        result.token_visits += n_positions
        result.selected_positions.append(chosen.clone())
        result.per_call.append({
            "call": k,
            "n_unfilled": n_k,
            "b_k": int(rows.numel()),
            "positions": chosen.tolist(),
            "filled_values": [int(v) for v in values.tolist()],
            "entropy_at_selection": entropy[rows].tolist(),
            "changed_from": [int(v) for v in before.tolist()],
            "mean_entropy_unfilled": float(entropy.mean().item()),
        })

    remaining = int((eligible & work.eq(mask_id)).sum().item())
    if remaining:
        result.failures.append(
            f"{remaining} eligible targets remain masked after {result.calls} calls")
    result.canvas = work.clone()
    return result


def _reopen_positions(view: torch.Tensor,
                      eligible: torch.Tensor,
                      immutable: torch.Tensor,
                      config: SamplerConfig,
                      model: ModelCall,
                      generator: torch.Generator | None,
                      mask_id: int) -> torch.Tensor:
    """Reopen the lowest-confidence filled targets, as the revisable mode requires.

    Reopening is logged as part of ``U_k`` and leaves the output capacity
    unchanged: the paper's requirement is that a revisable policy may not quietly
    buy itself extra answer positions.  ``mask_id`` is the caller's own id: a
    probe with an in-range sentinel mask must reopen against that sentinel, or
    every still-masked position would count as filled and a reopen could be spent
    on a position that was never written.
    """
    filled = eligible & ~view.eq(mask_id)
    n_filled = int(filled.sum().item())
    if n_filled == 0:
        return view.new_zeros(0, dtype=torch.long)
    take = max(1, int(config.remask_fraction * n_filled))
    take = min(take, n_filled)
    positions = filled[0].nonzero(as_tuple=False).flatten()
    block_t = block_t_from_canvas(eligible, view.eq(mask_id), config.block_size)
    codes = vis_codes(view, eligible | immutable, eligible, mask_id)
    with torch.no_grad():
        logits = model(view, block_t=block_t, codes=codes, gather_idx=positions,
                       R=config.r, doc_starts=None,
                       attention_mask=eligible | immutable,
                       block_size=config.block_size).float()
    probs = torch.softmax(logits / config.tau, dim=-1)
    current = probs.gather(1, view[0, positions].unsqueeze(-1).long()).squeeze(-1)
    order = torch.argsort(current, stable=True)
    return torch.sort(positions[order[:take]]).values


def shuffle_feedback(seed: int, generator: torch.Generator | None = None) -> FeedbackHook:
    """M1: permute generated values among filled eligible targets.

    The visibility mask, condition and requested schedule are preserved, and the
    permutation is seeded so it can be paired with the genuine run's seed.  This is
    an intervention on feedback *content* -- not a model with normal inference
    semantics -- and the caller must report it as such.
    """
    def hook(view: torch.Tensor, k: int, state: dict) -> torch.Tensor:
        # The mask id and the eligible region come from the sampler that called
        # this hook.  Reading the module-level MASK_ID instead would be wrong for
        # any caller with a different vocabulary, and would treat immutable
        # observed tokens as filled -- permuting the prompt.
        mask_id = state["mask_id"]
        eligible = state["eligible"]
        filled = eligible & ~view.eq(mask_id)
        positions = filled[0].nonzero(as_tuple=False).flatten()
        if positions.numel() < 2:
            return view
        gen = generator if generator is not None else \
            torch.Generator(device=view.device).manual_seed(seed + k)
        perm = torch.randperm(positions.numel(), generator=gen, device=view.device)
        out = view.clone()
        out[0, positions] = view[0, positions[perm]]
        return out
    return hook


def restore_feedback(history: list[torch.Tensor]) -> FeedbackHook:
    """M2: show an earlier canvas, so newly written positions revert to masks.

    That is a *visibility* change, not a content-only one, so ``per_call`` records
    the number of positions whose mask state moved and the paper's
    content-only-comparison clause does not apply to this row.
    """
    def hook(view: torch.Tensor, k: int, state: dict) -> torch.Tensor:
        prior = history[-1] if history else view
        mask_id = state["mask_id"]
        # Recompute on the intervened input; the caller records the difference.
        state.setdefault("mask_changes", []).append(
            int((prior.eq(mask_id) != view.eq(mask_id)).sum().item()))
        return prior
    return hook
