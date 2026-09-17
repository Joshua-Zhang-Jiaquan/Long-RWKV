"""Influence-decay bounds for the RWKV-7 recurrent state.

This module implements the *numerical* half of the long-context theorem: the
objects the theorem is stated about, the bounds it asserts, and the checks that
decide whether a bound actually certifies anything on a given model.  The
proofs live in ``DAN/nonlatent_iclr/theory.md``; this file is what turns them
into something executable, so that a claim of the form "this layer contracts at
rate kappa" is a measurement rather than a reading of the paper.

The recurrence, per head, one layer, one direction (``fla/layers/rwkv7.py``,
reference form in ``scale/scratch_1b/rwkv_models.py:rwkv7_step``)::

    S_t = M_t S_{t-1} + k_t v_t^T          M_t = D_t - a_t k_hat_t k_hat_t^T
    o_t = (r_t / sqrt(K))^T S_t            D_t = diag(exp(w_t))

Three facts this module exists to make checkable, because the whole theorem
rests on them and each was wrong in an earlier draft of the plan:

1. **The rank-1 term is subtracted from the UNDECAYED state.**  The kernel
   applies the delta correction to ``S`` and the decay to the diagonal, giving
   ``M = D - a k k^T`` -- *not* ``D (I - a k k^T)``.  The distinction is not
   cosmetic: the first is symmetric, the second is not, and only the first has
   the contraction property below.  ``test_the_two_factorisations_differ``
   pins it, and ``closed_form_matches_reference`` checks the whole closed form
   against the reference step.
2. **M_t is symmetric, hence a strict contraction for EVERY input.**  Weyl's
   inequality gives ``lambda_max(M) <= lambda_max(D) = max_d exp(w_d)`` (the
   subtracted term is PSD) and ``lambda_min(M) >= lambda_min(D) - a >=
   e^{-c} - 1``.  Since ``max_d exp(w_d) > e^{-c} > 1 - e^{-c}``, the spectral
   norm is bounded by ``max_d exp(w_d) < 1``.  No hypothesis on trained weights
   is needed -- contraction is a property of the *parameterization*.
3. **What is NOT free is how close that bound is to 1.**  ``exp(w)`` is in
   ``(e^{-c}, 1)`` *open*, so the model can make a channel decay arbitrarily
   slowly, and the memory horizon depends on the trained distribution of ``w``,
   not on the bound.  ``require_contraction`` exists so a caller that wants a
   horizon cannot silently get a certificate from a trivially-true inequality.

Nothing here claims the model *retains* information: an influence bound is an
upper bound on how far a write can reach, and the plan's guardrails forbid
reading it as retention or as superiority over attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: Magnitude of the rwkv7 log-decay: ``w = -DECAY_SCALE * sigmoid(...)``.
#: ``-exp(-1/2)``; pinned identically in ``scale/scratch_1b/rwkv_config.py``
#: and in ``fla/layers/rwkv7.py``.
DECAY_SCALE = 0.6065306597126334

#: Infimum of the per-channel decay ``exp(w)``, approached as ``sigmoid -> 1``.
#: Not attained: ``exp(w)`` is a strictly increasing function of a bounded
#: sigmoid, so the range is the OPEN interval ``(MIN_DECAY, 1)``.
MIN_DECAY = math.exp(-DECAY_SCALE)

#: Weyl floor on the transition's most negative eigenvalue: ``e^{-c} - 1``.
#: Since this is below ``MIN_DECAY`` in magnitude, the spectral norm of a
#: transition is governed by its top eigenvalue, i.e. by ``max_d exp(w_d)``.
WEYL_FLOOR = MIN_DECAY - 1.0

#: Slack for the decay interval.  Over the reals ``sigmoid`` maps to the OPEN
#: interval, so ``exp(w)`` lies strictly in ``(e^{-c}, 1)``.  In floating point
#: ``sigmoid`` saturates at both ends: a large negative logit rounds to exactly
#: ``0.0`` (decay exactly ``1.0``) and a large positive one to exactly ``1.0``
#: (decay exactly ``MIN_DECAY``).  The bounds only require
#: ``max_d exp(w_d) <= 1``, so admitting the closed endpoints is correct; a
#: decay of exactly 1 simply makes ``contraction`` False, which is the honest
#: report rather than a bug.  Rejecting the endpoints would instead refuse
#: legitimate saturating inputs.
_DECAY_SLACK = 1e-12


class BoundRefusal(ValueError):
    """The bound does not certify the requested quantity.

    Raised rather than returning a bound. The research plan's failure QA is
    explicit that ``q >= 1`` must *expose* the limitation instead of receiving
    a certificate, and a function that quietly returned ``inf`` or a negative
    horizon would do the opposite.
    """


@dataclass(frozen=True, slots=True)
class TransitionBounds:
    """What can be asserted about one per-token transition ``M = D - a k k^T``."""

    #: ``max_d exp(w_d)`` -- the Lemma-1 spectral-norm bound. Strictly < 1.
    spectral_norm_upper: float
    #: Same quantity: Weyl gives ``lambda_max(M) <= lambda_max(D)``.
    lambda_max_upper: float
    #: ``lambda_min(D) - a`` -- Weyl's lower bound on the smallest eigenvalue.
    lambda_min_lower: float

    @property
    def contraction(self) -> bool:
        """Is the transition a contraction with a bound strictly below 1?"""
        return self.spectral_norm_upper < 1.0

    @property
    def horizon_half_life(self) -> float:
        """Half-life in tokens under this bound, or refuse if it certifies nothing."""
        return half_life(self.spectral_norm_upper)


def _require_decay_in_range(decay: torch.Tensor) -> None:
    """The decay must lie in ``[e^{-c}, 1]``; outside it the parameterization is not this model's."""
    if bool((decay < MIN_DECAY - _DECAY_SLACK).any()) or bool((decay > 1.0 + _DECAY_SLACK).any()):
        msg = (
            f"decay left the interval [e^{-1/2}, 1] = "
            f"[{MIN_DECAY:.6f}, 1]; the rwkv7 parameterization cannot produce "
            f"this and every bound in this module assumes it"
        )
        raise BoundRefusal(msg)


def decay_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """The per-channel decay ``exp(w)`` from the pre-sigmoid decay logits.

    ``w = -DECAY_SCALE * sigmoid(logits)``, so the result lies in
    ``(e^{-c}, 1)`` over the reals and ``[e^{-c}, 1]`` in floating point (see
    ``_DECAY_SLACK``).  The range check is a real assertion about the
    parameterization, not a tolerance: a decay above 1 would make every bound
    here false.
    """
    decay = torch.exp(-DECAY_SCALE * torch.sigmoid(logits))
    _require_decay_in_range(decay)
    return decay


def transition(w: torch.Tensor, a: float | torch.Tensor,
               khat: torch.Tensor) -> torch.Tensor:
    """The per-token state transition ``M = D - a k_hat k_hat^T``.

    ``w`` are the log-decays ``(K,)``, ``a`` the delta-rule gate in ``(0, 1)``,
    ``khat`` the L2-normalised write key ``(K,)``.  The result is symmetric by
    construction; ``test_transition_is_symmetric`` asserts that rather than
    trusting it.
    """
    if w.dim() != 1 or khat.dim() != 1 or w.shape != khat.shape:
        msg = f"w and khat must be matching 1-D vectors, got {tuple(w.shape)} and {tuple(khat.shape)}"
        raise BoundRefusal(msg)
    norm = float(torch.linalg.vector_norm(khat))
    if abs(norm - 1.0) > 1e-5:
        msg = f"khat must be unit-norm, got {norm:.6f}"
        raise BoundRefusal(msg)
    a_value = float(a)
    if not 0.0 <= a_value <= 1.0:
        # a = sigmoid(...) is in the OPEN interval; 1.0 is allowed here because
        # the bound is valid at the boundary and a fitted a may round to it.
        msg = f"the delta gate a must lie in [0, 1], got {a_value}"
        raise BoundRefusal(msg)
    decay = torch.exp(w)
    _require_decay_in_range(decay)
    return torch.diag(decay) - a_value * torch.outer(khat, khat)


def transition_bounds(w: torch.Tensor, a: float | torch.Tensor) -> TransitionBounds:
    """The Lemma-1 bounds for one transition, from ``w`` and ``a`` alone.

    Deliberately does not need ``khat``: the whole point of the Weyl argument is
    that the bound holds for *every* unit write key, so a caller cannot obtain a
    better certificate by choosing one.
    """
    decay = torch.exp(w)
    return TransitionBounds(
        spectral_norm_upper=float(decay.max()),
        lambda_max_upper=float(decay.max()),
        lambda_min_lower=float(decay.min()) - float(a),
    )


def per_step_bound(w: torch.Tensor) -> float:
    """``max_d exp(w_d)`` -- the contraction factor at one token."""
    return float(torch.exp(w).max())


def propagator(ws: torch.Tensor, a: torch.Tensor,
               khats: torch.Tensor) -> torch.Tensor:
    """``P = M_t M_{t-1} ... M_{tau+1}`` for the given ordered slice.

    The arrays are ordered EARLIEST-FIRST: ``ws[0]`` is the transition at
    ``tau + 1`` and the last entry is the transition at ``t``, so the product is
    accumulated right-to-left.  Passing one token's worth of rows gives that
    single transition; an empty slice is refused, because the identity is the
    answer for "no elapsed tokens" and a caller asking for a product almost
    certainly means at least one step.
    """
    if ws.dim() != 2 or khats.dim() != 2:
        msg = f"ws and khats must be (T, K), got {tuple(ws.shape)} and {tuple(khats.shape)}"
        raise BoundRefusal(msg)
    if ws.shape != khats.shape or a.shape != ws.shape[:1]:
        msg = (
            f"ws {tuple(ws.shape)}, khats {tuple(khats.shape)} and a "
            f"{tuple(a.shape)} are not aligned over time"
        )
        raise BoundRefusal(msg)
    if ws.shape[0] == 0:
        msg = "an empty transition slice has no product to bound; pass at least one step"
        raise BoundRefusal(msg)
    result: torch.Tensor | None = None
    for index in range(ws.shape[0]):
        step = transition(ws[index], a[index], khats[index])
        result = step if result is None else result @ step
    assert result is not None  # noqa: S101 - unreachable given the length check
    return result


def influence_bound(ws: torch.Tensor) -> float:
    """``prod_s max_d exp(w_{s,d})`` -- the Proposition-1 influence bound.

    Submultiplicativity of the spectral norm, applied to Lemma 1 at each step.
    This is what bounds how much a write at ``tau`` can still contribute at
    ``t``.
    """
    if ws.dim() != 2 or ws.shape[0] == 0:
        msg = f"ws must be a non-empty (T, K) slice, got {tuple(ws.shape)}"
        raise BoundRefusal(msg)
    return float(torch.exp(ws).max(dim=1).values.prod())


def require_contraction(kappa: float) -> float:
    """Return ``kappa`` if it certifies a contraction, else refuse.

    The failure-QA gate.  ``kappa >= 1`` means the bound says nothing about
    decay, and the honest response is to report the limitation -- not to return
    an infinite horizon or a bound whose geometric series diverges.
    """
    if not math.isfinite(kappa):
        msg = f"kappa must be finite, got {kappa!r}"
        raise BoundRefusal(msg)
    if kappa >= 1.0:
        msg = (
            f"kappa = {kappa:.6f} >= 1: the transition does not contract at this "
            f"rate, so no horizon follows. Report the limitation rather than a "
            f"bound; the geometric series a horizon would need does not converge."
        )
        raise BoundRefusal(msg)
    if kappa <= 0.0:
        msg = f"kappa must be positive, got {kappa}"
        raise BoundRefusal(msg)
    return kappa


def half_life(kappa: float) -> float:
    """Tokens for the bound to halve, ``ln 2 / ln(1/kappa)``. Refuses if >= 1."""
    return math.log(2.0) / math.log(1.0 / require_contraction(kappa))


def horizon_tokens(kappa: float, tol: float) -> float:
    """Tokens for the bound to fall below ``tol``: ``ln(1/tol)/ln(1/kappa)``."""
    if not 0.0 < tol < 1.0:
        msg = f"tol must lie in (0, 1), got {tol}"
        raise BoundRefusal(msg)
    return math.log(1.0 / tol) / math.log(1.0 / require_contraction(kappa))


def recurrent_state_bytes(*, n_layers: int, n_heads: int, head_dim: int,
                          directions: int = 2, dtype_bytes: int = 2,
                          token_shift_per_layer: int = 0) -> int:
    """Bytes of recurrent state -- CONSTANT in sequence length.

    Per head the WKV state is ``head_dim x head_dim``; the token-shift state is
    a small per-layer vector carried separately.  Arithmetic from the config,
    not a measurement: the paper must not present this as measured.
    """
    if min(n_layers, n_heads, head_dim, directions) <= 0:
        msg = "geometry must be positive"
        raise BoundRefusal(msg)
    per_direction = n_layers * n_heads * head_dim * head_dim
    return (directions * per_direction + n_layers * token_shift_per_layer) * dtype_bytes


def kv_cache_bytes(*, n_layers: int, hidden_size: int, seq_len: int,
                   dtype_bytes: int = 2, kv_heads: int | None = None,
                   n_heads: int | None = None) -> int:
    """Bytes of a same-size Transformer KV cache -- LINEAR in sequence length.

    ``kv_heads``/``n_heads`` express grouped-query attention: the cache shrinks
    by ``kv_heads / n_heads``.  Defaults to full multi-head (the comparison
    most favourable to the Transformer is the honest one to quote, provided the
    reduction is stated).
    """
    if min(n_layers, hidden_size, seq_len) <= 0:
        msg = "geometry must be positive"
        raise BoundRefusal(msg)
    width = hidden_size
    if kv_heads is not None and n_heads is not None:
        if not 0 < kv_heads <= n_heads:
            msg = f"kv_heads must be in (0, n_heads], got {kv_heads}/{n_heads}"
            raise BoundRefusal(msg)
        width = hidden_size * kv_heads // n_heads
    return 2 * n_layers * width * seq_len * dtype_bytes


def closed_form_matches_reference(seed: int = 0, size: int = 8) -> float:
    """Max abs deviation between the closed form and the reference kernel step.

    Kept in the module rather than the test so the number is reproducible from
    the library itself: the theorem is about the closed form, and if the kernel
    ever changes shape the deviation is the thing that moves.

    The reference step lives in ``scale/scratch_1b/rwkv_models.py``, which is part
    of the training tree and is NOT published with the paper repository.  Absence
    is therefore a named refusal rather than an ImportError, so a caller can tell
    "this check cannot run here" from "this check ran and failed" -- the two are
    very different claims about the closed form, and a bare ModuleNotFoundError
    reads as the second.
    """
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from scratch_1b.rwkv_models import rwkv7_step  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        msg = (
            "the reference kernel step is unavailable: scale/scratch_1b/ is part "
            "of the training tree and is not published with the paper repository. "
            "This check has NOT run, and no conclusion about the closed form "
            "should be drawn from its absence."
        )
        raise BoundRefusal(msg) from exc

    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    state = torch.randn(size, size, generator=generator, dtype=dtype)
    r = torch.randn(size, generator=generator, dtype=dtype)
    w = -DECAY_SCALE * torch.rand(size, generator=generator, dtype=dtype)
    k = torch.randn(size, generator=generator, dtype=dtype)
    v = torch.randn(size, generator=generator, dtype=dtype)
    a = float(torch.rand(1, generator=generator, dtype=dtype))
    khat = torch.nn.functional.normalize(k, dim=0)
    _, reference = rwkv7_step(state, r, w, k, v, torch.full((size,), a, dtype=dtype), khat)
    closed = transition(w, a, khat) @ state + torch.outer(k, v)
    return float((reference - closed).abs().max())
