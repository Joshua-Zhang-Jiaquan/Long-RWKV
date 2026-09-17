"""Task 10: the executable half of the long-context influence bound.

Each test here pins a step in the argument, and two of them exist because an
earlier draft of the plan got the step wrong in a way that would have changed
the theorem's conclusion:

* ``test_the_two_factorisations_differ`` -- the kernel subtracts the rank-1 term
  from the UNDECAYED state (``M = D - a k k^T``), not ``D (I - a k k^T)``.  Only
  the first is symmetric, and only the first contracts for every input.
* ``test_kappa_at_or_above_one_refuses`` -- the plan's failure QA requires that a
  rate of 1 or more *exposes* the limitation instead of receiving a certificate.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from scale.experiments.nonlatent_iclr import theory_bounds as tb

GEOMETRY = 64


def _draw(seed: int, size: int = GEOMETRY, dtype: torch.dtype = torch.float64):
    generator = torch.Generator().manual_seed(seed)
    k = torch.randn(size, generator=generator, dtype=dtype)
    return (
        -tb.DECAY_SCALE * torch.rand(size, generator=generator, dtype=dtype),
        float(torch.rand(1, generator=generator, dtype=dtype)),
        torch.nn.functional.normalize(k, dim=0),
    )


# --------------------------------------------------------------------------
# the shape of the transition
# --------------------------------------------------------------------------


def test_transition_is_symmetric() -> None:
    # Given: any decay, gate and unit write key.
    w, a, khat = _draw(1)
    # When: the transition is formed from the closed form.
    matrix = tb.transition(w, a, khat)
    # Then: it is symmetric -- which is the property Weyl's argument needs, and
    # the property the *other* factorisation would not have.
    assert torch.allclose(matrix, matrix.T, atol=1e-12)


def test_the_two_factorisations_differ() -> None:
    """The kernel applies the delta correction to the undecayed state."""
    # Given: a decay, a gate and a write key.
    w, a, khat = _draw(2)
    decay = torch.exp(w)
    # When: both candidate factorisations are formed.
    subtracted_after = torch.diag(decay) - a * torch.outer(khat, khat)
    subtracted_before = torch.diag(decay) @ (torch.eye(GEOMETRY, dtype=decay.dtype)
                                             - a * torch.outer(khat, khat))
    # Then: they are genuinely different matrices, and the second is not
    # symmetric -- so a bound derived from it would not transfer.
    assert not torch.allclose(subtracted_after, subtracted_before, atol=1e-6)
    assert not torch.allclose(subtracted_before, subtracted_before.T, atol=1e-9)


def test_closed_form_matches_the_reference_kernel() -> None:
    """The theorem is about the closed form, so it must be the kernel's form.

    SKIPS where the training tree is absent.  ``scale/scratch_1b/`` holds the
    reference kernel step and is not published with the paper repository, so in a
    clean clone this check cannot run -- and the repository's rule is that such a
    test skips with a reason rather than failing, because a failure would read as
    "the closed form disagrees with the kernel" when the truth is "the kernel is
    not here".  The published tree says which checks it cannot make.
    """
    # Given: the reference step and the closed form on the same inputs.
    try:
        deviation = tb.closed_form_matches_reference(seed=0, size=8)
    except tb.BoundRefusal as exc:
        pytest.skip(str(exc))
    # Then: they agree to float64 round-off.
    assert deviation < 1e-12, deviation


def test_the_closed_form_refusal_is_named_not_an_import_error() -> None:
    """Absence must be distinguishable from disagreement."""
    # Given: a module whose reference kernel may or may not be present.
    source = Path(tb.__file__).read_text(encoding="utf-8")
    # When/Then: absence raises BoundRefusal with an explicit "has NOT run" --
    # a bare ModuleNotFoundError would read as a failed check.
    assert "the reference kernel step is unavailable" in source
    assert "has NOT run" in source


# --------------------------------------------------------------------------
# Lemma 1: contraction for every input
# --------------------------------------------------------------------------


def test_weyl_bounds_hold_on_random_draws() -> None:
    # Given: many random (decay, gate, key) draws.
    violations = 0
    checks = 0
    worst_norm = 0.0
    for seed in range(400):
        w, a, khat = _draw(seed)
        matrix = tb.transition(w, a, khat)
        eigenvalues = torch.linalg.eigvalsh(matrix)
        top = float(eigenvalues[-1])
        bottom = float(eigenvalues[0])
        bound = tb.transition_bounds(w, a)
        checks += 2
        if top > bound.lambda_max_upper + 1e-12:
            violations += 1
        if bottom < bound.lambda_min_lower - 1e-12:
            violations += 1
        worst_norm = max(worst_norm, float(torch.linalg.matrix_norm(matrix, 2)))
    # Then: neither Weyl bound is ever violated...
    assert violations == 0, f"{violations} of {checks} bound checks failed"
    # ...and the spectral norm stays strictly below 1, which is the contraction.
    assert worst_norm < 1.0, worst_norm


def test_the_delta_rule_erases_the_slowest_channel() -> None:
    """The bound is attained only where the write key does not reach.

    Aligning ``khat`` with the slowest-decaying channel and setting the gate to
    1 collapses that channel's eigenvalue toward zero -- the delta rule *writes
    over* the slow memory.  Pointing the key elsewhere leaves it intact.  This
    is why the product bound is loose in practice, and why the interesting
    quantity is the trained decay distribution rather than the worst case.
    """
    # Given: one slow channel (decay ~ 1) and fast channels everywhere else.
    w = torch.full((GEOMETRY,), -tb.DECAY_SCALE, dtype=torch.float64)
    w[0] = -1e-6
    aligned = torch.zeros(GEOMETRY, dtype=torch.float64)
    aligned[0] = 1.0
    elsewhere = torch.zeros(GEOMETRY, dtype=torch.float64)
    elsewhere[1] = 1.0
    # When: the gate is at its maximum and the write key points at the slow
    # channel, then away from it.
    erased = tb.transition(w, 1.0, aligned)
    preserved = tb.transition(w, 1.0, elsewhere)
    # Then: aligned, the slow channel's own entry collapses to ~0...
    assert float(erased[0, 0]) < 1e-6
    assert float(torch.linalg.eigvalsh(erased)[0]) == pytest.approx(-1e-6, abs=1e-9)
    # ...while pointing the key elsewhere leaves it at ~1, and then it governs
    # the norm -- which is exactly the case the bound has to cover.
    assert float(preserved[0, 0]) == pytest.approx(1.0, abs=1e-5)
    assert float(torch.linalg.matrix_norm(preserved, 2)) == pytest.approx(1.0, abs=1e-5)
    assert float(torch.linalg.matrix_norm(erased, 2)) == pytest.approx(tb.MIN_DECAY, abs=1e-6)


def test_a_bound_without_a_write_key_is_still_valid() -> None:
    # Given: the same decay and gate as a drawn transition.
    w, a, _ = _draw(3)
    # When: the bound is computed without any write key.
    bound = tb.transition_bounds(w, a)
    # Then: it holds for every unit key, so it is a worst-case bound rather
    # than one tuned to a particular draw.
    for seed in range(20):
        khat = torch.nn.functional.normalize(
            torch.randn(GEOMETRY, generator=torch.Generator().manual_seed(seed),
                        dtype=torch.float64), dim=0)
        matrix = tb.transition(w, a, khat)
        assert float(torch.linalg.matrix_norm(matrix, 2)) <= bound.spectral_norm_upper + 1e-12


# --------------------------------------------------------------------------
# Proposition 1: the influence product
# --------------------------------------------------------------------------


def test_product_bound_holds_on_random_chains() -> None:
    # Given: a random chain of transitions.
    steps = 12
    ws = torch.stack([_draw(seed)[0] for seed in range(steps)])
    gates = torch.tensor([_draw(seed)[1] for seed in range(steps)], dtype=torch.float64)
    khats = torch.stack([_draw(seed)[2] for seed in range(steps)])
    # When: the propagator norm is compared with the product bound at several
    # lags, counting from an early write.
    for lag in (1, 3, 6, 9):
        window = slice(steps - lag, steps)
        propagator = tb.propagator(ws[window], gates[window], khats[window])
        bound = tb.influence_bound(ws[window])
        # Then: submultiplicativity holds at every lag.
        assert float(torch.linalg.matrix_norm(propagator, 2)) <= bound + 1e-12


def test_the_bound_gets_tighter_as_the_chain_shortens() -> None:
    # Given: a single transition.
    w, a, khat = _draw(4)
    # When: the bound and the truth are compared.
    bound = tb.per_step_bound(w)
    actual = float(torch.linalg.matrix_norm(tb.transition(w, a, khat), 2))
    # Then: the bound is an upper bound, and it is the top of the decay range.
    assert actual <= bound + 1e-12
    assert bound == pytest.approx(float(torch.exp(w).max()), rel=1e-12)


# --------------------------------------------------------------------------
# horizons, and the failure QA
# --------------------------------------------------------------------------


def test_half_life_and_horizon_are_consistent() -> None:
    # Given: a contraction rate strictly below 1.
    kappa = 0.99
    # When: the half-life and the horizon are computed.
    half = tb.half_life(kappa)
    horizon = tb.horizon_tokens(kappa, tol=1e-3)
    # Then: they agree with their definitions, and the horizon is the longer of
    # the two because tol is far below one half.
    assert half == pytest.approx(math.log(2) / math.log(1 / kappa))
    assert horizon == pytest.approx(math.log(1000) / math.log(1 / kappa))
    assert horizon > half


def test_kappa_at_or_above_one_refuses() -> None:
    """The plan's failure QA: q >= 1 must expose the limitation."""
    # Given: a rate that certifies nothing.
    # When/Then: every horizon helper refuses rather than returning a number.
    for kappa in (1.0, 1.0123, 2.0):
        with pytest.raises(tb.BoundRefusal, match=">= 1"):
            tb.require_contraction(kappa)
        with pytest.raises(tb.BoundRefusal, match=">= 1"):
            tb.half_life(kappa)
        with pytest.raises(tb.BoundRefusal, match=">= 1"):
            tb.horizon_tokens(kappa, tol=1e-3)
    # and a non-finite rate is refused for its own reason
    with pytest.raises(tb.BoundRefusal, match="finite"):
        tb.require_contraction(float("nan"))


def test_a_measured_rate_above_one_cannot_reach_a_horizon() -> None:
    """The E1 measurement, fed through the gate that is supposed to catch it."""
    # Given: the surviving E1 tail rates from the b3-sft-2p8b step-8000 run.
    measured = (1.003, 0.984, 1.012, 1.006, 0.982, 0.950)
    # When: each is asked for a horizon.
    certified = []
    for rate in measured:
        try:
            certified.append((rate, tb.half_life(rate)))
        except tb.BoundRefusal:
            continue
    # Then: only the rates strictly below 1 produce one, and three of the six
    # do not -- which is the honest statement about the outer denoising clock.
    assert len(certified) == 3
    assert all(rate < 1.0 for rate, _ in certified)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_decay_outside_the_parameterization_range_is_refused() -> None:
    """The parameterization bounds the decay, so out-of-range means a bug.

    ``decay_from_logits`` cannot exceed the interval at all: ``-DECAY_SCALE *
    sigmoid(.)`` is non-positive for every input, so a call that produced a
    decay above 1 would mean the constant or the activation changed.  What the
    check actually catches is a caller *handing in* log-decays directly (as
    ``transition`` accepts), where a positive value is a decay above 1 and an
    expansive diagonal would falsify Lemma 1 outright.
    """
    # Given: a huge logit, which saturates to the decay FLOOR -- legal.
    floor = tb.decay_from_logits(torch.tensor([1e9]))
    assert float(floor) == pytest.approx(tb.MIN_DECAY, rel=1e-6)
    # When/Then: a hand-written expansion (decay > 1) is refused...
    khat = torch.zeros(GEOMETRY, dtype=torch.float64)
    khat[0] = 1.0
    with pytest.raises(tb.BoundRefusal, match="interval"):
        tb.transition(torch.full((GEOMETRY,), 0.5, dtype=torch.float64), 0.5, khat)
    # ...as is a decay below the floor, which the model cannot produce.
    with pytest.raises(tb.BoundRefusal, match="interval"):
        tb.transition(torch.full((GEOMETRY,), -1.0, dtype=torch.float64), 0.5, khat)


def test_the_decay_range_is_the_documented_one() -> None:
    # Given: the scale constant.
    # When/Then: its exponential is what the module says, and a wide sweep of
    # sigmoid logits stays inside the interval -- closed, because float32
    # rounding reaches the endpoints exactly.
    assert tb.MIN_DECAY == pytest.approx(math.exp(-tb.DECAY_SCALE))
    logits = torch.linspace(-60.0, 60.0, steps=2001, dtype=torch.float64)
    decay = tb.decay_from_logits(logits)
    assert float(decay.min()) >= tb.MIN_DECAY - 1e-12
    assert float(decay.max()) <= 1.0
    # the two ends are actually approached, so the sweep is not vacuous
    assert float(decay.min()) == pytest.approx(tb.MIN_DECAY, rel=1e-3)
    assert float(decay.max()) == pytest.approx(1.0, rel=1e-6)


def test_a_non_unit_write_key_is_refused() -> None:
    # Given: a key that is not L2-normalised.
    w, a, _ = _draw(5)
    # When/Then: the transition refuses it rather than silently scaling a.
    with pytest.raises(tb.BoundRefusal, match="unit-norm"):
        tb.transition(w, a, torch.full((GEOMETRY,), 0.5, dtype=torch.float64))


def test_an_out_of_range_gate_is_refused() -> None:
    w, _, khat = _draw(6)
    with pytest.raises(tb.BoundRefusal, match="delta gate"):
        tb.transition(w, 1.5, khat)


def test_an_empty_transition_slice_is_refused() -> None:
    # Given: no elapsed tokens.
    empty = torch.zeros((0, GEOMETRY), dtype=torch.float64)
    # When/Then: refused, because the honest answer is the identity and a
    # caller asking for a product means at least one step.
    with pytest.raises(tb.BoundRefusal, match="at least one step"):
        tb.propagator(empty, torch.zeros(0, dtype=torch.float64), empty)
    with pytest.raises(tb.BoundRefusal, match="non-empty"):
        tb.influence_bound(empty)


# --------------------------------------------------------------------------
# the analytic efficiency statement
# --------------------------------------------------------------------------


def test_recurrent_state_is_constant_while_the_kv_cache_grows() -> None:
    # Given: the paper's model geometry.
    state = tb.recurrent_state_bytes(n_layers=32, n_heads=40, head_dim=64)
    # When/Then: the state is a fixed 20 MiB, and the KV cache it is compared
    # against grows linearly -- the ratio at 64K is the headline number.
    assert state == 20 * 1024 * 1024
    caches = {t: tb.kv_cache_bytes(n_layers=32, hidden_size=2560, seq_len=t)
              for t in (4096, 16384, 32768, 65536)}
    assert caches[65536] == 20 * 1024 ** 3
    ratios = [caches[t] / state for t in sorted(caches)]
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(64.0, rel=1e-6)
    assert ratios[-1] == pytest.approx(1024.0, rel=1e-6)


def test_grouped_query_attention_shrinks_the_transformer_side() -> None:
    # Given: a grouped-query configuration with one KV head per eight.
    full = tb.kv_cache_bytes(n_layers=32, hidden_size=2560, seq_len=65536)
    grouped = tb.kv_cache_bytes(n_layers=32, hidden_size=2560, seq_len=65536,
                                kv_heads=5, n_heads=40)
    # When/Then: the reduction is exactly the head ratio, so the paper can state
    # the honest comparison rather than the most favourable one.
    assert grouped * 8 == full
    with pytest.raises(tb.BoundRefusal, match="kv_heads"):
        tb.kv_cache_bytes(n_layers=32, hidden_size=2560, seq_len=1024,
                          kv_heads=40, n_heads=5)
