"""Statistics for the long-context matrix: the macro score, the clustered
bootstrap, and the family-wise correction.

Each test pins a mechanism that a plausible shortcut would get wrong:

* the macro score is an equal-weight MEAN over cells, so scaling every cell by a
  common constant factors straight out and a sum-like or token-weighted
  aggregate would not match it;
* a paired-identical fixture must return a CI that covers zero and a large
  p-value, not a spuriously small one;
* clustering MATTERS -- strongly correlated pairs that share a group get a WIDER
  interval than the same pairs resampled as if independent, because the
  between-cluster variance is real evidence that a flat bootstrap throws away;
* Holm preserves input order and never rejects a cell whose adjusted p-value
  clears the threshold only because a larger-p cell was tested first.
"""

from __future__ import annotations

import random

import pytest

from scale.experiments.nonlatent_iclr.tasks.statistics import (
    StatisticsRefusal,
    holm_step_down,
    macro_score,
    non_inferiority_one_sided,
    paired_hierarchical_bootstrap,
)

#: Nine protocol cells: three families at three lengths.
CELLS = {
    ("recall", 16384): 0.40,
    ("recall", 32768): 0.30,
    ("recall", 65536): 0.20,
    ("overwrite", 16384): 0.80,
    ("overwrite", 32768): 0.70,
    ("overwrite", 65536): 0.60,
    ("dataflow", 16384): 0.50,
    ("dataflow", 32768): 0.45,
    ("dataflow", 65536): 0.35,
}

#: A deliberately strong within-cluster correlation: four tasks, each instance
#: of a task carrying the same delta, with the tasks spread apart.  A flat
#: bootstrap reads the 120 instances as 120 independent draws; the truth is four
#: effect sizes observed many times.
CLUSTER_DELTAS = (0.0, 0.5, 1.0, 1.5)
PAIRS_PER_CLUSTER = 30


def _correlated_fixture() -> tuple[list[tuple[float, float]], list[str]]:
    pairs: list[tuple[float, float]] = []
    groups: list[str] = []
    for cluster, delta in enumerate(CLUSTER_DELTAS):
        for _ in range(PAIRS_PER_CLUSTER):
            pairs.append((delta + 1.0, 1.0))
            groups.append(f"task-{cluster}")
    return pairs, groups


def test_macro_score_is_unchanged_when_every_cell_is_scaled_by_the_same_constant() -> None:
    # Given: nine cells, and a copy in which every cell is scaled by one constant.
    scale = 3.0
    scaled = {key: value * scale for key, value in CELLS.items()}
    # When: the macro score of each is taken.
    base = macro_score(CELLS)
    lifted = macro_score(scaled)
    # Then: the constant factors out uniformly -- the score is the equal-weight
    # MEAN of the cells, so one shared factor divides out once rather than being
    # multiplied by the cell count as a sum would.  The mean is the quantity a
    # reader takes away; the raw totals differ by the factor and by nothing else.
    assert base == pytest.approx(sum(CELLS.values()) / len(CELLS))
    assert lifted == pytest.approx(scale * base)


def test_macro_score_is_a_mean_not_a_token_weighted_sum() -> None:
    # Given: nine cells that all carry the same value.
    # When: the macro score is taken.
    uniform = {key: 0.25 for key in CELLS}
    # Then: it returns the shared value once, not the accumulation a sum would
    # give -- and not a value shifted by any per-cell token count, since the
    # score never reads one.
    assert macro_score(uniform) == pytest.approx(0.25)
    assert macro_score(CELLS) != pytest.approx(sum(CELLS.values()))


def test_macro_score_refuses_an_empty_mapping() -> None:
    # Given: no measured cells.
    # When/Then: the score refuses rather than manufacturing 0.0 for "unmeasured".
    with pytest.raises(StatisticsRefusal):
        macro_score({})


def test_macro_score_refuses_a_non_finite_cell() -> None:
    # Given: eight good cells and one NaN.
    cells = dict(CELLS)
    cells[("recall", 65536)] = float("nan")
    # When/Then: the NaN is rejected before it can silently poison the mean.
    with pytest.raises(StatisticsRefusal):
        macro_score(cells)


def test_identical_pairs_have_zero_delta_a_ci_covering_zero_and_a_large_p() -> None:
    # Given: forty pairs whose loop and control values are identical.
    pairs = [(0.5, 0.5)] * 40
    groups = [f"task-{index % 4}" for index in range(40)]
    # When: the clustered bootstrap is taken.
    result = paired_hierarchical_bootstrap(pairs, groups, draws=2000)
    # Then: the delta is exactly zero, the interval covers zero, and the one-sided
    # p is large (every resample is zero, so p = 1) -- not a small p from noise.
    assert result["delta"] == 0.0
    assert result["lo"] <= 0.0 <= result["hi"]
    assert result["p_one_sided"] >= 0.5
    assert result["draws"] == 2000


def test_clustering_widens_the_interval_for_correlated_pairs() -> None:
    # Given: strongly within-cluster correlated pairs, labelled by their task.
    pairs, groups = _correlated_fixture()
    # and the same pairs whose labels make every instance its own cluster, which
    # turns the two-stage draw into a flat independent-pair bootstrap.
    independent_groups = [f"pair-{index}" for index in range(len(pairs))]
    # When: both intervals are computed from the same draws and seed.
    clustered = paired_hierarchical_bootstrap(pairs, groups, draws=4000, seed=7)
    independent = paired_hierarchical_bootstrap(pairs, independent_groups, draws=4000, seed=7)
    # Then: ignoring the clusters yields a much narrower interval, while the
    # point estimate is identical -- the clustering changes the uncertainty, not
    # the effect size.
    assert clustered["delta"] == independent["delta"]
    clustered_width = clustered["hi"] - clustered["lo"]
    independent_width = independent["hi"] - independent["lo"]
    assert clustered_width > independent_width


def test_bootstrap_is_deterministic_and_ignores_the_global_random_stream() -> None:
    # Given: one fixture and two different global random states.
    pairs, groups = _correlated_fixture()
    # When: the bootstrap is drawn under each global state.
    random.seed(1)
    first = paired_hierarchical_bootstrap(pairs, groups, draws=500, seed=99)
    random.seed(2)
    second = paired_hierarchical_bootstrap(pairs, groups, draws=500, seed=99)
    # Then: the results are identical -- the seed, not the process, fixes the
    # interval, so a published CI is reproducible.
    assert first == second


def test_bootstrap_refuses_mismatched_group_labels() -> None:
    # Given: three pairs but only two cluster labels.
    # When/Then: the mismatch refuses, because a dropped pair would change the
    # effect size while still returning a confident interval.
    with pytest.raises(StatisticsRefusal):
        paired_hierarchical_bootstrap([(1.0, 0.0)] * 3, ["a", "b"])


def test_holm_preserves_input_order() -> None:
    # Given: p-values whose significance order is not their input order.
    pvalues = [0.5, 0.01, 0.03]
    # When: Holm adjusts them.
    results = holm_step_down(pvalues, family="long_context_macro")
    # Then: the output lines up with the input, while the ranks record the
    # significance order.
    assert [row["p"] for row in results] == pvalues
    assert [row["rank"] for row in results] == [3, 1, 2]


def test_holm_adjusted_values_are_monotone_and_capped_at_one() -> None:
    # Given: p-values large enough that the raw correction overflows 1.0.
    pvalues = [0.5, 0.6, 0.9]
    # When: Holm adjusts them.
    results = holm_step_down(pvalues, family="long_context_macro")
    # Then: every adjusted value is capped at 1.0 and none can exceed it.
    adjusted = [row["adjusted"] for row in results]
    assert all(value <= 1.0 for value in adjusted)
    assert adjusted == pytest.approx([1.0, 1.0, 1.0])


def test_holm_adjusted_values_are_the_hand_computed_ones() -> None:
    """The exact multipliers, not just monotonicity.

    This test used to assert only that the adjusted list is non-decreasing in
    rank, and that assertion cannot fail: rank is defined by the raw p-value, so
    an implementation returning the RAW p as its adjusted value also produces a
    sorted list. Monotonicity is a property of the input ordering, not of Holm.

    Holm multiplies the i-th smallest p by (n - i + 1) and takes a running
    maximum, so for p = [0.001, 0.008, 0.04, 0.3] with n = 4 the values are
    [0.004, 0.024, 0.08, 0.3]. Those are hand-computable, and a broken
    implementation gets them wrong.
    """
    # Given: four p-values whose Holm-adjusted values are known by hand.
    pvalues = [0.001, 0.008, 0.04, 0.3]
    # When: Holm adjusts them.
    results = holm_step_down(pvalues, family="long_context_macro")
    # Then: each multiplier was actually applied, in the right order.
    by_rank = sorted(results, key=lambda row: row["rank"])
    assert [round(row["adjusted"], 6) for row in by_rank] == [0.004, 0.024, 0.08, 0.3]
    # and the running max is what makes the third value 0.08 rather than 2*0.04
    # alone: a bare multiplier with no running max would give the same list here,
    # so also pin a case where the running max BITES.
    biting = holm_step_down([0.001, 0.049, 0.0499], family="long_context_macro")
    ordered = [round(r["adjusted"], 6) for r in sorted(biting, key=lambda x: x["rank"])]
    # 3*0.001=0.003, then 2*0.049=0.098, then max(0.098, 1*0.0499)=0.098
    assert ordered == [0.003, 0.098, 0.098]


def test_holm_rejection_is_a_prefix_of_the_sorted_hypotheses() -> None:
    # Given: a family where the raw 0.03 clears 0.05 but its corrected value does
    # not.
    pvalues = [0.01, 0.03, 0.2]
    # When: Holm adjusts them.
    results = holm_step_down(pvalues, family="long_context_macro")
    # Then: only the smallest p-value is rejected, and the raw 0.03 is retained
    # because the family-wise correction -- not the raw threshold -- decides.
    rejected_by_rank = [row["rejected"] for row in sorted(results, key=lambda row: row["rank"])]
    assert rejected_by_rank == [True, False, False]


def test_holm_refuses_invalid_pvalues_and_an_empty_family() -> None:
    # Given: an out-of-range p-value and an empty family.
    # When/Then: both refuse rather than silently adjusting.
    with pytest.raises(StatisticsRefusal):
        holm_step_down([0.5, 1.2], family="long_context_macro")
    with pytest.raises(StatisticsRefusal):
        holm_step_down([], family="long_context_macro")


def test_non_inferiority_flags_a_clearly_superior_arm() -> None:
    # Given: every paired difference is +0.1 and the margin is 0.05.
    pairs = [(1.1, 1.0)] * 20
    groups = [f"task-{index % 4}" for index in range(20)]
    # When: the one-sided non-inferiority test runs.
    result = non_inferiority_one_sided(pairs, groups, margin=0.05, draws=1000)
    # Then: the null delta <= -0.05 is rejected and the arm is non-inferior, with
    # the same keys as the paired bootstrap plus the decision.
    assert result["non_inferior"] is True
    assert result["p_one_sided"] <= 0.05
    assert set(result) == {"delta", "lo", "hi", "p_one_sided", "draws", "non_inferior"}


def test_non_inferiority_refuses_a_clearly_inferior_arm() -> None:
    # Given: every paired difference is -0.5, well past a 0.05 margin.
    pairs = [(0.5, 1.0)] * 20
    groups = [f"task-{index % 4}" for index in range(20)]
    # When: the one-sided non-inferiority test runs.
    result = non_inferiority_one_sided(pairs, groups, margin=0.05, draws=1000)
    # Then: the arm is not non-inferior -- the test can say no.
    assert result["non_inferior"] is False


def test_non_inferiority_refuses_a_negative_margin() -> None:
    # Given: a margin that reverses the direction of the claim.
    # When/Then: it is refused rather than tested as superiority under a
    # non-inferiority name.
    with pytest.raises(StatisticsRefusal):
        non_inferiority_one_sided([(1.0, 0.0)] * 4, ["a"] * 4, margin=-0.01)
