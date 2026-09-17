"""Macro score and resampling statistics for the long-context matrix.

The protocol's long-context macro score is one number: the EQUAL-weight mean of
nine cells -- three families (recall, overwrite, dataflow) at three lengths
(16K, 32K, 64K).  The comparison between two arms is PAIRED: both numbers of a
cell come from the *same* instance, so only their difference is evidence and the
between-instance spread is the thing a single-instance method averages over.

Two defects this module exists to prevent, both of which flatter a result:

1. Resampling TOKENS (or instances) instead of task/repository CLUSTERS.  A task
   contributes many instances that share its template, its gold generator and
   its repository, and those siblings are NOT independent evidence.  A flat
   bootstrap treats correlated siblings as fresh draws and returns a confidence
   interval far too narrow -- making an effect look established when the matrix
   has really only seen a handful of distinct tasks.  ``paired_hierarchical_
   bootstrap`` resamples the CLUSTERS first and the pairs within each drawn
   cluster second, so the between-cluster variance -- exactly the part a flat
   bootstrap discards -- enters the interval.
2. Reporting a raw p-value for each of nine cells with no family-wise control.
   Nine cells tested at alpha = 0.05 is a coin-flip machine: some cell clears the
   threshold by chance and reads as an effect.  ``holm_step_down`` gives each
   cell a corrected threshold that holds the family error rate at ``HOLM_ALPHA``.

Determinism is part of the contract: every bootstrap here draws from its own
``random.Random(seed)`` and never from the global ``random`` module, because a
published interval that moves when some unrelated caller reseeds the process is
not reproducible evidence.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from typing import Final

#: Resamples for the hierarchical bootstrap.  Large enough that the 2.5/97.5
#: percentiles are stable to roughly the Monte-Carlo standard error; the exact
#: figure is a fixed protocol constant so two analysts reproduce one interval.
DEFAULT_DRAWS: Final = 10000

#: Protocol seed.  Frozen so the published interval is a reproduction, not a
#: drawing: changing it changes the reported bounds and must be a deliberate act.
DEFAULT_SEED: Final = 20260917

#: Two-sided level for the percentile interval of the paired bootstrap.
DEFAULT_ALPHA: Final = 0.05

#: Family-wise level Holm holds.  Named separately from ``DEFAULT_ALPHA``
#: because the correction is over nine cells, not over the two CI tails, and a
#: reader must be able to change one without silently moving the other.
HOLM_ALPHA: Final = 0.05

#: Level for the non-inferiority decision.  One-sided by construction: the
#: hypothesis is directional (delta > -margin), so a two-tailed level would
#: halve the power of a claim that is only ever made in one direction.
NON_INFERIORITY_ALPHA: Final = 0.05


class StatisticsRefusal(ValueError):
    """The statistic cannot be computed as declared.

    Raised rather than returning ``0.0``/``NaN``/``1.0``.  A silent default here
    would be indistinguishable from a measurement -- an empty matrix would score
    zero, an all-identical fixture would report p = 1 -- and publishing a
    fabricated number is precisely the failure the program forbids.
    """


def macro_score(cells: dict[tuple[str, int], float]) -> float:
    """The equal-weight mean over the cells present.

    Empty input refuses rather than returning ``0.0``: a macro score of an empty
    matrix is undefined, and a caller handed ``0.0`` would publish "the model
    scored nothing" for "nothing was measured".

    The mean is over CELLS, never over tokens.  Weighting by token count would
    let a single 64K cell outvote the two 16K cells of the same family eightfold,
    turning a task-level average into a length-level one -- and the protocol
    fixes the three lengths precisely so the longest does not dominate.
    """
    if not cells:
        raise StatisticsRefusal(
            "macro score is undefined over an empty cell mapping; there is "
            "nothing to average and 0.0 would be a fabricated score"
        )
    total = 0.0
    for key, value in cells.items():
        if (not isinstance(key, tuple) or len(key) != 2
                or not isinstance(key[0], str) or not isinstance(key[1], int)):
            raise StatisticsRefusal(
                f"a macro cell is keyed by (family, length); got {key!r}, whose "
                f"shape would pool two different cells under one key"
            )
        if not math.isfinite(value):
            raise StatisticsRefusal(
                f"cell {key} has a non-finite score {value!r}; a NaN would "
                f"poison the whole mean and be invisible in it"
            )
        total += value
    return total / len(cells)


def paired_hierarchical_bootstrap(
    pairs: Sequence[tuple[float, float]],
    group_ids: Sequence[Hashable],
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float | int]:
    """Clustered percentile interval and one-sided p for a paired delta.

    ``pairs`` are ``(loop, control)`` observations and ``group_ids`` labels the
    cluster each belongs to -- the task or repository that makes siblings
    dependent.  The point estimate is the mean paired difference ``loop -
    control``.

    Each of ``draws`` resamples draws CLUSTERS with replacement (every pair of a
    drawn cluster moves together), then resamples that many pairs WITHIN each
    drawn cluster.  The two-stage draw is what keeps within-cluster correlation
    in the interval: a flat pair bootstrap would treat the correlated siblings
    as independent and shrink the interval toward zero.

    ``p_one_sided`` is the fraction of resampled deltas at or below zero -- the
    bootstrap CDF at the null boundary -- testing the one-sided alternative that
    the loop arm is better (delta > 0).  A fixture whose every pair is identical
    therefore returns p = 1.0, which is the honest reading of "no effect", not a
    small p.
    """
    if draws <= 0:
        raise StatisticsRefusal(f"draws must be positive, got {draws}")
    if not 0.0 < alpha < 1.0:
        raise StatisticsRefusal(f"alpha must lie in (0, 1), got {alpha}")
    deltas, clusters = _pair_deltas(pairs, group_ids)
    replicates = _bootstrap_deltas(clusters, draws=draws, seed=seed)
    lo, hi = _percentile_interval(replicates, alpha)
    below = sum(1 for value in replicates if value <= 0.0)
    return {
        "delta": sum(deltas) / len(deltas),
        "lo": lo,
        "hi": hi,
        "p_one_sided": below / len(replicates),
        "draws": draws,
    }


def holm_step_down(pvalues: list[float], *, family: str) -> list[dict]:
    """Holm step-down adjustment over one hypothesis family.

    One dict per input p-value, in INPUT order: ``{"p", "adjusted", "rejected",
    "rank"}``.  The family is named so a refusal says which family it could not
    adjust; ``HOLM_ALPHA`` is the family-wise level held.

    The k-th smallest p-value is adjusted by multiplying it by ``m - k + 1`` and
    taking a running maximum over the smaller p-values, capped at 1.0.  The
    running maximum is what makes ``adjusted`` monotone non-decreasing in the
    rank: without it, a cell could be "more significant after correction" than a
    smaller-p cell it is supposed to follow, which is a contradiction a reader
    cannot see and a table would hide.  ``rejected`` is the prefix property
    ``adjusted <= HOLM_ALPHA``, so a cell is never rejected while a more
    significant one is retained.
    """
    if not isinstance(family, str) or not family:
        raise StatisticsRefusal("a hypothesis family must be named, not empty")
    if not pvalues:
        raise StatisticsRefusal(
            f"the {family} family has no p-values to adjust; an empty correction "
            f"is not a correction"
        )
    for value in pvalues:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise StatisticsRefusal(
                f"the {family} family contains a p-value outside [0, 1]: {value!r}"
            )
    count = len(pvalues)
    # Ties are broken by input index so the assignment of consecutive ranks is
    # deterministic -- two equal p-values must not swap ranks run to run.
    ascending = sorted(range(count), key=lambda index: (pvalues[index], index))
    adjusted_by_rank: list[float] = []
    running_max = 0.0
    for rank, index in enumerate(ascending, start=1):
        running_max = max(running_max, (count - rank + 1) * pvalues[index])
        adjusted_by_rank.append(min(1.0, running_max))
    results: list[dict] = [{} for _ in range(count)]
    for rank, index in enumerate(ascending, start=1):
        adjusted = adjusted_by_rank[rank - 1]
        results[index] = {
            "p": pvalues[index],
            "adjusted": adjusted,
            "rejected": adjusted <= HOLM_ALPHA,
            "rank": rank,
        }
    return results


def non_inferiority_one_sided(
    pairs: Sequence[tuple[float, float]],
    group_ids: Sequence[Hashable],
    *,
    margin: float,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int | bool]:
    """Test the directional null ``delta <= -margin`` against ``delta > -margin``.

    The same clustered bootstrap as :func:`paired_hierarchical_bootstrap`, so the
    interval and the decision share one resampling and cannot disagree.  The
    p-value is the fraction of resampled deltas at or below ``-margin`` -- the
    bootstrap CDF at the margin -- and ``non_inferior`` is ``p <=
    NON_INFERIORITY_ALPHA``.  A negative margin is refused: it would reverse the
    direction of the claim while keeping the name.
    """
    if not math.isfinite(margin) or margin < 0.0:
        raise StatisticsRefusal(
            f"a non-inferiority margin is a non-negative loss, got {margin!r}; a "
            f"negative margin would test superiority while still reading as "
            f"non-inferiority"
        )
    if draws <= 0:
        raise StatisticsRefusal(f"draws must be positive, got {draws}")
    deltas, clusters = _pair_deltas(pairs, group_ids)
    replicates = _bootstrap_deltas(clusters, draws=draws, seed=seed)
    lo, hi = _percentile_interval(replicates, DEFAULT_ALPHA)
    at_or_below = sum(1 for value in replicates if value <= -margin)
    p_value = at_or_below / len(replicates)
    return {
        "delta": sum(deltas) / len(deltas),
        "lo": lo,
        "hi": hi,
        "p_one_sided": p_value,
        "draws": draws,
        "non_inferior": p_value <= NON_INFERIORITY_ALPHA,
    }


def _pair_deltas(
    pairs: Sequence[tuple[float, float]], group_ids: Sequence[Hashable]
) -> tuple[list[float], list[list[float]]]:
    """Paired differences and the clusters that own them, in first-seen order.

    Order is fixed by first appearance rather than by sorted group id so the
    resampling consumes the seed identically on every run; a dict iteration order
    that changed with insertion would make the published interval move.
    """
    if len(pairs) != len(group_ids):
        raise StatisticsRefusal(
            f"every pair needs a cluster label: got {len(pairs)} pairs and "
            f"{len(group_ids)} group ids"
        )
    if not pairs:
        raise StatisticsRefusal("a paired bootstrap over no pairs has nothing to resample")
    deltas: list[float] = []
    clusters: dict[Hashable, list[float]] = {}
    order: list[Hashable] = []
    for (loop, control), group in zip(pairs, group_ids, strict=True):
        delta = float(loop) - float(control)
        if not math.isfinite(delta):
            raise StatisticsRefusal(
                f"a paired difference is non-finite ({loop!r} - {control!r}); a "
                f"NaN would make every resampled mean NaN"
            )
        deltas.append(delta)
        if group not in clusters:
            clusters[group] = []
            order.append(group)
        clusters[group].append(delta)
    return deltas, [clusters[group] for group in order]


def _bootstrap_deltas(clusters: list[list[float]], *, draws: int, seed: int) -> list[float]:
    """``draws`` two-stage resampled means: clusters first, pairs within a draw.

    Using a private ``random.Random(seed)`` and never the module-level ``random``
    is deliberate: the global stream is shared with every other caller, so a
    reseed elsewhere would silently change a published interval.
    """
    generator = random.Random(seed)
    cluster_count = len(clusters)
    replicates: list[float] = []
    for _ in range(draws):
        total = 0.0
        drawn = 0
        for _ in range(cluster_count):
            cluster = clusters[generator.randrange(cluster_count)]
            size = len(cluster)
            for _ in range(size):
                total += cluster[generator.randrange(size)]
                drawn += 1
        replicates.append(total / drawn)
    return replicates


def _percentile_interval(values: list[float], alpha: float) -> tuple[float, float]:
    """The ``alpha``-level percentile interval, by linear interpolation."""
    ordered = sorted(values)
    return _quantile(ordered, alpha / 2.0), _quantile(ordered, 1.0 - alpha / 2.0)


def _quantile(ordered: Sequence[float], q: float) -> float:
    """The ``q`` quantile of an already-sorted sequence, linearly interpolated."""
    if not ordered:
        raise StatisticsRefusal("cannot take a quantile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
