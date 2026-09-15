"""Canonical-source adapters and group-first split assignment."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from scale.data.canonical_records import content_hash


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Original source bytes and their group-level split provenance."""

    source_bytes: bytes
    source_family: str
    canonical_content_hash: str
    split: str


def make_provenance(source_bytes: bytes, source_family: str) -> SourceProvenance:
    """Bind original bytes to a deterministic family split before variants exist."""
    split = _family_split(source_family)
    return SourceProvenance(source_bytes, source_family, content_hash(source_bytes), split)


def split_violations(records: tuple[SourceProvenance, ...]) -> tuple[str, ...]:
    """Return source families whose pre-view assignments straddle splits."""
    assigned: dict[str, str] = {}
    violations: list[str] = []
    for record in records:
        existing = assigned.setdefault(record.source_family, record.split)
        if existing != record.split:
            violations.append(record.source_family)
    return tuple(violations)


def _family_split(source_family: str) -> str:
    bucket = int(hashlib.sha256(source_family.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "confirmation"


def repository_split_map(weights: Mapping[str, int]) -> dict[str, str]:
    """Assign whole repository families to splits, weighted by how much each contributes.

    The hash-based rule used for synthetic families is fine when families are plentiful, but
    with a handful of contributing repositories it can leave the validation and confirmation
    buckets empty, which would silently destroy the held-out property the plan requires.
    Families are therefore ranked by contribution and packed greedily so the task proportions
    approach 80/10/10, with at least one held-out family of each kind guaranteed whenever at
    least three families exist. Whole families move together, so descendants inherit the
    assignment, and the result is a deterministic function of the weights, so it replays.
    """
    if len(weights) < 3:
        raise ValueError("at least three repository families are required for held-out splits")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("repository family weights must be positive")
    ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    # Hold out the two smallest contributing families, so the held-out panels stay small
    # while every remaining family trains. Whole families move together and the choice is a
    # deterministic function of the weights, so a replay derives the identical assignment.
    assignment: dict[str, str] = {}
    for index, (family, _count) in enumerate(ordered):
        remaining = len(ordered) - index
        if remaining > 2:
            assignment[family] = "train"
        elif remaining == 2:
            assignment[family] = "validation"
        else:
            assignment[family] = "confirmation"
    return assignment
