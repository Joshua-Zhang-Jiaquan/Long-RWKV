"""Lazy Task4 registry and readiness verification API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .tasks.exact_tasks import generate_exact_task
from .tasks.exact_tasks import InfeasibleTaskError
from .tasks.exact_tasks import validate_exact_gold
from .tasks.external_assets import AssetStatus
from .tasks.external_assets import inventory_external_assets
from .tasks.models import ExactTaskRequest
from .tasks.provenance import make_provenance

type JsonValue = str | int | bool | list["JsonValue"] | dict[str, "JsonValue"]

FAMILIES = ("associative_recall", "overwrite_delayed_query", "finite_hmm", "code_dataflow")
LENGTHS = (4096, 8192, 16384, 32768, 65536)
POSITIONS = (10, 50, 90)
LOADS = (1, 8, 32, 128)
DISTRACTORS = ("none", "random", "similar")
DATA_SEEDS = (101, 102, 103, 104, 105)


@dataclass(frozen=True, slots=True)
class TaskCell:
    """A lazy 200-instance exact-task cell; it never stores a giant corpus."""

    family: str
    length: int
    position_fraction: int
    load: int
    distractor: str
    instances_per_cell: int = 200


@dataclass(frozen=True, slots=True)
class Registry:
    """Task declarations plus their local external-asset root."""

    cells: tuple[TaskCell, ...]
    data_seeds: tuple[int, ...]
    asset_root: Path
    materialized_instances: tuple[()] = ()

    def split_violations(self) -> tuple[str, ...]:
        """Confirm variants share a source-family split assigned before projection."""
        families: dict[str, str] = {}
        for cell in self.cells:
            for seed in self.data_seeds:
                request = ExactTaskRequest(cell.family, seed, cell.length, cell.position_fraction, cell.load, cell.distractor, 0)
                try:
                    task = generate_exact_task(request)
                except InfeasibleTaskError:
                    continue
                source = make_provenance(task.condition.source_bytes, task.condition.source_family)
                existing = families.setdefault(source.source_family, source.split)
                if existing != source.split:
                    return (source.source_family,)
        return ()


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Readiness separates passing CPU checks from blocked benchmark completion."""

    exact_checks_passed: bool
    successful_cells: int
    infeasible_cells: int
    blocked: tuple[AssetStatus, ...]

    @property
    def ready(self) -> bool:
        """Task4 is complete only with valid exact checks and all accepted assets."""
        return self.exact_checks_passed and not self.blocked

    @property
    def exit_code(self) -> int:
        """Return the documented fail-closed process status."""
        return 0 if self.ready else 2


def build_registry(asset_root: Path | None = None) -> Registry:
    """Build declarations only; generation occurs only for an explicitly requested cell."""
    root = asset_root if asset_root is not None else Path("DAN/nonlatent_iclr/task4_assets")
    cells = tuple(TaskCell(family, length, position, load, distractor) for family in FAMILIES for length in LENGTHS for position in POSITIONS for load in LOADS for distractor in DISTRACTORS)
    return Registry(cells, DATA_SEEDS, root)


def verify_registry(registry: Registry) -> ReadinessReport:
    """Verify independent exact gold and then inventory non-generated requirements."""
    successful = 0
    infeasible = 0
    exact_checks = True
    for cell in registry.cells:
        request = ExactTaskRequest(cell.family, 101, cell.length, cell.position_fraction, cell.load, cell.distractor, 0)
        try:
            task = generate_exact_task(request)
        except InfeasibleTaskError:
            infeasible += 1
            continue
        successful += 1
        exact_checks = exact_checks and validate_exact_gold(task)
    exact_checks = exact_checks and successful > 0 and not registry.split_violations()
    blocked = tuple(status for status in inventory_external_assets(registry.asset_root) if not status.ready)
    return ReadinessReport(exact_checks, successful, infeasible, blocked)


def unbound_length_qualification() -> dict[str, JsonValue]:
    """Conservative placeholder when no replay-bound summary exists at the asset root."""
    return {
        "status": "BLOCKED",
        "representative_token_lengths_qualified": False,
        "matrix_token_lengths_qualified": False,
        "reason": "no accepted length-qualification summary is bound at the asset root",
    }


def registry_document(registry: Registry, report: ReadinessReport, length_qualification: JsonValue | None = None) -> dict[str, JsonValue]:
    """Render serializable declarations without any generated answers or prompts.

    ``length_qualification`` is emitted verbatim when an accepted summary is bound, so
    regenerating the registry cannot silently downgrade already-qualified token evidence.
    """
    return {
        "schema_version": 1,
        "status": "READY" if report.ready else "BLOCKED",
        "length_qualification": unbound_length_qualification() if length_qualification is None else length_qualification,
        "data_seeds": list(registry.data_seeds),
        "instances_per_cell": 200,
        "representative_cell_check": {"seed": 101, "instance_index": 0, "successful_cells": report.successful_cells, "infeasible_cells": report.infeasible_cells},
        "families": list(FAMILIES),
        "cells": [
            {"family": cell.family, "length": cell.length, "position_fraction": cell.position_fraction, "load": cell.load, "distractor": cell.distractor}
            for cell in registry.cells
        ],
        "blocked": [{"requirement": item.requirement, "reason": item.reason} for item in report.blocked],
    }
