#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      PYTHONPATH=<repo-root> uv run --offline validation_probe.py
# 3. Or use the repository interpreter without writing caches:
#      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<repo-root> python validation_probe.py
# ──────────────────

from __future__ import annotations

import json
import shutil
from importlib import import_module
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, TypeVar

metric_records_module = import_module("scale.experiments.nonlatent_iclr.metric_records")
metrics_module = import_module("scale.experiments.nonlatent_iclr.metrics")
load_sampler_panel = metric_records_module.load_sampler_panel
summarize_sampler_pair = metric_records_module.summarize_sampler_pair
MetricInputError = metrics_module.MetricInputError
MetricName = metrics_module.MetricName


class Metrics(TypedDict, total=False):
    max_run_frac: float


class RawRecord(TypedDict, total=False):
    document_id: str
    seed: int
    arm: str
    metrics: Metrics
    failure: NotRequired[str | None]


class Grid(TypedDict):
    mask_ratios: list[float]
    steps: list[int]


class RawPanel(TypedDict, total=False):
    schema: str
    num_shards: int
    shard_index: int
    n_records: int
    grid: Grid
    records: list[RawRecord]


@dataclass(frozen=True, slots=True)
class Probe:
    accepted: bool
    detail: str


Result = TypeVar("Result")


def _capture(action: Callable[[], Result]) -> Probe:
    try:
        value = action()
    except MetricInputError as error:
        return Probe(False, str(error))
    return Probe(True, repr(value))


def _write(path: Path, payload: RawPanel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=True) + "\n", encoding="utf-8")


def _record(document_id: str, value: float) -> RawRecord:
    return {
        "document_id": document_id,
        "seed": 42,
        "arm": "r70_s1",
        "metrics": {"max_run_frac": value},
    }


def _single_panel(record: RawRecord) -> RawPanel:
    return {
        "schema": "qz_capability_sampler_shard_v1",
        "num_shards": 1,
        "shard_index": 0,
        "n_records": 1,
        "grid": {"mask_ratios": [0.7], "steps": [1]},
        "records": [record],
    }


def main() -> None:
    work = Path(__file__).with_name("validation-work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    duplicate = work / "duplicate"
    duplicate_loop = _single_panel(_record("doc-1", 0.2))
    duplicate_loop["n_records"] = 2
    duplicate_loop["records"] = [_record("doc-1", 0.2), _record("doc-1", 0.3)]
    _write(duplicate / "loop/sampler.shard0of1.json", duplicate_loop)
    _write(duplicate / "control/sampler.shard0of1.json", _single_panel(_record("doc-1", 0.1)))

    nonfinite = work / "nonfinite"
    _write(nonfinite / "loop/sampler.shard0of1.json", _single_panel(_record("doc-1", float("nan"))))
    _write(nonfinite / "control/sampler.shard0of1.json", _single_panel(_record("doc-1", 0.1)))

    failed_finite = work / "failed-finite"
    failed_record = _record("doc-1", 0.9)
    failed_record["failure"] = "OOM"
    _write(failed_finite / "loop/sampler.shard0of1.json", _single_panel(failed_record))
    _write(failed_finite / "control/sampler.shard0of1.json", _single_panel(_record("doc-1", 0.1)))

    failed_missing_metric = work / "failed-missing-metric"
    failed_without_value = _record("doc-1", 0.9)
    failed_without_value["failure"] = "OOM"
    failed_without_value["metrics"] = {}
    _write(
        failed_missing_metric / "panel/sampler.shard0of1.json",
        _single_panel(failed_without_value),
    )

    incomplete = work / "incomplete"
    incomplete_payload = _single_panel(_record("doc-1", 0.2))
    incomplete_payload["num_shards"] = 8
    incomplete_payload["n_records"] = 999
    incomplete_payload["grid"] = {"mask_ratios": [0.7, 1.0], "steps": [1, 32]}
    _write(incomplete / "panel/sampler.shard0of8.json", incomplete_payload)

    missing_pair = work / "missing-pair"
    _write(missing_pair / "loop/sampler.shard0of1.json", _single_panel(_record("doc-1", 0.2)))
    _write(missing_pair / "control/sampler.shard0of1.json", _single_panel(_record("doc-2", 0.1)))

    malformed = work / "malformed"
    malformed_payload = _single_panel(_record("doc-1", 0.2))
    malformed_payload["schema"] = "wrong"
    _write(malformed / "panel/sampler.shard0of1.json", malformed_payload)

    boolean = work / "boolean"
    boolean_record = _record("doc-1", True)
    boolean_record["seed"] = True
    _write(boolean / "panel/sampler.shard0of1.json", _single_panel(boolean_record))

    incomplete_panel = load_sampler_panel(incomplete, "panel", MetricName.MAX_RUN_FRAC)
    boolean_panel = load_sampler_panel(boolean, "panel", MetricName.MAX_RUN_FRAC)
    outcomes = {
        "missing_panel": asdict(_capture(lambda: load_sampler_panel(work, "absent", MetricName.MAX_RUN_FRAC))),
        "duplicate_pair": asdict(_capture(lambda: summarize_sampler_pair(duplicate, "loop", "control", MetricName.MAX_RUN_FRAC))),
        "nonfinite_pair": asdict(_capture(lambda: summarize_sampler_pair(nonfinite, "loop", "control", MetricName.MAX_RUN_FRAC))),
        "failed_record_with_finite_metric": asdict(_capture(lambda: summarize_sampler_pair(failed_finite, "loop", "control", MetricName.MAX_RUN_FRAC)[2])),
        "failed_record_without_metric": asdict(_capture(lambda: load_sampler_panel(failed_missing_metric, "panel", MetricName.MAX_RUN_FRAC))),
        "incomplete_shards_grid_and_count": {
            "accepted": True,
            "file_count": incomplete_panel.file_count,
            "pair_count": len(incomplete_panel.pairs),
            "declared_num_shards": 8,
            "declared_n_records": 999,
            "declared_grid_cells": 4,
        },
        "mismatched_pair_sets": asdict(_capture(lambda: summarize_sampler_pair(missing_pair, "loop", "control", MetricName.MAX_RUN_FRAC))),
        "malformed_schema": asdict(_capture(lambda: load_sampler_panel(malformed, "panel", MetricName.MAX_RUN_FRAC))),
        "boolean_seed_and_metric": {
            "accepted": True,
            "seed": boolean_panel.pairs[0].seed,
            "value": boolean_panel.pairs[0].value,
        },
    }
    print(json.dumps(outcomes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
