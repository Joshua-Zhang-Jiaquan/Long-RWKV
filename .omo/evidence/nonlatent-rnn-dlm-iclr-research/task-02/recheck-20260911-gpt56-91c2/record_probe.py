from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TypeVar

metric_records_module = import_module("scale.experiments.nonlatent_iclr.metric_records")
metrics_module = import_module("scale.experiments.nonlatent_iclr.metrics")
load_sampler_panel = metric_records_module.load_sampler_panel
summarize_sampler_cells = metric_records_module.summarize_sampler_cells
MetricInputError = metrics_module.MetricInputError
MetricName = metrics_module.MetricName
Pair = metrics_module.Pair
paired_delta = metrics_module.paired_delta


Result = TypeVar("Result")


def capture(action: Callable[[], Result]) -> dict[str, bool | str]:
    try:
        detail = action()
    except MetricInputError as error:
        return {"accepted": False, "detail": str(error)}
    return {"accepted": True, "detail": str(detail)}


def record(index: int) -> dict[str, object]:
    return {
        "document_id": f"doc-{index}",
        "seed": 42,
        "arm": "r70_s1",
        "failure": None,
        "metrics": {"max_run_frac": 0.2 + index / 10_000},
    }


def write_panel(
    root: Path,
    name: str,
    *,
    n_records: int | None = None,
    grid: dict[str, list[int | float | bool]] | None = None,
    mutate: Callable[[dict[str, object], int], None] | None = None,
    shard_limit: int = 8,
    schema: str = "qz_capability_sampler_shard_v1",
    omit_grid: bool = False,
) -> None:
    panel_dir = root / name
    panel_dir.mkdir(parents=True)
    panel_grid = grid if grid is not None else {"mask_ratios": [0.7], "steps": [1]}
    for shard_index in range(shard_limit):
        records = [record(shard_index + 8 * offset) for offset in range(32)]
        if mutate is not None:
            for item in records:
                mutate(item, int(str(item["document_id"]).split("-")[1]))
        payload: dict[str, object] = {
            "schema": schema,
            "num_shards": 8,
            "shard_index": shard_index,
            "n_records": len(records) if n_records is None else n_records,
            "records": records,
        }
        if not omit_grid:
            payload["grid"] = panel_grid
        path = panel_dir / f"sampler.shard{shard_index}of8.json"
        path.write_text(json.dumps(payload, allow_nan=True) + "\n", encoding="utf-8")


def load_summary(root: Path, name: str) -> str:
    panel = load_sampler_panel(root, name, MetricName.MAX_RUN_FRAC)
    return f"file_count={panel.file_count},pair_count={len(panel.pairs)}"


def cell_summary(root: Path, left: str, right: str) -> str:
    _, _, cells = summarize_sampler_cells(root, left, right, MetricName.MAX_RUN_FRAC)
    return f"cell_count={len(cells)}"


def mutate_seed_bool(item: dict[str, object], index: int) -> None:
    if index == 0:
        item["seed"] = True


def mutate_metric_bool(item: dict[str, object], index: int) -> None:
    if index == 0:
        item["metrics"] = {"max_run_frac": True}


def mutate_failed(item: dict[str, object], index: int) -> None:
    if index == 0:
        item["failure"] = "OOM"


def mutate_failed_missing(item: dict[str, object], index: int) -> None:
    if index == 0:
        item["failure"] = "OOM"
        item["metrics"] = {}


def mutate_nonfinite(item: dict[str, object], index: int) -> None:
    if index == 0:
        item["metrics"] = {"max_run_frac": float("nan")}


def mutate_duplicate(item: dict[str, object], index: int) -> None:
    if index == 0:
        item["document_id"] = "doc-8"


def mutate_missing_pair(item: dict[str, object], index: int) -> None:
    if index == 0:
        item["document_id"] = "different-doc"


def main() -> None:
    temporary_path = ""
    with tempfile.TemporaryDirectory(prefix="task02-record-recheck-", dir="/tmp/opencode") as temporary:
        root = Path(temporary)
        temporary_path = temporary
        write_panel(root, "valid")
        write_panel(root, "bad-count", n_records=999)
        write_panel(root, "bad-seed-bool", mutate=mutate_seed_bool)
        write_panel(root, "bad-metric-bool", mutate=mutate_metric_bool)
        write_panel(root, "failed-finite", mutate=mutate_failed)
        write_panel(root, "failed-missing", mutate=mutate_failed_missing)
        write_panel(root, "nonfinite", mutate=mutate_nonfinite)
        write_panel(root, "duplicate", mutate=mutate_duplicate)
        write_panel(root, "missing-pair", mutate=mutate_missing_pair)
        write_panel(root, "control")
        write_panel(root, "incomplete-shards", shard_limit=1)
        write_panel(
            root,
            "incomplete-grid",
            grid={"mask_ratios": [0.3, 0.5, 0.7, 0.85, 1.0], "steps": [1, 2, 4, 8]},
        )
        write_panel(root, "bool-grid", grid={"mask_ratios": [True], "steps": [1]})
        write_panel(root, "missing-grid", omit_grid=True)
        write_panel(root, "bad-schema", schema="wrong")

        outcomes = {
            "valid_full_arm": capture(lambda: load_summary(root, "valid")),
            "missing_panel": capture(lambda: load_summary(root, "absent")),
            "malformed_schema": capture(lambda: load_summary(root, "bad-schema")),
            "boolean_seed": capture(lambda: load_summary(root, "bad-seed-bool")),
            "boolean_metric": capture(lambda: load_summary(root, "bad-metric-bool")),
            "boolean_grid": capture(lambda: load_summary(root, "bool-grid")),
            "failed_record_with_finite_metric": capture(lambda: load_summary(root, "failed-finite")),
            "failed_record_without_metric": capture(lambda: load_summary(root, "failed-missing")),
            "incomplete_shards": capture(lambda: load_summary(root, "incomplete-shards")),
            "incomplete_declared_grid": capture(lambda: load_summary(root, "incomplete-grid")),
            "missing_grid": capture(lambda: load_summary(root, "missing-grid")),
            "declared_n_records_mismatch": capture(lambda: load_summary(root, "bad-count")),
            "duplicate_pair": capture(lambda: cell_summary(root, "duplicate", "control")),
            "mismatched_pair_sets": capture(lambda: cell_summary(root, "valid", "missing-pair")),
            "nonfinite_pair": capture(lambda: cell_summary(root, "nonfinite", "control")),
        }
        expected_acceptance = {
            "valid_full_arm": True,
            "missing_panel": False,
            "malformed_schema": False,
            "boolean_seed": False,
            "boolean_metric": False,
            "boolean_grid": False,
            "failed_record_with_finite_metric": False,
            "failed_record_without_metric": False,
            "incomplete_shards": False,
            "incomplete_declared_grid": False,
            "missing_grid": False,
            "declared_n_records_mismatch": False,
            "duplicate_pair": False,
            "mismatched_pair_sets": False,
            "nonfinite_pair": False,
        }
        checks = {
            name: outcome["accepted"] is expected_acceptance[name]
            for name, outcome in outcomes.items()
        }
        polarity = paired_delta(
            MetricName.MAX_RUN_FRAC,
            (Pair("doc", 42, "r70_s1", 0.9),),
            (Pair("doc", 42, "r70_s1", 0.1),),
        )
        checks["positive_max_run_delta_is_worse"] = polarity.interpretation == "worse"
        output: dict[str, object] = {
            "checks": checks,
            "overall_pass": all(checks.values()),
            "outcomes": outcomes,
            "polarity": {
                "mean_delta": polarity.mean_delta,
                "interpretation": polarity.interpretation,
            },
        }
    output["cleanup"] = {
        "temporary_path": temporary_path,
        "exists_after_context": Path(temporary_path).exists(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
