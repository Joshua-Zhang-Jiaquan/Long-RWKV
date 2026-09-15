from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final


class EvidenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Record:
    document_id: str
    seed: int
    arm: str
    values: dict[str, float]


@dataclass(frozen=True, slots=True)
class Panel:
    name: str
    sha256: str
    records: tuple[Record, ...]
    decoded_sequence_records: int


ROOT: Final = Path(__file__).resolve().parents[5]
EXTERNAL: Final = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")
REPORT: Final = ROOT / "DAN/nonlatent_iclr/metric_corrections.json"
MARKDOWN: Final = ROOT / "DAN/nonlatent_iclr/metric_corrections.md"
PAIRS: Final = (
    ("sampler_gate_m4_loop_s4750", "sampler_gate_m4_n2_s4000"),
    ("sampler_gate_m4_loop_s4750", "sampler_gate_m4_n2_s6000"),
    ("sampler_gate_m4_loop_s4750_reps2", "sampler_gate_m4_loop_s4750"),
    ("sampler_gate_m4_loop_s4750_reps3", "sampler_gate_m4_loop_s4750"),
    ("sampler_gate_m4_loop_s4750_reps4", "sampler_gate_m4_loop_s4750"),
)
METRICS: Final = (
    ("masked_token_accuracy", "em"),
    ("max_run_frac", "max_run_frac"),
    ("tau_a_commit_order", "tau"),
    ("distinct_frac", "distinct_frac"),
    ("residue", "residue"),
)
EXPECTED_ARMS: Final = {
    f"r{ratio}_s{step}"
    for ratio in (70, 95, 99, 100)
    for step in (1, 8, 16, 32, 64)
}


def object_value(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvidenceError(f"{label} is not an object")
    return value


def list_value(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise EvidenceError(f"{label} is not a list")
    return value


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} is not finite numeric data")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError(f"{label} is not finite numeric data")
    return number


def load_panel(name: str) -> Panel:
    files = tuple(sorted((EXTERNAL / name).glob("sampler.shard*of*.json")))
    if len(files) != 8:
        raise EvidenceError(f"{name}: expected 8 shards, got {len(files)}")
    digest = hashlib.sha256()
    records: list[Record] = []
    indices: set[int] = set()
    canonical_grid: object | None = None
    decoded_count = 0
    for path in files:
        content = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(content)
        raw = object_value(json.loads(content), str(path))
        if raw.get("schema") != "qz_capability_sampler_shard_v1":
            raise EvidenceError(f"{name}: bad schema")
        index, count = raw.get("shard_index"), raw.get("num_shards")
        shard_records = list_value(raw.get("records"), f"{name}:records")
        if type(index) is not int or count != 8 or raw.get("n_records") != len(shard_records):
            raise EvidenceError(f"{name}: bad shard or declared record count")
        indices.add(index)
        grid = object_value(raw.get("grid"), f"{name}:grid")
        canonical_grid = grid if canonical_grid is None else canonical_grid
        if grid != canonical_grid:
            raise EvidenceError(f"{name}: inconsistent grid")
        for raw_record in shard_records:
            item = object_value(raw_record, f"{name}:record")
            if item.get("failure") is not None:
                raise EvidenceError(f"{name}: failed record in completed panel")
            document_id, seed, arm = item.get("document_id"), item.get("seed"), item.get("arm")
            if not isinstance(document_id, str) or type(seed) is not int or not isinstance(arm, str):
                raise EvidenceError(f"{name}: invalid identity")
            raw_metrics = object_value(item.get("metrics"), f"{name}:metrics")
            values = {field: finite_number(raw_metrics.get(field), f"{name}:{field}") for _, field in METRICS}
            records.append(Record(document_id, seed, arm, values))
            decoded_count += int("target_tokens" in item and "prediction_tokens" in item)
    if indices != set(range(8)):
        raise EvidenceError(f"{name}: incomplete shard indices")
    grid = object_value(canonical_grid, f"{name}:grid")
    ratios = list_value(grid.get("mask_ratios"), f"{name}:ratios")
    steps = list_value(grid.get("steps"), f"{name}:steps")
    grid_arms = {f"r{round(finite_number(ratio, name) * 100)}_s{int(finite_number(step, name))}" for ratio in ratios for step in steps}
    keys = {(record.document_id, record.seed, record.arm) for record in records}
    arm_counts = {arm: sum(record.arm == arm for record in records) for arm in EXPECTED_ARMS}
    if grid_arms != EXPECTED_ARMS or len(keys) != len(records) or set(arm_counts.values()) != {256}:
        raise EvidenceError(f"{name}: duplicate or incomplete 20x256 coverage")
    return Panel(name, digest.hexdigest(), tuple(records), decoded_count)


def expected_interpretation(metric: str, delta: float) -> str:
    if metric in {"tau_a_commit_order", "distinct_frac", "residue"}:
        return "diagnostic_only"
    if abs(delta) <= 1e-12:
        return "neutral"
    if metric == "max_run_frac":
        return "worse" if delta > 0 else "better"
    return "better" if delta > 0 else "worse"


def code_hash() -> str:
    digest = hashlib.sha256()
    source = ROOT / "scale/experiments/nonlatent_iclr"
    for name in ("metrics.py", "metric_records.py", "metrics_task.py"):
        digest.update((source / name).read_bytes())
    return digest.hexdigest()


def main() -> None:
    report_bytes = REPORT.read_bytes()
    report = object_value(json.loads(report_bytes), "report")
    comparisons = list_value(report.get("comparisons"), "comparisons")
    names = tuple(sorted({name for pair in PAIRS for name in pair}))
    panels = {name: load_panel(name) for name in names}
    reported_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for raw_comparison in comparisons:
        comparison = object_value(raw_comparison, "comparison")
        key = (str(comparison.get("candidate_panel")), str(comparison.get("control_panel")))
        if key in reported_by_pair:
            raise EvidenceError(f"duplicate report comparison: {key}")
        reported_by_pair[key] = comparison
    rows: list[dict[str, object]] = []
    comparison_controls: list[dict[str, object]] = []
    numeric_field_count = 0
    exact_numeric_matches = 0
    max_abs_error = 0.0
    for candidate_name, control_name in PAIRS:
        candidate, control = panels[candidate_name], panels[control_name]
        reported = reported_by_pair[(candidate_name, control_name)]
        cells = {str(cell["arm"]): cell for cell in (object_value(item, "cell") for item in list_value(reported.get("cells"), "cells"))}
        hashes_match = reported.get("candidate_sha256") == candidate.sha256 and reported.get("control_sha256") == control.sha256
        ordered_keys_match = True
        for arm in sorted(EXPECTED_ARMS):
            left = tuple(record for record in candidate.records if record.arm == arm)
            right = tuple(record for record in control.records if record.arm == arm)
            left_keys = tuple((item.document_id, item.seed) for item in left)
            right_keys = tuple((item.document_id, item.seed) for item in right)
            if set(left_keys) != set(right_keys):
                raise EvidenceError(f"pair mismatch: {candidate_name}:{control_name}:{arm}")
            ordered_keys_match = ordered_keys_match and left_keys == right_keys
            control_by_key = {(item.document_id, item.seed): item for item in right}
            cell = object_value(cells[arm], f"cell:{arm}")
            reported_metrics = {str(item["name"]): item for item in (object_value(value, "metric") for value in list_value(cell.get("metrics"), "metrics"))}
            for metric, raw_field in METRICS:
                left_values = tuple(item.values[raw_field] for item in left)
                right_values = tuple(control_by_key[(item.document_id, item.seed)].values[raw_field] for item in left)
                deltas = tuple(left_values[index] - right_values[index] for index in range(len(left_values)))
                mean_delta = sum(deltas) / len(deltas)
                variance = sum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
                computed = {
                    "candidate_mean": sum(left_values) / len(left_values),
                    "control_mean": sum(right_values) / len(right_values),
                    "mean_delta": mean_delta,
                    "legacy_normal_ci95_half_width": 1.96 * math.sqrt(variance / len(deltas)),
                }
                published = object_value(reported_metrics[metric], f"reported:{metric}")
                errors = {field: abs(value - finite_number(published.get(field), field)) for field, value in computed.items()}
                exact = {field: value == published.get(field) for field, value in computed.items()}
                numeric_field_count += len(exact)
                exact_numeric_matches += sum(exact.values())
                max_abs_error = max(max_abs_error, *errors.values())
                metadata_match = (
                    published.get("pair_count") == 256
                    and published.get("interpretation") == expected_interpretation(metric, mean_delta)
                    and published.get("ci_status") == "descriptive_only_not_multiseed_confirmation"
                )
                rows.append({
                    "candidate_panel": candidate_name, "control_panel": control_name,
                    "arm": arm, "metric": metric, "computed": computed,
                    "reported": {field: published.get(field) for field in computed},
                    "exact_numeric_match": exact, "absolute_error": errors,
                    "metadata_match": metadata_match,
                })
        comparison_controls.append({
            "candidate_panel": candidate_name, "control_panel": control_name,
            "cell_count": len(cells), "hashes_match": hashes_match,
            "ordered_pair_keys_match": ordered_keys_match,
        })
    attempt = report.get("attempt")
    receipt_path = ROOT / ".omo/evidence/nonlatent-rnn-dlm-iclr-research/task-02" / str(attempt) / "prepare_receipt.json"
    receipt = object_value(json.loads(receipt_path.read_bytes()), "receipt")
    computed_code_hash = code_hash()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    row_mismatches = sum(
        not row["metadata_match"] or not all(object_value(row["exact_numeric_match"], "exact").values())
        for row in rows
    )
    metadata_checks = {
        "schema_version_2": report.get("schema_version") == 2,
        "comparison_set_exact": set(reported_by_pair) == set(PAIRS),
        "analysis_code_hash_match": report.get("analysis_code_sha256") == computed_code_hash,
        "sequence_exact_match_correctly_unavailable": report.get("sequence_exact_match") == "unavailable: historical decoded target/prediction sequences are absent" and sum(panel.decoded_sequence_records for panel in panels.values()) == 0,
        "receipt_exact": receipt == {
            "schema_version": 2, "attempt": attempt, "payload_sha256": report_sha,
            "analysis_code_sha256": computed_code_hash, "status": "METRICS_COMPLETE",
        },
    }
    summary = {
        "overall_pass": len(rows) == 500 and row_mismatches == 0 and all(metadata_checks.values()) and all(item["cell_count"] == 20 and item["hashes_match"] and item["ordered_pair_keys_match"] for item in comparison_controls),
        "comparison_count": len(comparison_controls), "cell_count": len(rows) // len(METRICS),
        "metric_row_count": len(rows), "raw_paired_value_count": len(rows) * 256,
        "numeric_field_count": numeric_field_count, "exact_numeric_match_count": exact_numeric_matches,
        "metric_row_mismatch_count": row_mismatches, "max_absolute_error": max_abs_error,
        "report_sha256": report_sha, "markdown_sha256": hashlib.sha256(MARKDOWN.read_bytes()).hexdigest(),
        "analysis_code_sha256": computed_code_hash, "metadata_checks": metadata_checks,
        "panels": [{"name": panel.name, "sha256": panel.sha256, "file_count": 8, "record_count": len(panel.records), "arm_count": 20, "records_per_arm": 256, "decoded_sequence_records": panel.decoded_sequence_records} for panel in panels.values()],
        "comparisons": comparison_controls,
    }
    output = {"summary": summary, "rows": rows}
    output_path = Path(__file__).with_name("raw-comparison-controls.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
