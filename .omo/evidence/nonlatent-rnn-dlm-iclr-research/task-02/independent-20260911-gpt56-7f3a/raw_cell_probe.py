#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic==2.11.5"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly using an already cached dependency set:
#      uv run --offline raw_cell_probe.py
# 3. Or use the repository interpreter without writing caches:
#      PYTHONDONTWRITEBYTECODE=1 python raw_cell_probe.py
# ──────────────────

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Final, assert_never

from raw_cell_models import (
    CellControl,
    ComparisonControl,
    ComparisonInputs,
    CorrectionReport,
    EvidenceError,
    MetricControl,
    MetricField,
    Panel,
    PanelControl,
    ProbeOutput,
    RawRecord,
    RawShard,
    RecordKey,
    ReportCoverage,
    ReportedCorrection,
    Values,
)


EXTERNAL_ROOT: Final = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")
REPO_ROOT: Final = Path(__file__).parents[5]
REPORT_PATH: Final = REPO_ROOT / "DAN/nonlatent_iclr/metric_corrections.json"
ARM_PATTERN: Final = re.compile(r"^r(?P<ratio>70|95|99|100)_s(?P<steps>1|8|16|32|64)$")
COMPARISONS: Final = (
    ("sampler_gate_m4_loop_s4750", "sampler_gate_m4_n2_s4000"),
    ("sampler_gate_m4_loop_s4750", "sampler_gate_m4_n2_s6000"),
    ("sampler_gate_m4_loop_s4750_reps2", "sampler_gate_m4_loop_s4750"),
    ("sampler_gate_m4_loop_s4750_reps3", "sampler_gate_m4_loop_s4750"),
    ("sampler_gate_m4_loop_s4750_reps4", "sampler_gate_m4_loop_s4750"),
)


def _load_panel(name: str) -> Panel:
    files = tuple(sorted((EXTERNAL_ROOT / name).glob("sampler.shard*of*.json")))
    if not files:
        raise EvidenceError(f"missing panel: {name}")
    digest = hashlib.sha256()
    shards: list[RawShard] = []
    for path in files:
        content = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(content)
        shards.append(RawShard.model_validate_json(content))
    expected_indices = set(range(shards[0].num_shards))
    actual_indices = {shard.shard_index for shard in shards}
    if actual_indices != expected_indices or any(shard.num_shards != len(files) for shard in shards):
        raise EvidenceError(f"incomplete shard set: {name}")
    if any(shard.n_records != len(shard.records) for shard in shards):
        raise EvidenceError(f"record count mismatch: {name}")
    records = tuple(record for shard in shards for record in shard.records)
    keys: set[RecordKey] = set()
    for record in records:
        key = (record.document_id, record.seed, record.arm)
        if key in keys:
            raise EvidenceError(f"duplicate record: {name}:{key}")
        keys.add(key)
    return Panel(name, digest.hexdigest(), len(files), records, shards[0].grid)


def _repetitions(name: str) -> int:
    match re.search(r"_reps(?P<count>[234])$", name):
        case None:
            return 1
        case match:
            return int(match.group("count"))


def _corrected_name(metric: MetricField) -> str:
    match metric:
        case MetricField.MAX_RUN_FRAC:
            return "max_run_frac"
        case MetricField.MASKED_TOKEN_ACCURACY:
            return "masked_token_accuracy"
        case MetricField.TAU_A_COMMIT_ORDER:
            return "tau_a_commit_order_diagnostic"
        case MetricField.DISTINCT_FRAC:
            return "distinct_frac_diagnostic"
        case MetricField.RESIDUE:
            return "residue_diagnostic"
        case unreachable:
            assert_never(unreachable)


def _metric_value(record: RawRecord, metric: MetricField) -> float:
    match metric:
        case MetricField.MAX_RUN_FRAC:
            return record.metrics.max_run_frac
        case MetricField.MASKED_TOKEN_ACCURACY:
            return record.metrics.em
        case MetricField.TAU_A_COMMIT_ORDER:
            return record.metrics.tau
        case MetricField.DISTINCT_FRAC:
            return record.metrics.distinct_frac
        case MetricField.RESIDUE:
            return record.metrics.residue
        case unreachable:
            assert_never(unreachable)


def _values(panel: Panel, arm: str, metric: MetricField) -> Values:
    return {
        (record.document_id, record.seed): _metric_value(record, metric)
        for record in panel.records
        if record.arm == arm
    }


def _metric_control(inputs: ComparisonInputs, arm: str, metric: MetricField) -> MetricControl:
    candidate = _values(inputs.candidate, arm, metric)
    control = _values(inputs.control, arm, metric)
    if candidate.keys() != control.keys() or not candidate:
        raise EvidenceError(f"pair mismatch: {inputs.candidate.name}:{inputs.control.name}:{arm}:{metric}")
    deltas = tuple(candidate[key] - control[key] for key in sorted(candidate))
    mean_delta = math.fsum(deltas) / len(deltas)
    variance = math.fsum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
    return MetricControl(
        corrected_name=_corrected_name(metric), raw_field=metric.value, pair_count=len(deltas),
        candidate_mean=math.fsum(candidate.values()) / len(candidate),
        control_mean=math.fsum(control.values()) / len(control), mean_delta=mean_delta,
        historical_normal_ci95_half_width=1.96 * math.sqrt(variance / len(deltas)),
        ci_status="descriptive_only_not_multiseed_confirmation",
    )


def _cell(inputs: ComparisonInputs, arm: str) -> CellControl:
    match ARM_PATTERN.fullmatch(arm):
        case None:
            raise EvidenceError(f"invalid arm: {arm}")
        case arm_match:
            return CellControl(
                arm=arm, mask_ratio=int(arm_match.group("ratio")) / 100,
                steps=int(arm_match.group("steps")),
                candidate_repetitions=_repetitions(inputs.candidate.name),
                control_repetitions=_repetitions(inputs.control.name),
                metrics=tuple(_metric_control(inputs, arm, metric) for metric in MetricField),
            )


def _reported(report: CorrectionReport, names: tuple[str, str]) -> ReportedCorrection:
    candidate_name, control_name = names
    matches = tuple(item for item in report.corrections if item.loop_panel == candidate_name and item.control_panel == control_name)
    if len(matches) != 1:
        raise EvidenceError(f"reported comparison count is {len(matches)}: {names}")
    return matches[0]


def _comparison(inputs: ComparisonInputs, reported: ReportedCorrection) -> ComparisonControl:
    arms = tuple(sorted({record.arm for record in inputs.candidate.records}))
    if set(arms) != {record.arm for record in inputs.control.records}:
        raise EvidenceError(f"arm coverage mismatch: {inputs.candidate.name}:{inputs.control.name}")
    cells = tuple(_cell(inputs, arm) for arm in arms)
    max_run_controls = tuple(cell.metrics[0] for cell in cells)
    all_candidate = _values(inputs.candidate, "", MetricField.MAX_RUN_FRAC)
    all_control = _values(inputs.control, "", MetricField.MAX_RUN_FRAC)
    candidate_by_key = {(record.document_id, record.seed, record.arm): record.metrics.max_run_frac for record in inputs.candidate.records}
    control_by_key = {(record.document_id, record.seed, record.arm): record.metrics.max_run_frac for record in inputs.control.records}
    if all_candidate or all_control or candidate_by_key.keys() != control_by_key.keys():
        raise EvidenceError("unexpected pooled-key state")
    deltas = tuple(candidate_by_key[key] - control_by_key[key] for key in sorted(candidate_by_key))
    pooled = math.fsum(deltas) / len(deltas)
    return ComparisonControl(
        candidate_panel=inputs.candidate.name, control_panel=inputs.control.name,
        cell_count=len(cells), raw_pair_count=len(deltas), reported_pair_count=reported.pair_count,
        independently_computed_pooled_max_run_delta=pooled,
        reported_pooled_max_run_delta=reported.max_run_delta,
        pooled_delta_matches_raw=math.isclose(pooled, reported.max_run_delta, rel_tol=0.0, abs_tol=1e-15),
        panel_hashes_match_report=(inputs.candidate.digest == reported.loop_sha256 and inputs.control.digest == reported.control_sha256),
        max_run_positive_cell_count=sum(item.mean_delta > 0 for item in max_run_controls),
        max_run_negative_cell_count=sum(item.mean_delta < 0 for item in max_run_controls),
        heterogeneous_cell_directions_hidden_by_pooling=(any(item.mean_delta > 0 for item in max_run_controls) and any(item.mean_delta < 0 for item in max_run_controls)),
        cells=cells,
    )


def _panel_control(panel: Panel) -> PanelControl:
    arms = {record.arm for record in panel.records}
    expected_arms = {f"r{int(ratio * 100)}_s{steps}" for ratio in panel.grid.mask_ratios for steps in panel.grid.steps}
    counts = {arm: sum(record.arm == arm for record in panel.records) for arm in arms}
    return PanelControl(
        name=panel.name, sha256=panel.digest, file_count=panel.file_count,
        record_count=len(panel.records), arm_count=len(arms),
        records_per_arm=min(counts.values()), failure_count=sum(record.failure is not None for record in panel.records),
        complete_shard_indices=True, complete_grid_arms=arms == expected_arms and len(set(counts.values())) == 1,
    )


def main() -> None:
    report = CorrectionReport.model_validate_json(REPORT_PATH.read_bytes())
    panel_names = tuple(sorted({name for pair in COMPARISONS for name in pair}))
    panels = {name: _load_panel(name) for name in panel_names}
    comparisons = tuple(
        _comparison(ComparisonInputs(panels[names[0]], panels[names[1]]), _reported(report, names))
        for names in COMPARISONS
    )
    raw_records = tuple(record for panel in panels.values() for record in panel.records)
    output = ProbeOutput(
        panels=tuple(_panel_control(panels[name]) for name in panel_names), comparisons=comparisons,
        coverage=ReportCoverage(
            saved_report_has_per_cell_rows=False,
            saved_report_has_masked_token_accuracy_values=False,
            saved_report_has_tau_values=False,
            saved_report_has_diagnostic_values=False,
            historical_ci_clearly_labeled_descriptive=("descriptive only" in report.legacy_ci and "not multi-seed confirmation" in report.legacy_ci),
            decoded_sequences_available_in_raw_records=all(record.target_tokens is not None and record.prediction_tokens is not None for record in raw_records),
            whole_sequence_exact_match_reported=False,
        ),
    )
    output_path = Path(__file__).with_name("raw-cell-controls.json")
    output_path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(output.model_dump_json(include={"schema_version", "coverage"}, indent=2))
    print(f"panels={len(output.panels)} comparisons={len(output.comparisons)} cells={sum(item.cell_count for item in output.comparisons)} output={output_path}")


if __name__ == "__main__":
    main()
