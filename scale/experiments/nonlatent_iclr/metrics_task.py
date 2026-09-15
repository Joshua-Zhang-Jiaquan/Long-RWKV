from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4

from .metric_records import summarize_sampler_cells
from .metrics import MetricInputError, MetricName, prompt_preserving_corruption
from .models import AuditResult
from .publication import UnsafePathError, write_once
from .service import AuditPaths

CORRECTIONS_RELATIVE = Path("DAN/nonlatent_iclr/metric_corrections.json")
REPORT_RELATIVE = Path("DAN/nonlatent_iclr/metric_corrections.md")
PAIRS = (("sampler_gate_m4_loop_s4750", "sampler_gate_m4_n2_s4000"), ("sampler_gate_m4_loop_s4750", "sampler_gate_m4_n2_s6000"), ("sampler_gate_m4_loop_s4750_reps2", "sampler_gate_m4_loop_s4750"), ("sampler_gate_m4_loop_s4750_reps3", "sampler_gate_m4_loop_s4750"), ("sampler_gate_m4_loop_s4750_reps4", "sampler_gate_m4_loop_s4750"))
METRICS = (MetricName.MASKED_TOKEN_ACCURACY, MetricName.MAX_RUN_FRAC, MetricName.TAU_A_COMMIT_ORDER, MetricName.DISTINCT_FRAC, MetricName.RESIDUE)


def _code_hash() -> str:
    digest = hashlib.sha256()
    for name in ("metrics.py", "metric_records.py", "metrics_task.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def _ci(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        raise MetricInputError("fewer than two paired observations")
    mean = sum(values) / len(values)
    return 1.96 * math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1) / len(values))


def _analysis(paths: AuditPaths) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for candidate, control in PAIRS:
        cells: dict[str, dict[str, str | list[dict[str, object]]]] = {}
        hashes: tuple[str, str] | None = None
        for metric in METRICS:
            loop, reference, _ = summarize_sampler_cells(paths.external_root, candidate, control, metric)
            hashes = (loop.sha256, reference.sha256)
            for arm in sorted({pair.arm for pair in loop.pairs}):
                left = tuple(pair.value for pair in loop.pairs if pair.arm == arm)
                right = tuple(pair.value for pair in reference.pairs if pair.arm == arm)
                delta = sum(left[index] - right[index] for index in range(len(left))) / len(left)
                direction = "diagnostic_only" if metric in {MetricName.TAU_A_COMMIT_ORDER, MetricName.DISTINCT_FRAC, MetricName.RESIDUE} else ("worse" if metric is MetricName.MAX_RUN_FRAC and delta > 1e-12 else "better" if metric is MetricName.MAX_RUN_FRAC and delta < -1e-12 else "neutral")
                if metric is MetricName.MASKED_TOKEN_ACCURACY:
                    direction = "better" if delta > 1e-12 else "worse" if delta < -1e-12 else "neutral"
                if arm not in cells:
                    cells[arm] = {"arm": arm, "metrics": []}
                entries = cells[arm]["metrics"]
                assert isinstance(entries, list)
                entries.append({"name": metric.value, "pair_count": len(left), "candidate_mean": sum(left) / len(left), "control_mean": sum(right) / len(right), "mean_delta": delta, "legacy_normal_ci95_half_width": _ci(tuple(left[index] - right[index] for index in range(len(left)))), "interpretation": direction, "ci_status": "descriptive_only_not_multiseed_confirmation"})
        assert hashes is not None
        comparisons.append({"candidate_panel": candidate, "control_panel": control, "candidate_sha256": hashes[0], "control_sha256": hashes[1], "cells": [cells[arm] for arm in sorted(cells)]})
    return comparisons


def _payload(paths: AuditPaths, attempt: str) -> bytes:
    data = {"schema_version": 2, "attempt": attempt, "date": "2026-09-11", "analysis_code_sha256": _code_hash(), "comparisons": _analysis(paths), "sequence_exact_match": "unavailable: historical decoded target/prediction sequences are absent", "baseline_nll": "causal force_forward=True NLL is separate from denoising metrics", "tau": "tau-a commit-order diagnostic only; no reference-agreement claim", "coverage": "all 20 sampler arms and 256 paired records per arm required", "legacy_ci": "95% paired normal CI is descriptive only, not multi-seed confirmation"}
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def _markdown(payload: bytes) -> bytes:
    raw = json.loads(payload)
    lines = ["# Metric corrections (2026-09-11)", "", "Corrected: positive max_run_frac is worse; em is masked-token accuracy; tau-a is diagnostic only.", "", "Sequence exact match: unavailable (historical sequences absent).", "", "| Comparison | Arm | Metric | Pairs | Delta | Interpretation |", "|---|---|---|---:|---:|---|"]
    for comparison in raw["comparisons"]:
        for cell in comparison["cells"]:
            for metric in cell["metrics"]:
                lines.append(f"| {comparison['candidate_panel']} - {comparison['control_panel']} | {cell['arm']} | {metric['name']} | {metric['pair_count']} | {metric['mean_delta']:.12g} | {metric['interpretation']} |")
    lines.extend(("", "Legacy normal CIs are descriptive only. No prompted-generation claim or fresh trials.", ""))
    return "\n".join(lines).encode()


def _receipt(attempt: str, payload: bytes) -> bytes:
    return (json.dumps({"schema_version": 2, "attempt": attempt, "payload_sha256": hashlib.sha256(payload).hexdigest(), "analysis_code_sha256": _code_hash(), "status": "METRICS_COMPLETE"}, indent=2, sort_keys=True) + "\n").encode()


def prepare_task_two(paths: AuditPaths) -> AuditResult:
    correction = paths.repo_root / CORRECTIONS_RELATIVE
    if correction.exists():
        try:
            payload = correction.read_bytes()
            raw = json.loads(payload)
            if raw != json.loads(_payload(paths, raw["attempt"])):
                return AuditResult("TAMPERED", "existing correction differs from current raw analysis")
            report = _markdown(payload)
            receipt = _receipt(raw["attempt"], payload)
            report_written = write_once(paths.repo_root, REPORT_RELATIVE, report)
            receipt_written = write_once(paths.evidence_root, Path("task-02") / raw["attempt"] / "prepare_receipt.json", receipt)
            if not report_written and (paths.repo_root / REPORT_RELATIVE).read_bytes() != report:
                return AuditResult("TAMPERED", "existing markdown conflicts with correction")
            receipt_path = paths.evidence_root / "task-02" / raw["attempt"] / "prepare_receipt.json"
            if not receipt_written and receipt_path.read_bytes() != receipt:
                return AuditResult("TAMPERED", "existing receipt conflicts with correction")
        except (KeyError, OSError, UnsafePathError, MetricInputError, json.JSONDecodeError):
            return AuditResult("TAMPERED", "existing correction cannot recover safely")
        return verify_task_two(paths)
    attempt = uuid4().hex
    try:
        payload = _payload(paths, attempt)
        report = _markdown(payload)
        receipt = _receipt(attempt, payload)
        if not write_once(paths.repo_root, CORRECTIONS_RELATIVE, payload):
            return AuditResult("PUBLICATION_EXISTS", "metric correction already exists")
        if not write_once(paths.repo_root, REPORT_RELATIVE, report):
            return AuditResult("PUBLICATION_PENDING", "correction exists without replacement report")
        if not write_once(paths.evidence_root, Path("task-02") / attempt / "prepare_receipt.json", receipt):
            return AuditResult("PUBLICATION_PENDING", "correction exists without terminal receipt")
    except (OSError, UnsafePathError, MetricInputError, json.JSONDecodeError):
        return AuditResult("SOURCE_UNAVAILABLE", "raw metric records are unavailable or invalid")
    return AuditResult("METRICS_COMPLETE", "per-cell raw-bound metric correction published")


def verify_task_two(paths: AuditPaths) -> AuditResult:
    try:
        path = paths.repo_root / CORRECTIONS_RELATIVE
        report = paths.repo_root / REPORT_RELATIVE
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        if raw.get("schema_version") != 2 or not isinstance(raw.get("attempt"), str) or raw.get("analysis_code_sha256") != _code_hash() or not report.is_file():
            return AuditResult("TAMPERED", "metric report version, code hash, or markdown is invalid")
        expected = json.loads(_payload(paths, raw["attempt"]))
        if raw != expected or report.read_bytes() != _markdown(raw_bytes):
            return AuditResult("STALE", "metric report differs from fresh per-cell analysis")
        receipt = paths.evidence_root / "task-02" / raw["attempt"] / "prepare_receipt.json"
        terminal = json.loads(receipt.read_text(encoding="utf-8"))
        if terminal != json.loads(_receipt(raw["attempt"], raw_bytes)):
            return AuditResult("TAMPERED", "terminal receipt is absent or invalid")
    except (KeyError, OSError, TypeError, json.JSONDecodeError, MetricInputError):
        return AuditResult("MALFORMED", "metric correction cannot be verified")
    return AuditResult("METRICS_COMPLETE", "per-cell correction, report, raw sources, and terminal receipt verify")


def analyze_task_two(paths: AuditPaths) -> AuditResult:
    result = verify_task_two(paths)
    return result if result.status != "METRICS_COMPLETE" else AuditResult("METRICS_COMPLETE", "comparisons=5; cells=100; metrics_per_cell=5")


def failure_probe_task_two() -> AuditResult:
    from .metrics import Pair, paired_delta
    polarity = paired_delta(MetricName.MAX_RUN_FRAC, (Pair("probe", 1, "r100_s1", 0.9),), (Pair("probe", 1, "r100_s1", 0.1),))
    if polarity.mean_delta != 0.8 or polarity.interpretation != "worse":
        return AuditResult("PROBE_FAILED", "finite max-run polarity control did not yield delta=0.8 worse")
    try:
        prompt_preserving_corruption((1, 2, 0), (True, True, False), (False, True, False), pad_id=0, mask_token_id=99)
        return AuditResult("PROBE_FAILED", "future-answer prompt leakage accepted")
    except MetricInputError:
        pass
    try:
        paired_delta(MetricName.MAX_RUN_FRAC, (Pair("x", 1, "r70_s1", math.nan),), (Pair("x", 1, "r70_s1", 0.0),))
        return AuditResult("PROBE_FAILED", "nonfinite value accepted")
    except MetricInputError:
        pass
    try:
        paired_delta(MetricName.MAX_RUN_FRAC, (Pair("x", 1, "r70_s1", 0.0),), (Pair("y", 1, "r70_s1", 0.0),))
        return AuditResult("PROBE_FAILED", "missing pair accepted")
    except MetricInputError:
        pass
    return AuditResult("EXPECTED_FAILURE_CONFIRMED", f"finite max-run delta={polarity.mean_delta:g} {polarity.interpretation}; nonfinite, missing-pair, and prompt-leakage guards rejected")
