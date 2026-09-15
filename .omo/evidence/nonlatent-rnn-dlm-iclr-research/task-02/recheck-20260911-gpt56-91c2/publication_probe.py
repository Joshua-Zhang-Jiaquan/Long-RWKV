from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol

metrics_task = import_module("scale.experiments.nonlatent_iclr.metrics_task")
service_module = import_module("scale.experiments.nonlatent_iclr.service")
AuditPathsFactory = service_module.AuditPaths


class Paths(Protocol):
    repo_root: Path
    evidence_root: Path
    external_root: Path


ROOT = Path(__file__).resolve().parents[5]
EXTERNAL = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")
SOURCE_JSON = ROOT / metrics_task.CORRECTIONS_RELATIVE
SOURCE_MARKDOWN = ROOT / metrics_task.REPORT_RELATIVE
SOURCE_REPORT = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
SOURCE_ATTEMPT = str(SOURCE_REPORT["attempt"])
SOURCE_RECEIPT = (
    ROOT
    / ".omo/evidence/nonlatent-rnn-dlm-iclr-research/task-02"
    / SOURCE_ATTEMPT
    / "prepare_receipt.json"
)


def paths(case: Path) -> Paths:
    return AuditPathsFactory.from_roots(
        case / "repo",
        case / "evidence",
        external_root=EXTERNAL,
        live_root=case / "live",
        staged_root=case / "staged",
    )


def clone_case(root: Path, name: str, *, markdown: bool = True, receipt: bool = True) -> Paths:
    case_paths = paths(root / name)
    json_path = case_paths.repo_root / metrics_task.CORRECTIONS_RELATIVE
    json_path.parent.mkdir(parents=True)
    shutil.copy2(SOURCE_JSON, json_path)
    if markdown:
        shutil.copy2(SOURCE_MARKDOWN, case_paths.repo_root / metrics_task.REPORT_RELATIVE)
    if receipt:
        receipt_path = case_paths.evidence_root / "task-02" / SOURCE_ATTEMPT / "prepare_receipt.json"
        receipt_path.parent.mkdir(parents=True)
        shutil.copy2(SOURCE_RECEIPT, receipt_path)
    return case_paths


def result(case_paths: Paths) -> dict[str, str]:
    verified = metrics_task.verify_task_two(case_paths)
    return {"status": verified.status, "detail": verified.detail}


def outcome_field(outcomes: dict[str, object], name: str, field: str) -> object:
    outcome = outcomes[name]
    if not isinstance(outcome, dict):
        raise TypeError(f"{name} outcome is not an object")
    return outcome[field]


def mutate_json(case_paths: Paths, action: Callable[[dict[str, object]], None]) -> None:
    report_path = case_paths.repo_root / metrics_task.CORRECTIONS_RELATIVE
    report = json.loads(report_path.read_text(encoding="utf-8"))
    action(report)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def first_metric(report: dict[str, object]) -> dict[str, object]:
    comparisons = report["comparisons"]
    assert isinstance(comparisons, list)
    comparison = comparisons[0]
    assert isinstance(comparison, dict)
    cells = comparison["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    metrics = cell["metrics"]
    assert isinstance(metrics, list)
    metric = metrics[0]
    assert isinstance(metric, dict)
    return metric


def change_delta(report: dict[str, object]) -> None:
    first_metric(report)["mean_delta"] = 12345.0


def change_pair_count(report: dict[str, object]) -> None:
    first_metric(report)["pair_count"] = 999


def empty_comparisons(report: dict[str, object]) -> None:
    report["comparisons"] = []


def change_version(report: dict[str, object]) -> None:
    report["schema_version"] = 999


def change_code_hash(report: dict[str, object]) -> None:
    report["analysis_code_sha256"] = "0" * 64


def tamper_receipt(case_paths: Paths) -> None:
    path = case_paths.evidence_root / "task-02" / SOURCE_ATTEMPT / "prepare_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["payload_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def interruption(root: Path, name: str, selected: str) -> dict[str, object]:
    case_paths = paths(root / name)
    original = metrics_task.write_once

    def fail_selected(root_path: Path, relative: Path, data: bytes) -> bool:
        should_fail = (
            selected == "json" and root_path == case_paths.repo_root and relative == metrics_task.CORRECTIONS_RELATIVE
        ) or (
            selected == "markdown" and root_path == case_paths.repo_root and relative == metrics_task.REPORT_RELATIVE
        ) or (selected == "receipt" and root_path == case_paths.evidence_root)
        if should_fail:
            raise OSError(5, f"injected {selected} interruption")
        return original(root_path, relative, data)

    setattr(metrics_task, "write_once", fail_selected)
    try:
        first = metrics_task.prepare_task_two(case_paths)
    finally:
        setattr(metrics_task, "write_once", original)
    retry = metrics_task.prepare_task_two(case_paths)
    verified = metrics_task.verify_task_two(case_paths)
    report_files = tuple(case_paths.repo_root.glob("DAN/nonlatent_iclr/*"))
    receipts = tuple(case_paths.evidence_root.glob("task-02/*/prepare_receipt.json"))
    return {
        "first_prepare": first.status,
        "retry_prepare": retry.status,
        "verify": verified.status,
        "report_files": sorted(path.name for path in report_files),
        "receipt_count": len(receipts),
        "recovered": retry.status == "METRICS_COMPLETE" and verified.status == "METRICS_COMPLETE",
    }


def main() -> None:
    temporary_path = ""
    with tempfile.TemporaryDirectory(prefix="task02-publication-recheck-", dir="/tmp/opencode") as temporary:
        temporary_path = temporary
        root = Path(temporary)
        outcomes: dict[str, object] = {}
        outcomes["happy_current"] = result(clone_case(root, "happy"))

        for name, mutation in (
            ("tampered_derived_delta", change_delta),
            ("tampered_pair_count", change_pair_count),
            ("empty_comparisons", empty_comparisons),
            ("invalid_schema_version", change_version),
            ("changed_analysis_code_hash", change_code_hash),
        ):
            case_paths = clone_case(root, name)
            mutate_json(case_paths, mutation)
            outcomes[name] = result(case_paths)

        missing_receipt = clone_case(root, "missing-receipt", receipt=False)
        outcomes["missing_receipt"] = result(missing_receipt)

        receipt_tamper = clone_case(root, "tampered-receipt")
        tamper_receipt(receipt_tamper)
        outcomes["tampered_receipt"] = result(receipt_tamper)

        missing_markdown = clone_case(root, "missing-markdown", markdown=False)
        outcomes["missing_markdown"] = result(missing_markdown)

        markdown_tamper = clone_case(root, "tampered-markdown")
        markdown_path = markdown_tamper.repo_root / metrics_task.REPORT_RELATIVE
        markdown_path.write_text(markdown_path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
        outcomes["tampered_markdown"] = result(markdown_tamper)

        outcomes["json_write_interruption"] = interruption(root, "interrupt-json", "json")
        outcomes["markdown_write_interruption"] = interruption(root, "interrupt-markdown", "markdown")
        outcomes["receipt_write_interruption"] = interruption(root, "interrupt-receipt", "receipt")

        checks = {
            "happy_current": outcome_field(outcomes, "happy_current", "status") == "METRICS_COMPLETE",
            **{
                name: outcome_field(outcomes, name, "status") != "METRICS_COMPLETE"
                for name in (
                    "tampered_derived_delta", "tampered_pair_count", "empty_comparisons",
                    "invalid_schema_version", "changed_analysis_code_hash", "missing_receipt",
                    "tampered_receipt", "missing_markdown", "tampered_markdown",
                )
            },
            "json_write_interruption_recovers": outcome_field(outcomes, "json_write_interruption", "recovered") is True,
            "markdown_write_interruption_recovers": outcome_field(outcomes, "markdown_write_interruption", "recovered") is True,
            "receipt_write_interruption_recovers": outcome_field(outcomes, "receipt_write_interruption", "recovered") is True,
        }
        output = {"checks": checks, "overall_pass": all(checks.values()), "outcomes": outcomes}
    output["cleanup"] = {"temporary_path": temporary_path, "exists_after_context": Path(temporary_path).exists()}
    output_path = Path(__file__).with_name("publication-controls.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
