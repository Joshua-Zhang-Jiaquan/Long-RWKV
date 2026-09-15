from __future__ import annotations

import hashlib
import json
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Protocol


metric_records = import_module("scale.experiments.nonlatent_iclr.metric_records")
metrics = import_module("scale.experiments.nonlatent_iclr.metrics")
metrics_task = import_module("scale.experiments.nonlatent_iclr.metrics_task")
service = import_module("scale.experiments.nonlatent_iclr.service")
AuditPathsFactory = service.AuditPaths
MetricInputError = metrics.MetricInputError
MetricName = metrics.MetricName


class Paths(Protocol):
    repo_root: Path
    evidence_root: Path
    external_root: Path


EXTERNAL = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")


def make_paths(case: Path, *, external: Path = EXTERNAL) -> Paths:
    return AuditPathsFactory.from_roots(
        case / "repo",
        case / "evidence",
        external_root=external,
        live_root=case / "live",
        staged_root=case / "staged",
    )


def write_panel(root: Path, name: str, declared_records: int | bool) -> None:
    panel = root / name
    panel.mkdir(parents=True)
    for shard_index in range(8):
        records = [
            {
                "document_id": f"doc-{shard_index + 8 * offset}",
                "seed": 42,
                "arm": "r70_s1",
                "failure": None,
                "metrics": {"max_run_frac": 0.25},
            }
            for offset in range(32)
        ]
        payload = {
            "schema": "qz_capability_sampler_shard_v1",
            "num_shards": 8,
            "shard_index": shard_index,
            "n_records": declared_records,
            "grid": {"mask_ratios": [0.7], "steps": [1]},
            "records": records,
        }
        (panel / f"sampler.shard{shard_index}of8.json").write_text(
            json.dumps(payload) + "\n",
            encoding="utf-8",
        )


def load_outcome(root: Path, name: str) -> dict[str, object]:
    try:
        panel = metric_records.load_sampler_panel(root, name, MetricName.MAX_RUN_FRAC)
    except MetricInputError as error:
        return {"accepted": False, "detail": str(error)}
    return {"accepted": True, "file_count": panel.file_count, "pair_count": len(panel.pairs)}


def interrupted_prepare(root: Path, name: str, stage: str) -> dict[str, object]:
    audit_paths = make_paths(root / name)
    original = metrics_task.write_once
    injected = False

    def fail_once(root_path: Path, relative: Path, data: bytes) -> bool:
        nonlocal injected
        selected = (
            stage == "markdown"
            and root_path == audit_paths.repo_root
            and relative == metrics_task.REPORT_RELATIVE
        ) or (stage == "receipt" and root_path == audit_paths.evidence_root)
        if selected and not injected:
            injected = True
            raise OSError(5, f"injected {stage} interruption")
        return original(root_path, relative, data)

    setattr(metrics_task, "write_once", fail_once)
    try:
        first = metrics_task.prepare_task_two(audit_paths)
    finally:
        setattr(metrics_task, "write_once", original)
    correction = audit_paths.repo_root / metrics_task.CORRECTIONS_RELATIVE
    markdown = audit_paths.repo_root / metrics_task.REPORT_RELATIVE
    after_first = {
        "json": correction.is_file(),
        "markdown": markdown.is_file(),
        "receipt_count": len(tuple(audit_paths.evidence_root.glob("task-02/*/prepare_receipt.json"))),
    }
    retry = metrics_task.prepare_task_two(audit_paths)
    verified = metrics_task.verify_task_two(audit_paths)
    return {
        "injection_fired": injected,
        "first_status": first.status,
        "after_first": after_first,
        "retry_status": retry.status,
        "verify_status": verified.status,
        "receipt_count": len(tuple(audit_paths.evidence_root.glob("task-02/*/prepare_receipt.json"))),
        "recovered": retry.status == "METRICS_COMPLETE" and verified.status == "METRICS_COMPLETE",
    }


def invalid_existing(root: Path) -> dict[str, object]:
    audit_paths = make_paths(root / "invalid-existing")
    correction = audit_paths.repo_root / metrics_task.CORRECTIONS_RELATIVE
    correction.parent.mkdir(parents=True)
    invalid = b"{not-json}\n"
    correction.write_bytes(invalid)
    before = hashlib.sha256(invalid).hexdigest()
    outcome = metrics_task.prepare_task_two(audit_paths)
    after_bytes = correction.read_bytes()
    return {
        "status": outcome.status,
        "before_sha256": before,
        "after_sha256": hashlib.sha256(after_bytes).hexdigest(),
        "bytes_unchanged": after_bytes == invalid,
        "markdown_exists": (audit_paths.repo_root / metrics_task.REPORT_RELATIVE).exists(),
        "receipt_count": len(tuple(audit_paths.evidence_root.glob("task-02/*/prepare_receipt.json"))),
    }


def main() -> None:
    temporary_path = ""
    with tempfile.TemporaryDirectory(prefix="task02-final-recheck-", dir="/tmp/opencode") as temporary:
        temporary_path = temporary
        root = Path(temporary)
        record_root = root / "record-panels"
        write_panel(record_root, "valid-32", 32)
        write_panel(record_root, "mismatch-999", 999)
        write_panel(record_root, "bool-count", True)
        records = {
            "valid_32": load_outcome(record_root, "valid-32"),
            "mismatch_999": load_outcome(record_root, "mismatch-999"),
            "bool_count": load_outcome(record_root, "bool-count"),
        }
        interruptions = {
            "markdown": interrupted_prepare(root, "interrupt-markdown", "markdown"),
            "receipt": interrupted_prepare(root, "interrupt-receipt", "receipt"),
        }
        invalid = invalid_existing(root)
        checks = {
            "valid_32_accepted": records["valid_32"]["accepted"] is True,
            "mismatch_999_rejected": records["mismatch_999"]["accepted"] is False,
            "bool_count_rejected": records["bool_count"]["accepted"] is False,
            "markdown_retry_recovers": interruptions["markdown"]["recovered"] is True,
            "receipt_retry_recovers": interruptions["receipt"]["recovered"] is True,
            "invalid_existing_rejected": invalid["status"] == "TAMPERED",
            "invalid_existing_unchanged": invalid["bytes_unchanged"] is True,
            "invalid_existing_no_descendants": invalid["markdown_exists"] is False and invalid["receipt_count"] == 0,
        }
        output: dict[str, object] = {
            "checks": checks,
            "overall_pass": all(checks.values()),
            "records": records,
            "interruptions": interruptions,
            "invalid_existing": invalid,
        }
    output["cleanup"] = {
        "temporary_path": temporary_path,
        "exists_after_context": Path(temporary_path).exists(),
    }
    output_path = Path(__file__).with_name("focused-outcomes.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
