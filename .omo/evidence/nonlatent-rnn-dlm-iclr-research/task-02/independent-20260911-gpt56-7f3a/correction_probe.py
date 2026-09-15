#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic==2.11.5"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly using an already cached dependency set:
#      PYTHONPATH=<repo-root> uv run --offline correction_probe.py
# 3. Or use the repository interpreter without writing caches:
#      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<repo-root> python correction_probe.py
# ──────────────────

from __future__ import annotations

import shutil
from importlib import import_module
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

metrics_task = import_module("scale.experiments.nonlatent_iclr.metrics_task")
publication_module = import_module("scale.experiments.nonlatent_iclr.publication")
service_module = import_module("scale.experiments.nonlatent_iclr.service")
CORRECTIONS_RELATIVE = metrics_task.CORRECTIONS_RELATIVE
REPORT_RELATIVE = metrics_task.REPORT_RELATIVE
prepare_task_two = metrics_task.prepare_task_two
verify_task_two = metrics_task.verify_task_two
write_once = publication_module.write_once
AuditPathsFactory = service_module.AuditPaths


HERE = Path(__file__).parent
SOURCE_FIXTURE = HERE / "cli-fixture"
WORK = HERE / "correction-work"


class CorrectionRow(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    loop_panel: str
    control_panel: str
    loop_sha256: str
    control_sha256: str
    pair_count: int
    max_run_delta: float
    max_run_interpretation: str
    tau_status: str


class CorrectionDocument(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)
    schema_version: int
    date: str
    analysis_code_sha256: str
    corrections: tuple[CorrectionRow, ...]
    corrected_claims: tuple[str, ...]
    unmeasured_claims: tuple[str, ...]
    baseline_nll: str
    legacy_ci: str


class VerificationOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str
    detail: str


class InterruptionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    first_prepare_status: str
    retry_prepare_status: str
    verify_status: str
    correction_exists: bool
    markdown_exists: bool
    receipt_count: int


class ProbeOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    tampered_computed_values: VerificationOutcome
    empty_corrections: VerificationOutcome
    invalid_schema_version: VerificationOutcome
    missing_receipt: VerificationOutcome
    missing_markdown: VerificationOutcome
    changed_raw_source: VerificationOutcome
    changed_analysis_code_hash: VerificationOutcome
    report_write_interruption: InterruptionOutcome
    receipt_write_interruption: InterruptionOutcome


class AuditPaths(Protocol):
    repo_root: Path
    evidence_root: Path
    external_root: Path


def _paths(case: Path) -> AuditPaths:
    return AuditPathsFactory.from_roots(
        case / "repo",
        case / "evidence",
        external_root=case / "external",
        live_root=case / "live",
        staged_root=case / "staged",
    )


def _clone(name: str, include_receipt: bool) -> AuditPaths:
    case = WORK / name
    shutil.copytree(SOURCE_FIXTURE / "repo", case / "repo")
    shutil.copytree(SOURCE_FIXTURE / "external", case / "external")
    if include_receipt:
        shutil.copytree(SOURCE_FIXTURE / "evidence", case / "evidence")
    return _paths(case)


def _fresh(name: str) -> AuditPaths:
    case = WORK / name
    shutil.copytree(SOURCE_FIXTURE / "external", case / "external")
    return _paths(case)


def _load(paths: AuditPaths) -> CorrectionDocument:
    return CorrectionDocument.model_validate_json((paths.repo_root / CORRECTIONS_RELATIVE).read_bytes())


def _save(paths: AuditPaths, document: CorrectionDocument) -> None:
    (paths.repo_root / CORRECTIONS_RELATIVE).write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _verified(paths: AuditPaths) -> VerificationOutcome:
    result = verify_task_two(paths)
    return VerificationOutcome(status=result.status, detail=result.detail)


def _interruption(name: str, receipt_failure: bool) -> InterruptionOutcome:
    paths = _fresh(name)
    original = metrics_task.write_once

    def fail_selected_write(root: Path, relative: Path, data: bytes) -> bool:
        selected = root == paths.evidence_root if receipt_failure else root == paths.repo_root and relative == REPORT_RELATIVE
        if selected:
            raise OSError(5, "injected publication interruption")
        return write_once(root, relative, data)

    setattr(metrics_task, "write_once", fail_selected_write)
    try:
        first = prepare_task_two(paths)
    finally:
        setattr(metrics_task, "write_once", original)
    retry = prepare_task_two(paths)
    verified = verify_task_two(paths)
    return InterruptionOutcome(
        first_prepare_status=first.status,
        retry_prepare_status=retry.status,
        verify_status=verified.status,
        correction_exists=(paths.repo_root / CORRECTIONS_RELATIVE).is_file(),
        markdown_exists=(paths.repo_root / REPORT_RELATIVE).is_file(),
        receipt_count=len(tuple(paths.evidence_root.glob("task-02/*/prepare_receipt.json"))),
    )


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir()

    tampered = _clone("tampered-values", True)
    tampered_document = _load(tampered)
    first = tampered_document.corrections[0].model_copy(
        update={"pair_count": 999, "max_run_delta": 12345.0, "max_run_interpretation": "better"}
    )
    _save(tampered, tampered_document.model_copy(update={"corrections": (first, *tampered_document.corrections[1:])}))

    empty = _clone("empty-corrections", True)
    _save(empty, _load(empty).model_copy(update={"corrections": ()}))

    invalid_schema = _clone("invalid-schema", True)
    _save(invalid_schema, _load(invalid_schema).model_copy(update={"schema_version": 999}))

    missing_receipt = _clone("missing-receipt", False)

    missing_markdown = _clone("missing-markdown", True)
    (missing_markdown.repo_root / REPORT_RELATIVE).unlink()

    raw_stale = _clone("raw-stale", True)
    raw_path = raw_stale.external_root / "sampler_gate_m4_loop_s4750/sampler.shard0of1.json"
    raw_path.write_text(raw_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    code_stale = _clone("code-stale", True)
    _save(code_stale, _load(code_stale).model_copy(update={"analysis_code_sha256": "0" * 64}))

    output = ProbeOutput(
        tampered_computed_values=_verified(tampered),
        empty_corrections=_verified(empty),
        invalid_schema_version=_verified(invalid_schema),
        missing_receipt=_verified(missing_receipt),
        missing_markdown=_verified(missing_markdown),
        changed_raw_source=_verified(raw_stale),
        changed_analysis_code_hash=_verified(code_stale),
        report_write_interruption=_interruption("report-interruption", False),
        receipt_write_interruption=_interruption("receipt-interruption", True),
    )
    output_path = HERE / "correction-outcomes.json"
    output_path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(output.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
