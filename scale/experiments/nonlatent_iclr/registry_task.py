"""Standalone CPU-only Task4 prepare and verify entrypoint.

Run from the repository root with ``python -m scale.experiments.nonlatent_iclr.registry_task``.
It does not dispatch training, evaluation, containers, subprocesses, or network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, Mapping, cast

from .tasks.exact_tasks import ExactTaskRequest
from .tasks.exact_tasks import generate_exact_task
from .tasks.exact_tasks import validate_exact_gold
from .tasks.external_assets import inventory_external_assets
from .tasks.length_qualification import REPRESENTATIVE_ARTIFACT
from .tasks.length_qualification import SUMMARY_NAME
from .tasks.provenance import make_provenance
from .tasks.provenance import split_violations
from .task_registry import build_registry
from .task_registry import JsonValue
from .task_registry import ReadinessReport
from .task_registry import registry_document
from .task_registry import verify_registry

IMPLEMENTATION_SOURCES = (
    "tasks/exact_tasks.py", "tasks/provenance.py", "tasks/external_assets.py", "task_registry.py",
    "tasks/pinned_sources.py", "tasks/qualification_adapters.py", "tasks/tokenizer_provenance.py",
    "tasks/length_matrix.py", "tasks/repository_evaluator.py", "registry_task.py",
)

#: The representative length artifact never reads an external suite, so its own summary records
#: that consumption as blocked. That string describes the artifact, not the suites: read inside a
#: registry whose ``blocked`` list is empty it asserts a global blocker that no longer holds. The
#: registry states the scope explicitly instead of repeating the artifact-local wording.
EXTERNAL_SUITES_NOT_CONSUMED: Final = "NOT_CONSUMED_BY_LENGTH_QUALIFICATION"
EXTERNAL_SUITES_STATUS_SCOPE: Final = (
    "representative length artifact only; Task4 external asset readiness is reported by this "
    "registry's asset inventory and blocked[] entries"
)


def main(argv: tuple[str, ...] | None = None) -> int:
    """Prepare declarations or verify CPU checks with a fail-closed exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--case", choices=("happy", "failure"), default="happy")
    parser.add_argument("--asset-root", type=Path, default=Path("DAN/nonlatent_iclr/task4_assets"))
    parser.add_argument("--output-root", type=Path, default=Path("DAN/nonlatent_iclr"))
    parser.add_argument("--evidence-root", type=Path, default=Path(".omo/evidence/nonlatent-rnn-dlm-iclr-research/task-04"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return _prepare(args.output_root, args.evidence_root, args.asset_root)
    if args.case == "failure":
        return _failure_verify(args.output_root)
    return _happy_verify(args.output_root, args.evidence_root, args.asset_root)


def load_accepted_length_qualification(asset_root: Path) -> JsonValue | None:
    """Load the replay-bound token summary, or None when it is absent or no longer bound.

    The deep replay gate lives in the tokenizer adapter (``verify_length_artifact``); this
    reader only binds the summary to the artifact bytes it claims to describe, so a stale or
    tampered summary is never emitted into the registry.
    """
    summary_path = asset_root / SUMMARY_NAME
    artifact_path = asset_root / REPRESENTATIVE_ARTIFACT
    if not summary_path.is_file() or not artifact_path.is_file():
        return None
    try:
        raw: object = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    summary = {str(key): value for key, value in cast("dict[object, object]", raw).items()}
    if summary.get("artifact") != REPRESENTATIVE_ARTIFACT:
        return None
    if summary.get("artifact_sha256") != _file_sha256(artifact_path):
        return None
    hashes = summary.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        return None
    emitted = {key: value for key, value in summary.items() if key != "source_hashes"}
    emitted["external_suites_status"] = EXTERNAL_SUITES_NOT_CONSUMED
    emitted["external_suites_status_scope"] = EXTERNAL_SUITES_STATUS_SCOPE
    matrix_path = asset_root / "length_matrix.json"
    if matrix_path.is_file():
        emitted["matrix_token_lengths_qualified"] = _matrix_replays(asset_root, summary, matrix_path)
        emitted["length_matrix_sha256"] = _file_sha256(matrix_path)
    return cast("JsonValue", emitted)


def _matrix_replays(asset_root: Path, summary: dict[str, object], matrix_path: Path) -> bool:
    """Spot-check the measured matrix from the coordinates the summary declares."""
    model_root = summary.get("replay_model_root")
    evidence_root = summary.get("replay_evidence_root")
    receipt = summary.get("replay_receipt_path")
    if not all(isinstance(value, str) and value for value in (model_root, evidence_root, receipt)):
        return False
    from .tasks.length_matrix import verify_matrix_artifact
    from .tasks.length_qualification import QualificationInputs

    inputs = QualificationInputs(Path(str(model_root)), Path(str(evidence_root)), Path(str(receipt)))
    return verify_matrix_artifact(matrix_path, inputs, sample=3)


def _prepare(output_root: Path, evidence_root: Path, asset_root: Path) -> int:
    registry = build_registry(asset_root)
    report = verify_registry(registry)
    document = registry_document(registry, report, load_accepted_length_qualification(asset_root))
    _archive_then_write(output_root / "task_registry.json", document)
    _archive_then_write_text(output_root / "task_registry.md", _readiness_markdown(report))
    _write_unique(evidence_root, "prepare", _bound_receipt({"registry_sha256": _file_sha256(output_root / "task_registry.json"), "status": document["status"], "blocked": document["blocked"]}))
    return report.exit_code


def _happy_verify(output_root: Path, evidence_root: Path, asset_root: Path) -> int:
    registry = build_registry(asset_root)
    report = verify_registry(registry)
    document = registry_document(registry, report, load_accepted_length_qualification(asset_root))
    _archive_then_write(output_root / "task_registry.json", document)
    _write_unique(
        evidence_root,
        "happy_verify",
        _bound_receipt({"registry_sha256": _file_sha256(output_root / "task_registry.json"), "exact_checks_passed": report.exact_checks_passed, "status": document["status"], "blocked": document["blocked"]}),
    )
    return report.exit_code


def _failure_verify(output_root: Path) -> int:
    failures = {
        "fork_leakage_rejected": _reject_group_leakage(),
        "answer_field_rejected": _reject_hidden_answer_field(),
        "test_tampering_rejected": _reject_test_tampering(),
        "silent_truncation_rejected": _reject_silent_truncation(),
    }
    _write_once(output_root / "task04_failure_verification.json", failures)
    return 0 if all(failures.values()) else 2


def _reject_group_leakage() -> bool:
    original = make_provenance(b"original bytes", "repo-family-7")
    leaked = replace(original, source_bytes=b"paraphrase bytes", split="confirmation")
    return bool(split_violations((original, leaked)))


def _reject_hidden_answer_field() -> bool:
    task = generate_exact_task(ExactTaskRequest("overwrite_delayed_query", 101, 8192, 50, 8, "none", 0))
    altered = replace(task, condition=replace(task.condition, public_prompt="CONTEXT\nBEGIN EVIDENCE\nEND EVIDENCE\n"))
    return not validate_exact_gold(altered)


def _reject_test_tampering() -> bool:
    with TemporaryDirectory(prefix="task4-failure-") as directory:
        root = Path(directory)
        asset = root / "ruler" / "suite.json"
        asset.parent.mkdir(parents=True)
        asset.write_text("altered", encoding="utf-8")
        manifest = asset.parent / "development_manifest.json"
        manifest.write_text(json.dumps({"version": "1", "license": "cleared", "sha256": "0" * 64, "asset_path": "suite.json"}), encoding="utf-8")
        statuses = inventory_external_assets(root)
    return not next(status for status in statuses if status.requirement == "ruler_development_suite").ready


def _reject_silent_truncation() -> bool:
    task = generate_exact_task(ExactTaskRequest("finite_hmm", 101, 8192, 50, 8, "none", 0))
    altered = replace(task, condition=replace(task.condition, public_prompt=task.condition.public_prompt[:-1]))
    return not validate_exact_gold(altered)


def _write_once(path: Path, value: Mapping[str, JsonValue] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise RegistryWriteConflict(path)
        return
    path.write_text(content, encoding="utf-8")


def _archive_then_write(path: Path, value: Mapping[str, JsonValue]) -> None:
    """Archive a previous registry byte-for-byte before replacing its declaration."""
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        digest = hashlib.sha256(existing.encode("utf-8")).hexdigest()[:16]
        archive = path.with_name(f"{path.stem}.archive-{digest}{path.suffix}")
        if not archive.exists():
            archive.write_text(existing, encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _archive_then_write_text(path: Path, content: str) -> None:
    """Archive a previous human-readable summary byte-for-byte before replacing it."""
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        digest = hashlib.sha256(existing.encode("utf-8")).hexdigest()[:16]
        archive = path.with_name(f"{path.stem}.archive-{digest}{path.suffix}")
        if not archive.exists():
            _ = archive.write_text(existing, encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    """Bind a preparation receipt to the exact registry bytes it verified."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_source_hashes(source_root: Path | None = None) -> dict[str, str]:
    """Hash every validator source that qualifies a Task4 registry receipt."""
    root = source_root if source_root is not None else Path(__file__).parent
    return {relative: _file_sha256(root / relative) for relative in IMPLEMENTATION_SOURCES}


def receipt_is_current(receipt: Mapping[str, JsonValue], source_root: Path | None = None) -> bool:
    """Reject legacy or stale receipts that lack current implementation hashes."""
    hashes = receipt.get("implementation_source_hashes")
    return isinstance(hashes, dict) and hashes == implementation_source_hashes(source_root)


def _bound_receipt(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    source_hashes: dict[str, JsonValue] = {}
    for relative, digest in implementation_source_hashes().items():
        source_hashes[relative] = digest
    receipt = dict(payload)
    receipt["implementation_source_hashes"] = source_hashes
    return receipt


def _write_unique(evidence_root: Path, kind: str, value: Mapping[str, JsonValue]) -> None:
    attempt = evidence_root / f"{kind}-{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "result.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _readiness_markdown(report: ReadinessReport) -> str:
    state = "READY" if report.ready else "BLOCKED"
    lines = [
        "# Task 4 registry",
        "",
        f"CPU exact generators: {report.successful_cells} feasible / {report.infeasible_cells} infeasible cells checked.",
        f"Benchmark readiness: {state}.",
        "",
    ]
    for status in report.blocked:
        lines.append(f"- {status.requirement}: {status.reason}")
    return "\n".join(lines) + "\n"


class RegistryWriteConflict(RuntimeError):
    """Raised instead of silently replacing an existing registry artifact."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"registry artifact already exists with different content: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
