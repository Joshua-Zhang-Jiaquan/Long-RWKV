from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .claims import blocked_claims, build_claims
from .contract import SCHEMA_VERSION
from .inventory import (
    MAX_PANEL_FILES,
    METADATA_PATHS,
    PANEL_NAMES,
    SOURCES,
    Artifact,
    SourceChangedError,
    build_artifacts,
    build_views,
    observe_artifact,
)
from .models import (
    AuditResult,
    Claim,
    ClaimState,
    Ledger,
    SourceRoots,
    ledger_bytes,
    parse_ledger,
    result_json,
)
from .publication import UnsafePathError, write_once
from .schema import load_json_object, validate_receipt


DEFAULT_EXTERNAL_ROOT = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")
DEFAULT_STAGED_ROOT = DEFAULT_EXTERNAL_ROOT / "qz_stage_traj4096_v7/scale"
LEDGER_RELATIVE = Path("DAN/nonlatent_iclr/evidence_ledger.json")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


@dataclass(frozen=True, slots=True)
class AuditPaths:
    repo_root: Path
    evidence_root: Path
    external_root: Path
    live_root: Path
    staged_root: Path

    @classmethod
    def from_roots(
        cls,
        repo_root: Path,
        evidence_root: Path,
        *,
        external_root: Path | None = None,
        live_root: Path | None = None,
        staged_root: Path | None = None,
    ) -> AuditPaths:
        repo = _absolute(repo_root)
        return cls(
            repo_root=repo,
            evidence_root=_absolute(evidence_root),
            external_root=_absolute(external_root or DEFAULT_EXTERNAL_ROOT),
            live_root=_absolute(live_root or repo / "scale"),
            staged_root=_absolute(staged_root or DEFAULT_STAGED_ROOT),
        )

    @property
    def snapshot_root(self) -> Path:
        return self.repo_root / "DAN/v7_arch_round"

    @property
    def output_root(self) -> Path:
        return self.repo_root / "DAN/nonlatent_iclr"

    @property
    def ledger_path(self) -> Path:
        return self.repo_root / LEDGER_RELATIVE

    @property
    def source_roots(self) -> SourceRoots:
        return SourceRoots(
            snapshot=str(self.snapshot_root),
            external=str(self.external_root),
            live=str(self.live_root),
            staged=str(self.staged_root),
        )

    def receipt_relative(self, attempt: str) -> Path:
        return Path("task-01") / attempt / "prepare_receipt.json"

    def receipt_path(self, attempt: str) -> Path:
        return self.evidence_root / self.receipt_relative(attempt)


def _build_ledger(paths: AuditPaths) -> Ledger:
    return Ledger(
        schema_version=SCHEMA_VERSION,
        attempt=uuid4().hex,
        source_roots=paths.source_roots,
        artifacts=build_artifacts(paths.snapshot_root, paths.external_root, MAX_PANEL_FILES),
        claims=build_claims(paths.snapshot_root, paths.external_root),
        views=build_views(paths.snapshot_root, paths.live_root, paths.staged_root),
        accounting_note="Packed capacity is never counted as unique tokens.",
    )


def _load(paths: AuditPaths) -> tuple[Ledger, bytes]:
    if paths.ledger_path.is_symlink() or not paths.ledger_path.is_file():
        raise ValueError("ledger is absent, unsafe, or non-regular")
    payload = paths.ledger_path.read_bytes()
    return parse_ledger(payload.decode("utf-8")), payload


def load_task_one(paths: AuditPaths) -> Ledger:
    return _load(paths)[0]


def _receipt_bytes(ledger: Ledger, payload: bytes) -> bytes:
    receipt = {
        "schema_version": 1,
        "attempt": ledger.attempt,
        "ledger_sha256": hashlib.sha256(payload).hexdigest(),
        "status": "AUDIT_COMPLETE",
    }
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def _receipt_problem(paths: AuditPaths, ledger: Ledger, payload: bytes) -> AuditResult | None:
    receipt_path = paths.receipt_path(ledger.attempt)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        return AuditResult("PUBLICATION_PENDING", "ledger exists without a terminal receipt")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        return AuditResult("TAMPERED", "receipt is unsafe or non-regular")
    try:
        raw = load_json_object(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return AuditResult("TAMPERED", "receipt is malformed")
    error = validate_receipt(raw, expected_attempt=ledger.attempt)
    if error is not None:
        return AuditResult("TAMPERED", error)
    if raw["ledger_sha256"] != hashlib.sha256(payload).hexdigest():
        return AuditResult("TAMPERED", "prepared ledger changed")
    return None


def _source_problem(paths: AuditPaths, ledger: Ledger) -> str | None:
    if ledger.source_roots != paths.source_roots:
        return "source roots differ from the prepared ledger"
    current_views = build_views(paths.snapshot_root, paths.live_root, paths.staged_root)
    if current_views != ledger.views:
        return "snapshot/live/staged view state changed"
    for artifact in ledger.artifacts:
        try:
            current = observe_artifact(
                artifact,
                paths.snapshot_root,
                paths.external_root,
                MAX_PANEL_FILES,
            )
        except (OSError, ValueError):
            return f"source could not be inspected: {artifact.origin}:{artifact.path}"
        if current != artifact:
            return f"source changed: {artifact.origin}:{artifact.path}"
    return None


def _complete(ledger: Ledger, detail: str) -> AuditResult:
    return AuditResult("AUDIT_COMPLETE", detail, blocked_claims(ledger.claims))


def _finish_receipt(paths: AuditPaths, ledger: Ledger, payload: bytes) -> AuditResult:
    try:
        written = write_once(
            paths.evidence_root,
            paths.receipt_relative(ledger.attempt),
            _receipt_bytes(ledger, payload),
        )
    except (OSError, UnsafePathError):
        return AuditResult("PUBLICATION_PENDING", "ledger published; receipt retry required")
    if not written:
        problem = _receipt_problem(paths, ledger, payload)
        if problem is not None:
            return problem
    return _complete(ledger, "audit and terminal receipt published; scientific claims remain blocked")


def _recover_existing(paths: AuditPaths) -> AuditResult:
    try:
        ledger, payload = _load(paths)
    except (OSError, UnicodeDecodeError, ValueError):
        return AuditResult("MALFORMED", "existing immutable ledger is invalid")
    receipt_problem = _receipt_problem(paths, ledger, payload)
    if receipt_problem is None:
        return AuditResult("PUBLICATION_EXISTS", "complete immutable publication already exists", blocked_claims(ledger.claims))
    if receipt_problem.status != "PUBLICATION_PENDING":
        return receipt_problem
    source_problem = _source_problem(paths, ledger)
    if source_problem is not None:
        return AuditResult("PUBLICATION_PENDING", f"receipt recovery blocked: {source_problem}")
    return _finish_receipt(paths, ledger, payload)


def prepare_task_one(paths: AuditPaths) -> AuditResult:
    if paths.ledger_path.exists() or paths.ledger_path.is_symlink():
        return _recover_existing(paths)
    try:
        ledger = _build_ledger(paths)
    except SourceChangedError as error:
        return AuditResult("SOURCE_CHANGED", str(error))
    except (OSError, ValueError):
        return AuditResult("SOURCE_UNAVAILABLE", "source inventory could not be completed")
    payload = ledger_bytes(ledger)
    try:
        written = write_once(paths.repo_root, LEDGER_RELATIVE, payload)
    except (OSError, UnsafePathError):
        return AuditResult("PUBLICATION_FAILED", "ledger publication failed; no terminal receipt")
    if not written:
        return _recover_existing(paths)
    return _finish_receipt(paths, ledger, payload)


def verify_task_one(paths: AuditPaths) -> AuditResult:
    try:
        ledger, payload = _load(paths)
    except (OSError, UnicodeDecodeError, ValueError):
        return AuditResult("MALFORMED", "ledger is absent or invalid")
    receipt_problem = _receipt_problem(paths, ledger, payload)
    if receipt_problem is not None:
        return receipt_problem
    source_problem = _source_problem(paths, ledger)
    if source_problem is not None:
        return AuditResult("STALE", source_problem, blocked_claims(ledger.claims))
    return _complete(ledger, "fresh audit; scientific claims remain blocked")


def analyze_task_one(paths: AuditPaths) -> AuditResult:
    verified = verify_task_one(paths)
    if verified.status != "AUDIT_COMPLETE":
        return verified
    ledger = load_task_one(paths)
    unresolved = sum(artifact.identity != "sha256" for artifact in ledger.artifacts)
    detail = (
        f"artifacts={len(ledger.artifacts)}; claims={len(ledger.claims)}; "
        f"blocked_claims={blocked_claims(ledger.claims)}; unresolved_artifacts={unresolved}"
    )
    return AuditResult("AUDIT_COMPLETE", detail, blocked_claims(ledger.claims))


def failure_probe(paths: AuditPaths) -> AuditResult:
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        probe = temporary_root / "repo"
        for _, relative, _ in SOURCES:
            source = paths.snapshot_root / relative
            target = probe / "DAN/v7_arch_round" / relative
            if source.is_file() and not source.is_symlink():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        external = temporary_root / "external"
        external.mkdir()
        isolated = AuditPaths.from_roots(
            probe,
            temporary_root / "evidence",
            external_root=external,
            live_root=temporary_root / "live",
            staged_root=temporary_root / "staged",
        )
        if prepare_task_one(isolated).status != "AUDIT_COMPLETE":
            return AuditResult("PROBE_FAILED", "isolated preparation failed")
        report = probe / "DAN/v7_arch_round/report.md"
        report.write_text("isolated mutation\n", encoding="utf-8")
        changed = verify_task_one(isolated).status == "STALE"
        shutil.copyfile(paths.snapshot_root / "report.md", report)
        (probe / "DAN/v7_arch_round/results/gate_analysis_s4750_20260910.txt").unlink()
        missing = verify_task_one(isolated).status == "STALE"
        status = "EXPECTED_FAILURE_CONFIRMED" if changed and missing else "PROBE_FAILED"
        return AuditResult(status, "isolated checksum change and missing raw record rejected")
