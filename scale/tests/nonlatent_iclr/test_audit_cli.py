from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from scale.experiments.nonlatent_iclr import service
from scale.experiments.nonlatent_iclr.service import AuditPaths, prepare_task_one, verify_task_one


SOURCES = (
    "README.md", "ARCHITECTURE.md", "TRAINING_HISTORY.md", "report.md",
    "results/gate_analysis_s4750_20260910.txt", "results/ability_curve_m4loop.csv",
    "results/ability_curve_m4mhc.csv", "scripts/wl_gate_analysis.py",
    "code/eval/sampler_eval_offline.py", "code/models/birwkv7_diffusion.py",
    "code/models/residual_streams.py", "code/train/test_backbone_loop.py",
    "code/eval/lm_eval.py", "code/train/train_birwkv_diffusion.py",
    "specs/m4_loop_32gpu_half.json", "JOBS.md",
)


@pytest.fixture
def paths(tmp_path: Path) -> AuditPaths:
    # Given: a bounded v7-shaped isolated repository with no external raw records.
    root = tmp_path / "repo"
    source_root = root / "DAN" / "v7_arch_round"
    for relative in SOURCES:
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture:{relative}\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    return AuditPaths.from_roots(
        root,
        tmp_path / "evidence",
        external_root=external,
        live_root=tmp_path / "live",
        staged_root=tmp_path / "staged",
    )


def test_prepare_then_verify_when_sources_are_fresh(paths: AuditPaths) -> None:
    # Given: a fresh fixture.
    # When: preparation and happy verification run.
    # Then: audit passes while unresolved external evidence remains visible.
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    assert verify_task_one(paths).status == "AUDIT_COMPLETE"


def test_verify_refuses_when_prepared_source_changes(paths: AuditPaths) -> None:
    # Given: a prepared fixture.
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    (paths.repo_root / "DAN/v7_arch_round/report.md").write_text("changed\n", encoding="utf-8")
    # When: verification sees a changed source.
    # Then: it fails closed.
    assert verify_task_one(paths).status == "STALE"


def test_verify_refuses_when_prepared_source_disappears(paths: AuditPaths) -> None:
    # Given: a prepared fixture.
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    (paths.repo_root / "DAN/v7_arch_round/report.md").unlink()
    # When: verification sees a missing source.
    # Then: it fails closed.
    assert verify_task_one(paths).status == "STALE"


def test_verify_refuses_malformed_ledger(paths: AuditPaths) -> None:
    # Given: a malformed prepared-ledger boundary.
    paths.ledger_path.parent.mkdir(parents=True)
    paths.ledger_path.write_text("{", encoding="utf-8")
    # When: verification parses it.
    # Then: malformed schema cannot pass.
    assert verify_task_one(paths).status == "MALFORMED"


def test_prepare_is_non_overwriting_and_symlink_safe(paths: AuditPaths) -> None:
    # Given: a published immutable ledger.
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    original = paths.ledger_path.read_bytes()
    # When: preparation is interrupted/repeated at the publication boundary.
    # Then: existing publication is retained, never overwritten.
    assert prepare_task_one(paths).status == "PUBLICATION_EXISTS"
    assert paths.ledger_path.read_bytes() == original


def _artifact_values(ledger: dict[str, object]) -> list[object]:
    return cast(list[object], ledger["artifacts"])


def _first_artifact(ledger: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], _artifact_values(ledger)[0])


def _invalid_version(ledger: dict[str, object]) -> None:
    ledger["schema_version"] = True


def _invalid_identity(ledger: dict[str, object]) -> None:
    _first_artifact(ledger)["identity"] = "unknown"


def _missing_digest(ledger: dict[str, object]) -> None:
    _first_artifact(ledger)["sha256"] = None


def _scalar_artifact(ledger: dict[str, object]) -> None:
    _artifact_values(ledger)[0] = "scalar"


def _empty_inventory(ledger: dict[str, object]) -> None:
    ledger["artifacts"] = []


MUTATIONS: tuple[Callable[[dict[str, object]], None], ...] = (
    _invalid_version,
    _invalid_identity,
    _missing_digest,
    _scalar_artifact,
    _empty_inventory,
)


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_verify_rejects_digest_consistent_invalid_schema(
    paths: AuditPaths, mutation: Callable[[dict[str, object]], None]
) -> None:
    # Given: a digest-consistent isolated malformed ledger.
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    ledger = cast(dict[str, object], json.loads(paths.ledger_path.read_text(encoding="utf-8")))
    mutation(ledger)
    payload = (json.dumps(ledger, sort_keys=True) + "\n").encode()
    paths.ledger_path.write_bytes(payload)
    attempt = ledger["attempt"]
    assert isinstance(attempt, str)
    receipt = paths.evidence_root / "task-01" / attempt / "prepare_receipt.json"
    receipt.write_text(json.dumps({"schema_version": 1, "attempt": attempt, "ledger_sha256": hashlib.sha256(payload).hexdigest(), "status": "AUDIT_COMPLETE"}), encoding="utf-8")
    # When: verification parses the rewritten boundary.
    # Then: it returns a structured invalid status without a traceback.
    assert verify_task_one(paths).status in {"MALFORMED", "INVALID_COVERAGE"}


@pytest.mark.parametrize("attempt", (1, 2))
def test_prepare_recovers_when_ledger_publication_fails(paths: AuditPaths, monkeypatch: pytest.MonkeyPatch, attempt: int) -> None:
    # Given: an injected first ledger-write failure in an isolated fixture.
    def fail_ledger(root: Path, relative: Path, data: bytes) -> bool:
        assert root == paths.repo_root
        assert relative.name == "evidence_ledger.json"
        assert data
        raise OSError(5, f"injected-{attempt}")
    monkeypatch.setattr(service, "write_once", fail_ledger)
    assert prepare_task_one(paths).status == "PUBLICATION_FAILED"
    assert not paths.ledger_path.exists()
    # When: the same preparation is retried after the fault is removed.
    monkeypatch.undo()
    # Then: it publishes one fresh, verifiable ledger without overwriting a contender.
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    assert verify_task_one(paths).status == "AUDIT_COMPLETE"


def test_cli_rejects_unimplemented_future_task_with_structured_status(paths: AuditPaths) -> None:
    # Given: the real module CLI and an isolated root.
    repo = Path(__file__).parents[3]
    # When: an unimplemented task is requested.
    result = subprocess.run(
        [sys.executable, "-m", "scale.experiments.nonlatent_iclr.cli", "prepare", "--task", "3", "--repo-root", str(paths.repo_root), "--evidence-root", str(paths.evidence_root)],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    # Then: it never reports a stub pass.
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "UNSUPPORTED"
