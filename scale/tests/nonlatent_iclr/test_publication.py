from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr import publication, service
from scale.experiments.nonlatent_iclr.service import AuditPaths, prepare_task_one, verify_task_one


def _paths(tmp_path: Path) -> AuditPaths:
    repo = tmp_path / "repo"
    snapshot = repo / "DAN/v7_arch_round"
    for _, relative, _ in service.SOURCES:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture:{relative}\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    return AuditPaths.from_roots(
        repo,
        tmp_path / "evidence",
        external_root=external,
        live_root=tmp_path / "live",
        staged_root=tmp_path / "staged",
    )


def test_write_once_is_immutable_and_creates_contained_parents(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    relative = Path("nested/result.json")
    assert publication.write_once(root, relative, b"first")
    assert not publication.write_once(root, relative, b"second")
    assert (root / relative).read_bytes() == b"first"


@pytest.mark.parametrize("relative", (Path("../escape"), Path("/tmp/escape"), Path(".")))
def test_write_once_rejects_non_contained_destinations(
    tmp_path: Path, relative: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(publication.UnsafePathError):
        publication.write_once(root, relative, b"unsafe")


def test_write_once_rejects_symlink_parent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(publication.UnsafePathError):
        publication.write_once(root, Path("redirect/result.json"), b"unsafe")
    assert not (outside / "result.json").exists()


def test_competing_writers_publish_exactly_one_payload(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payloads = tuple(f"payload-{index}".encode() for index in range(8))
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = tuple(pool.map(lambda body: publication.write_once(root, Path("result"), body), payloads))
    assert sum(outcomes) == 1
    assert (root / "result").read_bytes() in payloads


def test_ledger_write_failure_leaves_no_terminal_receipt_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    real_write_once = publication.write_once

    def fail_ledger(root: Path, relative: Path, data: bytes) -> bool:
        if relative.name == "evidence_ledger.json":
            raise OSError(5, "injected ledger failure")
        return real_write_once(root, relative, data)

    monkeypatch.setattr(service, "write_once", fail_ledger)
    result = prepare_task_one(paths)
    assert result.status == "PUBLICATION_FAILED"
    assert not paths.ledger_path.exists()
    assert not tuple(paths.evidence_root.rglob("prepare_receipt.json"))

    monkeypatch.undo()
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    assert verify_task_one(paths).status == "AUDIT_COMPLETE"


def test_receipt_write_failure_is_nonterminal_then_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    real_write_once = publication.write_once

    def fail_receipt(root: Path, relative: Path, data: bytes) -> bool:
        if relative.name == "prepare_receipt.json":
            raise OSError(5, "injected receipt failure")
        return real_write_once(root, relative, data)

    monkeypatch.setattr(service, "write_once", fail_receipt)
    result = prepare_task_one(paths)
    assert result.status == "PUBLICATION_PENDING"
    assert paths.ledger_path.exists()
    assert verify_task_one(paths).status == "PUBLICATION_PENDING"

    monkeypatch.undo()
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    assert verify_task_one(paths).status == "AUDIT_COMPLETE"
    ledger = json.loads(paths.ledger_path.read_text(encoding="utf-8"))
    receipt = paths.receipt_path(ledger["attempt"])
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_data["status"] == "AUDIT_COMPLETE"
    assert receipt_data["ledger_sha256"] == hashlib.sha256(paths.ledger_path.read_bytes()).hexdigest()


def test_repeated_and_competing_prepares_never_overwrite(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = tuple(pool.map(lambda _: prepare_task_one(paths), range(4)))
    assert all(result.status in {"AUDIT_COMPLETE", "PUBLICATION_EXISTS"} for result in results)
    assert any(result.status == "AUDIT_COMPLETE" for result in results)
    assert verify_task_one(paths).status == "AUDIT_COMPLETE"

    original = paths.ledger_path.read_bytes()
    assert prepare_task_one(paths).status == "PUBLICATION_EXISTS"
    assert paths.ledger_path.read_bytes() == original
