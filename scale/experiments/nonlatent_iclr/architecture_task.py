"""Publication lifecycle for the intentionally blocked task-3 CPU subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence, assert_never

from .architecture_checks import ContractViolation, validate_contract_json, validate_receipt_json
from .architecture_contract import CONTRACT_STATUS, blocker_paths, blocked_claim_count, contract_json, receipt_json
from .architecture_evidence import EvidenceError
from .models import AuditResult
from .publication import UnsafePathError, write_once

CONTRACT_RELATIVE = Path("DAN/nonlatent_iclr/architecture_contract.json")
RECEIPT_RELATIVE = Path("DAN/nonlatent_iclr/architecture_contract.receipt.json")
ARCHIVE_ROOT = Path("DAN/nonlatent_iclr/archive/task-03")


def prepare(repo_root: Path) -> AuditResult:
    """Archive a stale owned publication, then publish fresh blocked-only evidence."""
    root = repo_root.resolve()
    try:
        contract = contract_json(root)
        receipt = receipt_json(root, contract)
        if _published_is_fresh(root, contract):
            return _blocked_result("ENGINEERING_ONLY CPU controls; bounded runtime evidence, if present, is fresh; whole architecture remains blocked", root)
        _archive_stale_contract(root)
        _publish(root, CONTRACT_RELATIVE, contract)
        _publish(root, RECEIPT_RELATIVE, receipt)
    except FileNotFoundError as error:
        return _rejected_result("SOURCE_UNAVAILABLE", str(error), root)
    except (ContractViolation, UnsafePathError, EvidenceError) as error:
        return _rejected_result("MALFORMED", str(error), root)
    return _blocked_result("ENGINEERING_ONLY CPU controls; canonical partial claims published; whole architecture remains blocked", root)


def verify(repo_root: Path) -> AuditResult:
    """Freshly validate contract bytes, receipt, hashes, and current source identities."""
    root = repo_root.resolve()
    try:
        contract = (root / CONTRACT_RELATIVE).read_text(encoding="utf-8")
        receipt = (root / RECEIPT_RELATIVE).read_text(encoding="utf-8")
        validate_contract_json(contract, root)
        validate_receipt_json(receipt, contract, root)
    except FileNotFoundError as error:
        return _rejected_result("SOURCE_UNAVAILABLE", str(error), root)
    except ContractViolation as error:
        return _rejected_result("MALFORMED", str(error), root)
    return _blocked_result("ENGINEERING_ONLY CPU controls are fresh; whole architecture remains blocked", root)


def analyze(repo_root: Path) -> AuditResult:
    """Return current dynamic blocker paths without treating installed FLA as executed proof."""
    result = verify(repo_root)
    if result.status != CONTRACT_STATUS:
        return result
    return _blocked_result("ENGINEERING_ONLY; " + _blocker_paths(repo_root.resolve()), repo_root.resolve())


def _published_is_fresh(root: Path, contract: str) -> bool:
    try:
        existing = (root / CONTRACT_RELATIVE).read_text(encoding="utf-8")
        receipt = (root / RECEIPT_RELATIVE).read_text(encoding="utf-8")
        validate_contract_json(existing, root)
        validate_receipt_json(receipt, existing, root)
    except (ContractViolation, FileNotFoundError):
        return False
    return existing == contract


def _archive_stale_contract(root: Path) -> None:
    contract_path = root / CONTRACT_RELATIVE
    if not contract_path.exists():
        return
    data = contract_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    archive = ARCHIVE_ROOT / f"architecture_contract.{digest}.json"
    _publish(root, archive, data.decode("utf-8"))
    contract_path.unlink()
    receipt_path = root / RECEIPT_RELATIVE
    if receipt_path.exists():
        receipt_path.unlink()


def _publish(root: Path, relative: Path, text: str) -> None:
    created = write_once(root, relative, text.encode("utf-8"))
    if not created:
        raise ContractViolation(f"publication_exists:{relative}")


def _blocked_result(detail: str, root: Path) -> AuditResult:
    return AuditResult(CONTRACT_STATUS, detail, _blocked_count(root))


def _rejected_result(status: str, detail: str, root: Path) -> AuditResult:
    try:
        count = _blocked_count(root)
    except (EvidenceError, OSError):
        count = 1
    return AuditResult(status, detail, count)


def _blocked_count(root: Path) -> int:
    count = blocked_claim_count(root)
    if count <= 0:
        raise ContractViolation("blocked_count")
    return count


def _blocker_paths(root: Path) -> str:
    return "; ".join(blocker_paths(root))


def main(argv: Sequence[str] | None = None) -> int:
    """Run an owned standalone CPU happy or rejection path without shared CLI changes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("prepare", "happy", "failure"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    match arguments.case:
        case "prepare":
            result = prepare(arguments.repo_root)
            print(json.dumps({"status": result.status, "detail": result.detail, "blocked_claims": result.blocked_claims}))
            return 0 if result.status == CONTRACT_STATUS else 1
        case "happy":
            result = verify(arguments.repo_root)
            print(json.dumps({"status": result.status, "detail": result.detail, "blocked_claims": result.blocked_claims}))
            return 0 if result.status == CONTRACT_STATUS else 1
        case "failure":
            try:
                validate_contract_json("{}", arguments.repo_root)
            except ContractViolation as error:
                print(json.dumps({"status": "EXPECTED_FAILURE_CONFIRMED", "detail": str(error)}))
                return 0
            return 1
        case unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    raise SystemExit(main())
