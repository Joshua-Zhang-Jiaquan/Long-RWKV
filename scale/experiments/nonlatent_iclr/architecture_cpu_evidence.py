from pathlib import Path
from typing import Final, Literal

from pydantic import ValidationError

from .architecture_evidence import EvidenceError, bound_bytes
from .architecture_evidence_models import Record
from .qualification.evidence_contracts import HexDigest

CPU_RECEIPT: Final = Path('.omo/evidence/nonlatent-rnn-dlm-iclr-research/task-03/reconciliation-20260914/cpu-verification.json')
CPU_SHA256: Final = '129c5637ad3c53c921771c87292f6263155713e029a39ee0cefd4822b2a8697a'


class SourceIdentity(Record):
    path: str
    sha256: HexDigest


class CPUVerification(Record):
    scope: Literal['source_hash_bound_cpu_tests_not_gpu_measurements']
    command: str
    exit_code: Literal[0]
    passed: Literal[72]
    trainer_tests: Literal[23]
    initialization_tests: Literal[49]
    warnings: tuple[str, ...]
    active_model_optimizer_membership: Literal[False]
    active_model_mask_loss_semantics: Literal[False]
    base_block_calls: Literal[32]
    recycled_block_calls: Literal[16]
    clock_semantics: tuple[str, ...]
    initialization_claims: tuple[str, ...]
    trainer_claims: tuple[str, ...]
    sources: tuple[SourceIdentity, ...]
    verification_receipt_sha256: HexDigest = CPU_SHA256


def load_cpu_evidence(root: Path) -> CPUVerification | None:
    evidence_tree = root / '.omo/evidence/nonlatent-rnn-dlm-iclr-research'
    if root.resolve() != Path(__file__).parents[3].resolve() and not evidence_tree.exists():
        return None
    try:
        evidence = CPUVerification.model_validate_json(bound_bytes(root / CPU_RECEIPT, CPU_SHA256))
    except ValidationError as error:
        raise EvidenceError('cpu_evidence_schema_mismatch') from error
    for source in evidence.sources:
        _ = bound_bytes(root / source.path, source.sha256)
    return evidence
