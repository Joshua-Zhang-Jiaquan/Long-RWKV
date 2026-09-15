from hashlib import sha256
from pathlib import Path


class EvidenceError(ValueError):
    code: str

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def bound_bytes(path: Path, digest: str) -> bytes:
    """Check and return the same bytes, avoiding a second unbound read."""
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise EvidenceError(f"evidence_sha256_mismatch:{path}")
    return data


def read_ledger(path: Path, digest: str) -> dict[str, bytes]:
    records: dict[str, bytes] = {}
    for line in bound_bytes(path, digest).decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) not in (2, 3):
            raise EvidenceError("evidence_ledger_schema")
        checksum, name = fields[:2]
        if name in records or Path(name).name != name:
            raise EvidenceError("evidence_ledger_duplicate_or_path")
        target = Path(fields[2]) if len(fields) == 3 else path.parent / name
        records[name] = bound_bytes(target, checksum)
    return records
