from __future__ import annotations

from pathlib import Path, PurePosixPath

from .contract import CLAIM_CONTRACTS, GATE_RECORD
from .models import Claim, ClaimState


SNAPSHOT_RECORDS = frozenset({GATE_RECORD})
CLAIM_DETAILS: tuple[str, ...] = (
    "LM1B historical loss/PPL assertion; interpretation requires task2",
    "WikiText103 historical loss/PPL assertion; interpretation requires task2",
    "causal path historical comparison; interpretation requires task2",
    "4.98B capacity accounting assertion; interpretation requires task2",
    "max_run_frac direction assertion; interpretation requires task2",
    "masked-token accuracy assertion; interpretation requires task2",
    "tau commit-order assertion; interpretation requires task2",
    "r100 fully-masked assertion; interpretation requires task2",
)


def _present(root: Path, relative: str) -> bool:
    path = PurePosixPath(relative)
    target = root.joinpath(*path.parts)
    return (
        not root.is_symlink()
        and not target.is_symlink()
        and target.is_file()
        and target.resolve().is_relative_to(root.resolve())
    )


def build_claims(snapshot_root: Path, external_root: Path) -> tuple[Claim, ...]:
    claims: list[Claim] = []
    for spec, detail in zip(CLAIM_CONTRACTS, CLAIM_DETAILS, strict=True):
        available = all(
            _present(snapshot_root if record in SNAPSHOT_RECORDS else external_root, record)
            for record in spec.records
        )
        state = ClaimState.CLAIMED if available else ClaimState.UNRESOLVED
        claims.append(Claim(spec.claim_id, spec.source, state, detail, spec.records, spec.fields))
    return tuple(claims)


def blocked_claims(claims: tuple[Claim, ...]) -> int:
    return sum(claim.state in {ClaimState.CLAIMED, ClaimState.UNRESOLVED} for claim in claims)
