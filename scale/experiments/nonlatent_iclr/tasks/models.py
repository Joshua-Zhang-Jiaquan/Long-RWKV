"""Typed values for public conditions and private exact-task gold."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

FAMILIES = frozenset({"associative_recall", "overwrite_delayed_query", "finite_hmm", "code_dataflow"})
LENGTHS = frozenset({4096, 8192, 16384, 32768, 65536})
POSITIONS = frozenset({10, 50, 90})
LOADS = frozenset({1, 8, 32, 128})
DISTRACTORS = frozenset({"none", "random", "similar"})
DATA_SEEDS = frozenset({101, 102, 103, 104, 105})


@dataclass(frozen=True, slots=True)
class ExactTaskRequest:
    """One lazy exact-task instance declaration."""

    family: str
    data_seed: int
    declared_length: int
    position_fraction: int
    load: int
    distractor: str
    instance_index: int

    def __post_init__(self) -> None:
        """Reject undeclared matrix coordinates before task construction."""
        if self.family not in FAMILIES:
            raise ValueError(f"undeclared family: {self.family}")
        if self.data_seed not in DATA_SEEDS:
            raise ValueError(f"undeclared data seed: {self.data_seed}")
        if self.declared_length not in LENGTHS:
            raise ValueError(f"undeclared character length: {self.declared_length}")
        if self.position_fraction not in POSITIONS:
            raise ValueError(f"undeclared position fraction: {self.position_fraction}")
        if self.load not in LOADS:
            raise ValueError(f"undeclared load: {self.load}")
        if self.distractor not in DISTRACTORS:
            raise ValueError(f"undeclared distractor: {self.distractor}")
        if not 0 <= self.instance_index < 200:
            raise ValueError(f"undeclared instance index: {self.instance_index}")


@dataclass(frozen=True, slots=True)
class PublicCondition:
    """Serializable prompt projection without evaluator-only gold."""

    public_prompt: str
    source_bytes: bytes
    source_family: str
    declared_length: int
    length_kind: str

    def public_dict(self) -> dict[str, str | int]:
        """Return the only condition fields eligible for registry serialization."""
        return {
            "public_prompt": self.public_prompt,
            "source_sha256": sha256(self.source_bytes).hexdigest(),
            "source_family": self.source_family,
            "declared_length": self.declared_length,
            "length_kind": self.length_kind,
        }


@dataclass(frozen=True, slots=True)
class ExactTask:
    """A public condition paired with evaluator-private exact gold."""

    condition: PublicCondition
    correct_answer: str
