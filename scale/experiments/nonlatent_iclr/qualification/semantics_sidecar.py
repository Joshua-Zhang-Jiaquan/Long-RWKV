"""Write-once model semantics evidence using the lifecycle identity contract."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from ..trainer_semantics_contract import SOURCE_SHA256
from .contracts import QualificationRuntimeError
from .lifecycle_binding import LifecycleBinding
from .lifecycle_evidence import StrictEvidence
from .lifecycle_settings import ModelSettings
from .lifecycle_sidecar import LifecycleIssue
from .semantics_evidence import GradientEvidence, MaskEvidence, MembershipEvidence


class SemanticsSidecar(StrictEvidence):
    schema_version: Literal[1] = 1
    scope: Literal["bound_rank_model_semantics"] = "bound_rank_model_semantics"
    runtime_promoted: Literal[False] = False
    trainer_source_sha256: str = SOURCE_SHA256
    binding: LifecycleBinding
    before: ModelSettings | None = None
    after: ModelSettings | None = None
    membership: tuple[MembershipEvidence, ...] = ()
    mask: MaskEvidence | None = None
    gradients: GradientEvidence | None = None
    issue: LifecycleIssue | None = None

    @property
    def passed(self) -> bool:
        if (self.issue is not None or self.before is None or self.after is None
                or self.mask is None or self.gradients is None or len(self.membership) != 2):
            return False
        baseline, frozen = self.membership
        rows = self.gradients.parameters
        return (self.trainer_source_sha256 == SOURCE_SHA256 and self.before == self.after
                and not self.before.training and not self.before.inference_mode and self.before.grad_enabled
                and self.before.parameter_devices == (self.binding.device,)
                and self.mask.device == self.binding.device and self.mask.shape[-1] == self.binding.geometry.vocab_size
                and self.gradients.n_layers == self.binding.geometry.num_hidden_layers
                and self.mask.passed and self.gradients.passed
                and all(item.passed for item in self.membership)
                and not baseline.freeze_backbone and frozen.freeze_backbone
                and baseline.total_tensors == frozen.total_tensors == len(rows)
                and baseline.total_numel == frozen.total_numel == sum(row.numel for row in rows)
                and {name for group in baseline.groups for name in group.names}
                == {row.name for row in rows if row.requires_grad})

    def write_once(self, output_dir: Path) -> Path:
        if output_dir.is_symlink():
            raise QualificationRuntimeError("semantics_output_directory_is_symlink")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"model-semantics-{self.binding.rank}.json"
        temporary = output_dir / f".model-semantics-{self.binding.rank}.{self.binding.nonce}.tmp"
        with temporary.open("x", encoding="utf-8") as handle:
            try:
                handle.write(self.model_dump_json(indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                os.link(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination
