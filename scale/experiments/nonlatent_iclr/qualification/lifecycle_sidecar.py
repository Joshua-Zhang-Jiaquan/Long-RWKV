"""Write-once per-rank lifecycle evidence; core rank schemas stay unchanged."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from .contracts import QualificationRuntimeError
from .lifecycle_binding import LifecycleBinding
from .lifecycle_evidence import LifecycleEvidence, StrictEvidence
from .lifecycle_settings import ModelSettings


class LifecycleIssue(StrictEvidence):
    code: Literal["precondition_failed", "execution_error", "invalid_observation"]
    exception_type: str
    detail: str


class LifecycleSidecar(StrictEvidence):
    schema_version: Literal[1] = 1
    scope: Literal["bound_rank_full_canvas_lifecycle"] = "bound_rank_full_canvas_lifecycle"
    runtime_promoted: Literal[False] = False
    binding: LifecycleBinding
    before: ModelSettings | None
    after: ModelSettings | None
    observations: LifecycleEvidence | None
    rejected_observation_json: str | None = None
    issue: LifecycleIssue | None

    @property
    def passed(self) -> bool:
        if (self.issue is not None or self.rejected_observation_json is not None
                or self.observations is None or self.before is None or self.after is None):
            return False
        return (self.observations.passed and self.observations.device == self.binding.device
                and self.observations.vocab_size == self.binding.geometry.vocab_size
                and self.before == self.after and not self.before.training
                and self.before.inference_mode and not self.before.grad_enabled
                and self.before.parameter_devices == (self.binding.device,))

    def write_once(self, output_dir: Path) -> Path:
        if output_dir.is_symlink():
            raise QualificationRuntimeError("lifecycle_output_directory_is_symlink")
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"lifecycle-{self.binding.rank}.json"
        temporary = output_dir / f".lifecycle-{self.binding.rank}.{self.binding.nonce}.tmp"
        # Only a successfully created temporary belongs to this invocation.
        with temporary.open("x", encoding="utf-8") as handle:
            try:
                handle.write(self.model_dump_json(indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                os.link(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination
