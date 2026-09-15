"""Current controller/checkpoint/rank binding, independent of historical schemas."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from torch import device as TorchDevice

from .contracts import CHECKPOINT_SHA256, ModelGeometry, QualificationContractError
from .controller_receipt import ControllerBindingEvidence, HexDigest, Nonce, RunId
from .lifecycle_evidence import StrictEvidence
from .manifest import ManifestFile, RuntimeManifest
from .runtime_config import ProbeConfig


class LifecycleBinding(StrictEvidence):
    controller: ControllerBindingEvidence
    run_id: RunId
    nonce: Nonce
    source_manifest_sha256: HexDigest
    checkpoint: ManifestFile
    checkpoint_step: Literal[4750]
    checkpoint_state_tensors: Literal[1955]
    geometry: ModelGeometry
    rank: int = Field(ge=0, le=7)
    local_rank: int = Field(ge=0, le=7)
    world_size: Literal[8]
    device: str = Field(pattern=r"^cuda:[0-7]$")

    @model_validator(mode="after")
    def enforce_identity(self) -> Self:
        if (self.run_id, self.nonce, self.source_manifest_sha256) != (
            self.controller.run_id, self.controller.nonce, self.controller.source_manifest_sha256,
        ):
            raise QualificationContractError("lifecycle_controller_identity_mismatch")
        if self.rank != self.local_rank or self.device != f"cuda:{self.local_rank}":
            raise QualificationContractError("lifecycle_rank_device_mismatch")
        if (self.checkpoint.role != "checkpoint_model" or self.checkpoint.sha256 != CHECKPOINT_SHA256
                or self.checkpoint.size_bytes != 16_367_167_378):
            raise QualificationContractError("lifecycle_checkpoint_identity_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class LifecycleInvocation:
    output_dir: Path
    binding: LifecycleBinding
    device: TorchDevice


def checkpoint_from_manifest(config: ProbeConfig) -> ManifestFile:
    """Read only manifest bytes; checkpoint bytes were verified by preflight."""
    payload = config.manifest_path.read_bytes()
    if sha256(payload).hexdigest() != config.manifest_sha256:
        raise QualificationContractError("lifecycle_manifest_digest_mismatch")
    manifest = RuntimeManifest.model_validate_json(payload)
    entries = tuple(entry for entry in manifest.files if entry.role == "checkpoint_model")
    if len(entries) != 1 or entries[0].path != config.checkpoint_dir / "model.pt":
        raise QualificationContractError("lifecycle_checkpoint_path_mismatch")
    return entries[0]
