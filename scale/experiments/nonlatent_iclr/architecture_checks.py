"""Strict validation and explicitly engineering-only CPU controls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .architecture_contract import contract_json, receipt_json
from .architecture_evidence import EvidenceError


class ContractViolation(Exception):
    """Signals a closed-schema, freshness, or state ownership contract failure."""


def validate_contract_json(serialized: str, repo_root: Path) -> None:
    """Require an exact, fresh canonical payload rather than marker-level agreement."""
    _require_json_object(serialized, "contract_schema")
    try:
        canonical = contract_json(repo_root)
    except (EvidenceError, OSError) as error:
        raise ContractViolation(f"runtime_evidence_unavailable:{error}") from error
    if serialized != canonical:
        raise ContractViolation("canonical_contract_mismatch")


def validate_receipt_json(serialized: str, contract: str, repo_root: Path) -> None:
    """Require the closed receipt schema and exact current contract/source identities."""
    receipt = _require_json_object(serialized, "receipt_schema")
    if set(receipt) != {
        "blocked_claims",
        "contract_sha256",
        "cpu_check_status",
        "receipt_version",
        "source_manifest_sha256",
        "status",
    }:
        raise ContractViolation("receipt_schema")
    validate_contract_json(contract, repo_root)
    if serialized != receipt_json(repo_root, contract):
        raise ContractViolation("receipt_mismatch")


def _require_json_object(serialized: str, label: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ContractViolation(label) from error
    if not isinstance(parsed, dict):
        raise ContractViolation(label)
    return parsed


class _TinyPairedLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn_fwd = nn.Linear(width, width, bias=False)
        self.attn_bwd = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.attn_bwd.weight.copy_(self.attn_fwd.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.attn_fwd(hidden) + self.attn_bwd(hidden)


class TinyArchitectureHarness(nn.Module):
    """Engineering-only torch control for identity and Python-loop counting."""

    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_TinyPairedLayer(width) for _ in range(layers))
        self.block_passes = 0

    def pair_at(self, index: int) -> _TinyPairedLayer:
        """Expose a typed test-only pair without claiming it is a staged RWKV block."""
        layer = self.layers[index]
        if not isinstance(layer, _TinyPairedLayer):
            raise ContractViolation("engineering_harness_corrupt")
        return layer

    def forward(self, hidden: torch.Tensor, outer_nfe: int) -> torch.Tensor:
        self.block_passes = 0
        for _ in range(outer_nfe):
            for layer in self.layers:
                hidden = layer(hidden)
                self.block_passes += 1
            for _ in range(2):
                for layer in self.layers:
                    hidden = layer(hidden)
                    self.block_passes += 1
        return hidden


@dataclass(frozen=True, slots=True)
class PrefixState:
    session_id: str
    canvas_revision: int
    prefix_end: int
    value: tuple[int, ...]


class CanvasStateOwner:  # noqa: MUTABLE_OK
    """Engineering-only caller-state control; not a real FLA cache implementation."""

    def __init__(self) -> None:
        self._canvas_revision = 0
        self._state: PrefixState | None = None

    def capture(self, session_id: str, canvas_revision: int, prefix_end: int, value: tuple[int, ...]) -> PrefixState:
        if canvas_revision != self._canvas_revision:
            raise ContractViolation("stale_canvas")
        self._state = PrefixState(session_id, canvas_revision, prefix_end, value)
        return self._state

    def edit_canvas(self) -> None:
        self._canvas_revision += 1
        self._state = None

    def carry(self, session_id: str, canvas_revision: int, prefix_end: int) -> PrefixState:
        state = self._state
        if canvas_revision != self._canvas_revision:
            raise ContractViolation("stale_canvas")
        if state is None:
            raise ContractViolation("missing_state")
        if state.session_id != session_id:
            raise ContractViolation("session_mismatch")
        if state.prefix_end != prefix_end:
            raise ContractViolation("prefix_mismatch")
        return state
