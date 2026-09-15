"""Bounded historical CPU verification (not an active trainer)."""
from __future__ import annotations

import ast
import builtins
import re
import symtable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Final

import torch
from torch import nn
from torch.nn import functional

from .trainer_semantics_contract import (
    SEGMENTS,
    SOURCE_PATH,
    SOURCE_SHA256,
    GateConfig as GateConfig,
    HistoricalFunctions,
    OptimizerConfig as OptimizerConfig,
    SourceContractError as SourceContractError,
)

_BUILTINS: Final = ("set", "id", "int", "float", "range", "min", "max", "len", "enumerate")


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Retain the exact verified bytes; subsequent reads cannot alter this snapshot."""

    content: bytes

    def __post_init__(self) -> None:
        if sha256(self.content).hexdigest() != SOURCE_SHA256:
            raise SourceContractError("historical trainer SHA-256 mismatch")

    @classmethod
    def read(cls) -> SourceSnapshot:
        return cls((Path(__file__).resolve().parents[3] / SOURCE_PATH).read_bytes())


def execute_segment(snapshot: SourceSnapshot, segment: str, namespace: ModuleType) -> None:
    """Execute only a pinned AST region after checking its external bindings."""
    if sha256(snapshot.content).hexdigest() != SOURCE_SHA256:
        raise SourceContractError("snapshot hash mismatch")
    if segment not in SEGMENTS:
        raise SourceContractError(f"unapproved segment: {segment}")
    start, end, expected = SEGMENTS[segment]
    selected: list[ast.stmt] = []
    tree = ast.parse(snapshot.content, filename=SOURCE_PATH)
    pending: list[ast.AST] = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.stmt) and start <= node.lineno <= end:
            if node.end_lineno is None or node.end_lineno > end:
                raise SourceContractError(f"segment boundary crossed: {segment}")
            selected.append(node)
        else:
            pending.extend(ast.iter_child_nodes(node))
    selected.sort(key=lambda node: node.lineno)
    module = ast.Module(body=selected, type_ignores=[])
    digest = sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
    if digest != expected:
        raise SourceContractError(f"AST shape mismatch: {segment}")
    # symtable includes globals referenced inside nested comprehensions/functions.
    root = symtable.symtable("from __future__ import annotations\n" + ast.unparse(module), SOURCE_PATH, "exec")
    tables = [root]
    required: set[str] = set()
    provided = {symbol.get_name() for symbol in root.get_symbols() if symbol.is_assigned()}
    while tables:
        table = tables.pop()
        required.update(s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced())
        tables.extend(table.get_children())
    missing = required - provided - set(namespace.__dict__) - set(_BUILTINS)
    if missing:
        raise SourceContractError(f"missing bindings for {segment}: {sorted(missing)}")
    fixed = {"torch": torch, "functional": functional, "re": re, "dist": _LocalOnly}
    for name, binding in fixed.items():
        if name in required and namespace.__dict__.get(name) is not binding:
            raise SourceContractError(f"changed binding: {name}")
    for name, token in (("MASK_TOKEN_ID", 65535), ("PAD_TOKEN_ID", 0)):
        if name in required and namespace.__dict__.get(name) != token:
            raise SourceContractError(f"changed binding: {name}")
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.lineno in (78, 79, 81):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in required:
                if namespace.__dict__.get(target.id) != ast.literal_eval(node.value):
                    raise SourceContractError(f"changed binding: {target.id}")
    if "_LAYER_RE" in required and namespace.__dict__.get("_LAYER_RE") != re.compile(r"layers\.(\d+)\."):
        raise SourceContractError("changed binding: _LAYER_RE")
    namespace.__dict__["__builtins__"] = {name: getattr(builtins, name) for name in _BUILTINS}
    # No imports, trainer main, FSDP wrapping, device conversion or AST rewriting.
    exec(compile(module, SOURCE_PATH, "exec"), namespace.__dict__)


class _LocalOnly:
    @staticmethod
    def is_initialized() -> bool:
        return False


def _namespace(snapshot: SourceSnapshot) -> ModuleType:
    namespace = ModuleType("historical_trainer_cpu")
    # Explicit imported-token bindings, matching historical birwkv7_diffusion.py:49-50.
    namespace.__dict__.update(torch=torch, functional=functional, re=re, dist=_LocalOnly,
                              MASK_TOKEN_ID=65535, PAD_TOKEN_ID=0)
    for segment in ("constants", "regex", "predicate", "corruption", "loss"):
        execute_segment(snapshot, segment, namespace)
    return namespace


def load_functions(snapshot: SourceSnapshot | None = None) -> HistoricalFunctions:
    namespace = _namespace(snapshot if snapshot is not None else SourceSnapshot.read())
    if not isinstance(namespace, HistoricalFunctions):
        raise SourceContractError("extracted function interface mismatch")
    return namespace


def _require_tiny_cpu(model: nn.Module) -> None:
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise SourceContractError("CPU parameters required")
    if sum(p.numel() for p in model.parameters()) > 100_000:
        raise SourceContractError("tiny fixture limit exceeded")


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.AdamW:
    """Mutate freeze flags using original code; construct only a tiny CPU AdamW."""
    _require_tiny_cpu(model)
    snapshot = SourceSnapshot.read()
    namespace = _namespace(snapshot)
    messages: list[str] = []
    namespace.__dict__.update(model=model, args=config, latent_on=config.latent_on, log=messages.append)
    execute_segment(snapshot, "freeze", namespace)
    execute_segment(snapshot, "optimizer", namespace)
    optimizer = namespace.__dict__.get("optimizer")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise SourceContractError("optimizer result mismatch")
    return optimizer


def update_learning_rate(optimizer: torch.optim.AdamW, base_lr: float) -> None:
    snapshot = SourceSnapshot.read()
    namespace = ModuleType("historical_lr_cpu")
    namespace.__dict__.update(optimizer=optimizer, base_lr=base_lr)
    execute_segment(snapshot, "learning_rate", namespace)


def apply_gradient_gate(model: nn.Module, config: GateConfig) -> None:
    """Mutate existing gradients using the original pre-step application block."""
    _require_tiny_cpu(model)
    snapshot = SourceSnapshot.read()
    namespace = _namespace(snapshot)
    namespace.__dict__.update(model=model, args=config, step=config.step, n_layers=config.n_layers)
    execute_segment(snapshot, "gate", namespace)
