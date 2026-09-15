"""Hash-bound original AST execution; no model imports or process-global shims."""
from __future__ import annotations

import ast
import builtins
import symtable
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Final

import torch
from torch import nn

from .initialization_contract import REGIONS, SOURCES, FixtureBase, SourceContractError

# __import__ is inert for the pinned bodies (hash-pinned ASTs contain no import
# statements; symtable would flag any __import__ reference) but torch's lazy
# TypedStorage C++ loader reads __import__ from the CALLING frame's builtins on
# first tensor creation; without it PyTorch asserts (DynamicTypes.cpp).
_BUILTINS: Final = ("__build_class__", "super", "property", "int", "float", "range",
                    "min", "isinstance", "ValueError", "RuntimeError", "__import__")


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    name: str
    content: bytes

    def __post_init__(self) -> None:
        if self.name not in SOURCES or sha256(self.content).hexdigest() != SOURCES[self.name]:
            raise SourceContractError("historical source SHA-256 mismatch")

    @classmethod
    def read(cls, name: str) -> SourceSnapshot:
        if name not in SOURCES:
            raise SourceContractError("unapproved source")
        root = Path(__file__).resolve().parents[3]
        return cls(name, (root / "DAN/v7_arch_round/code/models" / name).read_bytes())


def select_region(snapshot: SourceSnapshot, region: str) -> ast.Module:
    """Select complete original statements, retaining their original line metadata."""
    snapshot.__post_init__()
    if region not in REGIONS:
        raise SourceContractError("unapproved region")
    name, start, end, _ = REGIONS[region]
    if name != snapshot.name:
        raise SourceContractError("wrong source for region")
    selected: list[ast.stmt] = []
    pending: list[ast.AST] = [ast.parse(snapshot.content, filename=name)]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.stmt) and start <= node.lineno <= end:
            if node.end_lineno is None or node.end_lineno > end:
                raise SourceContractError("region boundary crossed")
            selected.append(node)
        else:
            pending.extend(ast.iter_child_nodes(node))
    return ast.Module(body=sorted(selected, key=lambda node: node.lineno), type_ignores=[])


_NAMED_SCOPE_NODES: Final = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_SCOPE_NODES: Final = (*_NAMED_SCOPE_NODES, ast.Lambda)


def _ordered_events(node: ast.AST, events: list[tuple[bool, str]]) -> None:
    """Name events (is_store, name) in Python evaluation order."""
    if isinstance(node, ast.Name):
        events.append((not isinstance(node.ctx, ast.Load), node.id))
        return
    if isinstance(node, _SCOPE_NODES):
        if isinstance(node, _NAMED_SCOPE_NODES):
            for decorator in node.decorator_list:
                _ordered_events(decorator, events)
            events.append((True, node.name))
        return
    children: Iterable[ast.AST]
    if isinstance(node, ast.Assign):
        children = (node.value, *node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
        children = (node.value, node.target)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        children = (node.iter, node.target, *node.body, *node.orelse)
    else:
        children = ast.iter_child_nodes(node)
    for child in children:
        _ordered_events(child, events)


def _live_in_inputs(module: ast.Module) -> set[str]:
    """Names read at region scope before their first region-scope assignment.

    symtable marks a later-assigned name as provided even when an earlier
    statement reads it first (e.g. ``h`` in the loop region); those reads are
    live-in inputs, not self-provided. Loads inside a statement count before
    its stores (fail-closed for read-modify-write assignments).
    """
    events: list[tuple[bool, str]] = []
    for stmt in module.body:
        _ordered_events(stmt, events)
    live_in: set[str] = set()
    written: set[str] = set()
    for is_store, name in events:
        if is_store:
            written.add(name)
        elif name not in written:
            live_in.add(name)
    return live_in


def execute_region(snapshot: SourceSnapshot, region: str, namespace: ModuleType) -> None:
    """Fail closed before compilation on changed bytes, AST or missing dependencies."""
    module = select_region(snapshot, region)
    if sha256(ast.dump(module, include_attributes=False).encode()).hexdigest() != REGIONS[region][3]:
        raise SourceContractError("AST shape mismatch")
    root = symtable.symtable("from __future__ import annotations\n" + ast.unparse(module), snapshot.name, "exec")
    tables = [root]
    required: set[str] = set()
    provided = {s.get_name() for s in root.get_symbols() if s.is_assigned()}
    while tables:
        table = tables.pop()
        required.update(s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced())
        tables.extend(table.get_children())
    if region == "methods":
        required.add("FixtureBase")
    missing = ((required - provided) | _live_in_inputs(module)) - set(namespace.__dict__) - set(_BUILTINS)
    if missing:
        raise SourceContractError(f"missing dependencies: {sorted(missing)}")
    for name, value in (("torch", torch), ("nn", nn), ("TypedTorchModule", nn.Module), ("FixtureBase", FixtureBase)):
        if name in required and namespace.__dict__.get(name) is not value:
            raise SourceContractError(f"changed binding: {name}")
    namespace.__dict__["__builtins__"] = {name: getattr(builtins, name) for name in _BUILTINS}
    # Methods remain unchanged; only their class container is a CPU fixture shell.
    if region == "methods":
        shell = ast.ClassDef(name="HistoricalMethods", bases=[ast.Name(id="FixtureBase", ctx=ast.Load())],
                             keywords=[], body=module.body, decorator_list=[], type_params=[])
        module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    exec(compile(module, snapshot.name, "exec"), namespace.__dict__)


def control_namespace() -> ModuleType:
    namespace = ModuleType("historical_initialization_cpu")
    # Explicit CPU fixture base; historical TypedTorchModule is absent locally.
    namespace.__dict__.update(torch=torch, nn=nn, TypedTorchModule=nn.Module)
    execute_region(SourceSnapshot.read("residual_streams.py"), "controls", namespace)
    return namespace
