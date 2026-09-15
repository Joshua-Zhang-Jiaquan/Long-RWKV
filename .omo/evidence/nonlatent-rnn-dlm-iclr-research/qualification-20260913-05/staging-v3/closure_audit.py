from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from importlib.util import resolve_name
from pathlib import Path
from typing import Final


HASH_CHUNK_BYTES: Final = 65_536


@dataclass(frozen=True, slots=True)
class ModuleSource:
    module: str
    path: Path
    imports: tuple[str, ...]


class ClosureAuditError(RuntimeError):
    def __init__(self, module: str) -> None:
        super().__init__(f"local_module_unavailable:{module}")
        self.module = module


def _module_source_path(source_root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    file_path = (source_root / relative).with_suffix(".py")
    if file_path.is_file():
        return file_path
    package_path = source_root / relative / "__init__.py"
    return package_path if package_path.is_file() else None


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, source_root: Path, package: str) -> None:
        self._source_root = source_root
        self._package = package
        self._modules: set[str] = set()

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(sorted(self._modules))

    def _include_local(self, module: str) -> None:
        if module and _module_source_path(self._source_root, module) is not None:
            self._modules.add(module)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._include_local(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            target = resolve_name(
                f"{'.' * node.level}{node.module or ''}", self._package
            )
        else:
            target = node.module or ""
        self._include_local(target)
        for alias in node.names:
            if alias.name != "*":
                self._include_local(f"{target}.{alias.name}")


def resolve_local_closure(
    source_root: Path, entrypoints: tuple[str, ...]
) -> tuple[ModuleSource, ...]:
    pending = list(entrypoints)
    resolved: dict[str, ModuleSource] = {}
    while pending:
        module = pending.pop()
        if module in resolved:
            continue
        path = _module_source_path(source_root, module)
        if path is None:
            raise ClosureAuditError(module)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        collector = _ImportCollector(source_root, package)
        collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        resolved[module] = ModuleSource(
            module=module,
            path=path,
            imports=collector.modules,
        )
        pending.extend(collector.modules)
    return tuple(resolved[module] for module in sorted(resolved))


def all_model_python_sources(source_root: Path) -> tuple[Path, ...]:
    return tuple(sorted((source_root / "models").rglob("*.py")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
