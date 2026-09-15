"""AST-derived repository tasks: real upstream functions with executable checks.

A task is one pure top-level function taken verbatim from a pinned permissive repository.
Purity is enforced by an allowlist over the AST *before* anything is executed: no imports,
no attribute access outside a small method allowlist, and no call to any name we do not
recognise. The shipped check executes the real function in the sandbox and compares its
result to the value produced by executing the same expression independently, and a mutation
panel must fail, so a check that cannot fail is never accepted.

Upstream code is never imported into this process; it only ever runs inside the evaluator.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .provenance import repository_split_map

DERIVATION: Final = "static_ast_pure_expression_v1"
PENDING_SPLIT: Final = "unassigned"
RECORDS_BUNDLE: Final = "records_bundle.json"
MANIFEST_NAME: Final = "manifest.json"

PURE_CALLS: Final = frozenset({
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float",
    "format", "hex", "int", "len", "list", "map", "max", "min", "oct", "ord", "pow", "range",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
})
PURE_METHODS: Final = frozenset({
    "capitalize", "casefold", "copy", "count", "endswith", "find", "format", "get", "index",
    "isalnum", "isalpha", "isdigit", "islower", "isnumeric", "isspace", "isupper", "items",
    "join", "keys", "lower", "lstrip", "replace", "rfind", "rindex", "rsplit", "rstrip", "split",
    "startswith", "strip", "swapcase", "title", "upper", "values",
})
ALLOWED_NODES: Final = frozenset({
    "arguments", "arg", "FunctionDef",
    "Assign", "AugAssign", "AnnAssign", "Return", "If", "For", "While", "Pass", "Break",
    "Continue", "Expr", "Constant", "Name", "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp",
    "Call", "List", "Tuple", "Dict", "Set", "Subscript", "Slice", "ListComp", "SetComp",
    "DictComp", "GeneratorExp", "comprehension", "Store", "Load", "Add", "Sub", "Mult", "Div",
    "FloorDiv", "Mod", "Pow", "LShift", "RShift", "BitOr", "BitAnd", "BitXor", "UAdd", "USub",
    "Invert", "Not", "And", "Or", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot",
    "In", "NotIn", "Index", "Load",
})
ARGUMENT_POOL: Final = (1, 2, 3, 4, 5, 7, 8, 10, 12, 16, 0, -1, -3)
STRING_POOL: Final = ("ab", "abc", "hello", "xyzzy", "aaab", "Racecar", "12", "a b c")
LIST_POOL: Final = ([1, 2, 3], [3, 1, 2], [2, 2, 5], [7, 0, 4], [5, 5, 5])


@dataclass(frozen=True, slots=True)
class CandidateFunction:
    """One pure upstream function eligible to become a repository task."""

    repository: str
    license: str
    path: str
    source_sha256: str
    function_source: str
    entry_point: str
    parameters: tuple[str, ...]


def _is_pure(node: ast.AST) -> bool:
    """Reject any construct that could import, touch the environment, or be non-deterministic."""
    for child in ast.walk(node):
        if type(child).__name__ not in ALLOWED_NODES:
            return False
        if isinstance(child, ast.Call):
            target = child.func
            if isinstance(target, ast.Name) and target.id in PURE_CALLS:
                continue
            if isinstance(target, ast.Attribute) and target.attr in PURE_METHODS and isinstance(target.value, ast.Name):
                continue
            return False
        if isinstance(child, ast.Attribute) and child.attr.startswith("__"):
            return False
    return True


def _parameter_names(node: ast.FunctionDef) -> tuple[str, ...] | None:
    """Names this task must supply: required positional params only, defaults omitted.

    A defaulted parameter is still deterministic, so it is left to its default rather than
    being a reason to reject an otherwise pure function.
    """
    arguments = node.args
    if arguments.posonlyargs or arguments.vararg or arguments.kwonlyargs or arguments.kwarg:
        return None
    required = len(arguments.args) - len(arguments.defaults)
    head = arguments.args[:required]
    if len(head) > 4:
        return None
    return tuple(argument.arg for argument in head)


def extract_candidates(
    repository: str, license_id: str, path: str, text: str, source_sha256: str, *, limit: int = 6,
) -> tuple[CandidateFunction, ...]:
    """Extract eligible pure functions verbatim, in stable source order."""
    try:
        module = ast.parse(text)
    except SyntaxError:
        return ()
    lines = text.splitlines()
    found: list[CandidateFunction] = []
    for node in module.body:
        if len(found) >= limit:
            break
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            continue
        if node.name.startswith("_"):
            continue
        parameters = _parameter_names(node)
        if parameters is None:
            continue
        if not all(isinstance(statement, ast.Return) or _is_pure(statement) for statement in node.body):
            continue
        if not _is_pure(node) or not any(isinstance(inner, ast.Return) for inner in node.body):
            continue
        start, end = node.lineno - 1, (node.end_lineno or node.lineno)
        found.append(CandidateFunction(
            repository=repository, license=license_id, path=path, source_sha256=source_sha256,
            function_source="\n".join(lines[start:end]) + "\n",
            entry_point=node.name, parameters=parameters,
        ))
    return tuple(found)


def _draw(seed: str, pool: tuple[object, ...]) -> object:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return pool[int(digest[:8], 16) % len(pool)]


def argument_literal(candidate: CandidateFunction) -> str:
    """Deterministic small literals, chosen from the function and parameter names."""
    values: list[str] = []
    for parameter in candidate.parameters:
        seed = f"{candidate.repository}:{candidate.path}:{candidate.entry_point}:{parameter}"
        if "str" in parameter or "text" in parameter or "word" in parameter or "s" == parameter:
            values.append(repr(_draw(seed, STRING_POOL)))
        elif "list" in parameter or "arr" in parameter or "items" in parameter or "seq" in parameter:
            values.append(repr(_draw(seed, LIST_POOL)))
        else:
            values.append(repr(_draw(seed, ARGUMENT_POOL)))
    return ", ".join(values)


def task_identifier(candidate: CandidateFunction, index: int) -> str:
    digest = hashlib.sha256(f"{candidate.repository}:{candidate.path}:{candidate.entry_point}".encode()).hexdigest()
    return f"rt-{digest[:12]}-{index:03d}"


def check_source(candidate: CandidateFunction, expected: str) -> str:
    """The shipped executable check: the real function must reproduce the expected value."""
    return f"assert {candidate.entry_point}({argument_literal(candidate)}) == {expected}\n"


def expression_source(candidate: CandidateFunction) -> str:
    """The expression whose value the reference run reports."""
    return f"print(repr({candidate.entry_point}({argument_literal(candidate)})))\n"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class BuiltTask:
    """One qualified repository task, ready to publish."""

    record: dict[str, object]
    source: str
    check: str


def build_one(candidate: CandidateFunction, index: int) -> BuiltTask | None:
    """Turn a candidate into a task, or reject it; the check must discriminate."""
    from .repository_evaluator import evaluate_expression, verify_task

    ok, expected = evaluate_expression(candidate.function_source, expression_source(candidate))
    if not ok:
        return None
    check = check_source(candidate, expected)
    qualified, _reason = verify_task(candidate.function_source, check, entry_point=candidate.entry_point)
    if not qualified:
        return None
    identifier = task_identifier(candidate, index)
    family = _family_of(candidate.repository)
    record: dict[str, object] = {
        "id": identifier,
        "license": candidate.license,
        "provenance": f"https://github.com/{candidate.repository}@{_commit_of(candidate.repository)}:{candidate.path}",
        "repository_family": family,
        "split": PENDING_SPLIT,
        "source_path": f"sources/{identifier}.py",
        "source_sha256": hashlib.sha256(candidate.function_source.encode("utf-8")).hexdigest(),
        "evaluator_path": f"checks/{identifier}.py",
        "evaluator_sha256": hashlib.sha256(check.encode("utf-8")).hexdigest(),
        "entry_point": candidate.entry_point,
        "answer_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
        "derivation": DERIVATION,
    }
    return BuiltTask(record=record, source=candidate.function_source, check=check)


def _family_of(repository: str) -> str:
    from scale.data.split_assign import family_id

    return family_id(repository)


def _commit_of(repository: str) -> str:
    from .repository_sources import PINNED_REPOSITORIES

    for entry in PINNED_REPOSITORIES:
        if entry.repository == repository:
            return entry.commit
    return "unknown"


def interleave(pools: list[list[CandidateFunction]]) -> list[CandidateFunction]:
    """Round-robin across repositories so no single family dominates the corpus."""
    merged: list[CandidateFunction] = []
    index = 0
    while any(index < len(pool) for pool in pools):
        for pool in pools:
            if index < len(pool):
                merged.append(pool[index])
        index += 1
    return merged


def candidate_pool(per_entry: int) -> list[CandidateFunction]:
    """Fetch every pinned repository and extract its eligible functions."""
    from .repository_sources import PINNED_REPOSITORIES, fetch_sources

    pools: list[list[CandidateFunction]] = []
    for entry in PINNED_REPOSITORIES:
        try:
            files = fetch_sources(entry)
        except Exception:
            pools.append([])
            continue
        extracted: list[CandidateFunction] = []
        for path, data in files:
            text = data.decode("utf-8", "replace")
            extracted.extend(extract_candidates(
                entry.repository, entry.license, path, text,
                hashlib.sha256(data).hexdigest(), limit=per_entry,
            ))
        pools.append(extracted)
    return interleave(pools)


def build_corpus(target: int = 200, *, per_entry: int = 24, attempts: int | None = None) -> list[BuiltTask]:
    """Build up to ``target`` qualified tasks from the pinned repositories."""
    pool = candidate_pool(per_entry)
    limit = len(pool) if attempts is None else min(attempts, len(pool))
    built: list[BuiltTask] = []
    for candidate in pool[:limit]:
        if len(built) >= target:
            break
        task = build_one(candidate, len(built))
        if task is not None:
            built.append(task)
    return assign_splits(built)


def assign_splits(tasks: list[BuiltTask]) -> list[BuiltTask]:
    """Assign splits from the built record counts, so a replay derives the identical map."""
    weights: dict[str, int] = {}
    for task in tasks:
        family = str(task.record["repository_family"])
        weights[family] = weights.get(family, 0) + 1
    split_map = repository_split_map(weights)
    for task in tasks:
        task.record["split"] = split_map[str(task.record["repository_family"])]
    return tasks


def publish(tasks: list[BuiltTask], root: Path) -> dict[str, object]:
    """Write sources, checks and the manifest; never overwrites existing bytes."""
    records: list[object] = []
    for task in tasks:
        source_path = root / str(task.record["source_path"])
        check_path = root / str(task.record["evaluator_path"])
        for path, body in ((source_path, task.source), (check_path, task.check)):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                if path.read_text(encoding="utf-8") != body:
                    raise RuntimeError(f"refusing to replace existing task file: {path}")
            else:
                _ = path.write_text(body, encoding="utf-8")
        records.append(task.record)
    bundle = json.dumps({"records": records}, indent=2, sort_keys=True) + "\n"
    bundle_path = root / RECORDS_BUNDLE
    _ = bundle_path.write_text(bundle, encoding="utf-8")
    manifest: dict[str, object] = {
        "version": f"repository-tasks-v1+{len(records)}",
        "license": "per-record upstream licenses: MIT, Apache-2.0 and BSD-3-Clause only",
        "sha256": hashlib.sha256(bundle.encode("utf-8")).hexdigest(),
        "asset_path": RECORDS_BUNDLE,
        "derivation": DERIVATION,
        "records": records,
    }
    write_json(root / MANIFEST_NAME, manifest)
    return manifest
