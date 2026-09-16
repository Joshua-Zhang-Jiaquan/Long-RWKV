from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, cast

from .allocation.task import main as run_allocation_task
from .models import AuditResult, Ledger, result_json
from .metrics_task import analyze_task_two, failure_probe_task_two, prepare_task_two, verify_task_two
from .publication import UnsafePathError, write_once
from .registry_task import main as run_registry_task
from .task_registry import build_registry, verify_registry
from .service import (
    DEFAULT_EXTERNAL_ROOT,
    DEFAULT_STAGED_ROOT,
    AuditPaths,
    analyze_task_one,
    failure_probe,
    load_task_one,
    prepare_task_one,
    verify_task_one,
)


REPO_DEFAULT = Path(__file__).parents[3]
EVIDENCE_DEFAULT = REPO_DEFAULT / ".omo/evidence/nonlatent-rnn-dlm-iclr-research"
#: The calibration aggregate a Task-6 ledger is built from. Its sibling record names the
#: manifest digest the calibration job was bound to.
CALIBRATION_AGGREGATE_DEFAULT = Path(os.environ.get(
    "NONLATENT_CALIBRATION_AGGREGATE",
    REPO_DEFAULT / ".omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260915-12-live/run-artifacts/aggregate-calibration.json",
))


class CliArgumentError(ValueError):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliArgumentError(message)


def _emit(result: AuditResult, code: int) -> int:
    print(result_json(result))
    return code


def _paths(args: argparse.Namespace) -> AuditPaths:
    return AuditPaths.from_roots(
        cast(Path, args.repo_root),
        cast(Path, args.evidence_root),
        external_root=cast(Path | None, args.external_root),
        live_root=cast(Path | None, args.live_root),
        staged_root=cast(Path | None, args.staged_root),
    )


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(ledger: Ledger) -> str:
    roots = ledger.source_roots
    lines = [
        "# Nonlatent historical evidence audit",
        "",
        "## Source roots",
        "",
        f"- snapshot: `{roots.snapshot}`",
        f"- external: `{roots.external}`",
        f"- live: `{roots.live}`",
        f"- staged: `{roots.staged}`",
        "",
        "## Artifacts",
        "",
        "| Source | Origin | Path | Kind | Identity | SHA-256 | Bytes | Files | Reason |",
        "|---|---|---|---|---|---|---:|---:|---|",
    ]
    for item in ledger.artifacts:
        lines.append(
            "| " + " | ".join(_cell(value) for value in (
                item.source,
                item.origin,
                item.path,
                item.kind,
                item.identity,
                item.sha256 or "",
                item.bytes if item.bytes is not None else "",
                item.file_count,
                item.reason or "",
            )) + " |"
        )
    lines.extend((
        "",
        "## Historical claims",
        "",
        "| Claim ID | Source | State | Detail | Exact records | Fields |",
        "|---|---|---|---|---|---|",
    ))
    for claim in ledger.claims:
        lines.append(
            "| " + " | ".join(_cell(value) for value in (
                claim.claim_id,
                claim.source,
                claim.state,
                claim.detail,
                ", ".join(claim.records),
                ", ".join(claim.fields),
            )) + " |"
        )
    lines.extend((
        "",
        "## Snapshot/live/staged views",
        "",
        "| View | State | Detail |",
        "|---|---|---|",
    ))
    for view in ledger.views:
        lines.append(f"| {_cell(view.source)} | {_cell(view.state)} | {_cell(view.detail)} |")
    lines.extend((
        "",
        "## Accounting",
        "",
        ledger.accounting_note,
        "",
        "Historical interpretation remains blocked pending task2; this audit does not recompute science.",
        "",
    ))
    return "\n".join(lines)


def _render(paths: AuditPaths, output: Path) -> AuditResult:
    verified = verify_task_one(paths)
    if verified.status != "AUDIT_COMPLETE":
        return verified
    candidate = Path(os.path.abspath(os.fspath(output)))
    if candidate == paths.output_root or not candidate.is_relative_to(paths.output_root):
        return AuditResult("UNSAFE_DESTINATION", "render output must be inside DAN/nonlatent_iclr")
    relative = candidate.relative_to(paths.repo_root)
    try:
        written = write_once(paths.repo_root, relative, _markdown(load_task_one(paths)).encode())
    except UnsafePathError:
        return AuditResult("UNSAFE_DESTINATION", "render path contains an unsafe component")
    except OSError:
        return AuditResult("PUBLICATION_FAILED", "render publication failed")
    if not written:
        return AuditResult("PUBLICATION_EXISTS", "render destination is immutable")
    return AuditResult("AUDIT_COMPLETE", "rendered immutable evidence audit", verified.blocked_claims)


def _result_task_four(args: argparse.Namespace) -> tuple[AuditResult, int]:
    """Route the plan's Task4 QA contract to the standalone registry entrypoint."""
    command = cast(str, args.command)
    if command not in {"prepare", "verify"}:
        return AuditResult("UNSUPPORTED", "task 4 supports prepare and verify only"), 2
    asset_root = cast("Path | None", args.asset_root) or Path("DAN/nonlatent_iclr/task4_assets")
    output_root = cast("Path | None", args.output_root) or Path("DAN/nonlatent_iclr")
    evidence_root = cast("Path | None", args.evidence_root) or EVIDENCE_DEFAULT
    plant_failure = command == "verify" and cast(str, args.case) == "failure"
    exit_code = run_registry_task((
        command, "--case", "failure" if plant_failure else "happy",
        "--asset-root", str(asset_root), "--output-root", str(output_root),
        "--evidence-root", str(evidence_root / "task-04"),
    ))
    if plant_failure:
        if exit_code == 0:
            return AuditResult("EXPECTED_FAILURE_CONFIRMED", "all planted failures were rejected"), 0
        return AuditResult("FAILURE_PROBE_MISSED", "a planted failure was not rejected"), 2
    report = verify_registry(build_registry(asset_root))
    blocked = len(report.blocked)
    if exit_code == 0 and report.ready:
        return AuditResult("TASK4_READY", "all declared external assets qualified by replay"), 0
    first = report.blocked[0].reason if report.blocked else "unqualified"
    return AuditResult("BLOCKED_EXTERNAL", f"{blocked} declared requirement(s) unqualified: {first}", blocked), 2


def _result_task_five(args: argparse.Namespace) -> tuple[AuditResult, int]:
    """Route the plan's Task 5 QA contract to the preregistration record."""
    from .preregistration import (
        PreregistrationRefusal, build_preregistration, read_preregistration,
        verify_preregistration, write_preregistration,
    )

    command = cast(str, args.command)
    if command not in {"prepare", "verify"}:
        return AuditResult("UNSUPPORTED", "task 5 supports prepare and verify only"), 2
    output_root = cast("Path | None", args.output_root) or Path("DAN/nonlatent_iclr")
    path = output_root / "preregistration.json"

    if command == "prepare":
        document = build_preregistration()
        write_preregistration(document, path)
        state = cast(str, document["seal_state"])
        open_now = cast("list[str]", document["open_blockers"])
        if state == "SEALED":
            return AuditResult("PREREGISTRATION_SEALED", "all sealing blockers closed"), 0
        return AuditResult(
            "DRAFT_UNSEALED",
            f"written as a draft; {len(open_now)} sealing blocker(s) open: {open_now}"), 1

    try:
        document = read_preregistration(path)
    except PreregistrationRefusal as error:
        return AuditResult("UNREADABLE", str(error)[:300]), 2

    case = cast(str, args.case) if args.case else "happy"
    try:
        outcome = verify_preregistration(document, case=case)
    except PreregistrationRefusal as error:
        if case == "failure":
            return AuditResult("FAILURE_PROBE_MISSED", str(error)[:300]), 2
        return AuditResult("PREREGISTRATION_INVALID", str(error)[:300]), 2
    if case == "failure":
        return AuditResult(outcome, "all planted preregistration violations were refused"), 0
    state = cast(str, document["seal_state"])
    if state != "SEALED":
        return AuditResult(
            "DRAFT_UNSEALED",
            "the record is internally consistent and unedited, but it is a DRAFT: "
            f"open blockers {document['open_blockers']}"), 1
    return AuditResult(outcome, "sealed preregistration verified"), 0


def _result_task_six(args: argparse.Namespace) -> tuple[AuditResult, int]:
    """Route the plan's Task-6 contract to the allocation ledger entrypoint."""
    command = cast(str, args.command)
    if command not in {"prepare", "analyze", "verify"}:
        return AuditResult("UNSUPPORTED", "task 6 supports prepare, analyze, and verify only"), 2
    aggregate = cast("Path | None", args.aggregate) or CALIBRATION_AGGREGATE_DEFAULT
    output_root = cast("Path | None", args.output_root) or Path("DAN/nonlatent_iclr")
    manifest_sha = cast("str | None", args.manifest_sha256) or _calibration_manifest_sha(aggregate)
    try:
        exit_code = run_allocation_task((
            command, "--case", "failure" if command == "verify" and cast(str, args.case) == "failure" else "happy",
            "--aggregate", str(aggregate), "--manifest-sha256", manifest_sha, "--output-root", str(output_root),
        ))
    except Exception as error:  # noqa: BROAD_EXCEPT_OK - a refusal must carry a reason, not a traceback
        return AuditResult(
            "BLOCKED_EXTERNAL",
            f"task 6 could not build a ledger from {aggregate}: {type(error).__name__}: {error}"[:300],
        ), 2
    if command == "verify" and cast(str, args.case) == "failure":
        if exit_code == 0:
            return AuditResult("EXPECTED_FAILURE_CONFIRMED", "all planted allocation violations were refused"), 0
        return AuditResult("FAILURE_PROBE_MISSED", "a planted allocation violation was admitted"), 2
    if exit_code == 0:
        return AuditResult("LEDGER_READY" if command != "verify" else "ALLOCATION_ADMITTED", "forecast reproduced from the bound calibration aggregate"), 0
    return AuditResult("BLOCKED_EXTERNAL", "no arm of the forecast can be priced from the current calibration"), 1


def _calibration_manifest_sha(aggregate: Path) -> str:
    """The calibration manifest a ledger must bind. Read from the record next to the aggregate."""
    companion = aggregate.with_name("calibration-manifest.sha256")
    if companion.is_file():
        return companion.read_text(encoding="utf-8").strip()
    return "0" * 64


def _result(args: argparse.Namespace) -> tuple[AuditResult, int]:
    task = cast(int, args.task)
    if task not in {1, 2, 4, 5, 6}:
        return AuditResult("UNSUPPORTED", f"task {task} is not implemented"), 2
    if task == 4:
        return _result_task_four(args)
    if task == 5:
        return _result_task_five(args)
    if task == 6:
        return _result_task_six(args)
    paths = _paths(args)
    command = cast(str, args.command)
    if task == 2:
        if command == "prepare":
            result = prepare_task_two(paths)
        elif command == "analyze":
            result = analyze_task_two(paths)
        elif command == "verify":
            result = failure_probe_task_two() if args.case == "failure" else verify_task_two(paths)
        else:
            return AuditResult("UNSUPPORTED", "task 2 supports prepare, analyze, and verify only"), 2
        return result, 0 if result.status in {"METRICS_COMPLETE", "EXPECTED_FAILURE_CONFIRMED"} else 2
    if command == "prepare":
        result = prepare_task_one(paths)
    elif command == "analyze":
        result = analyze_task_one(paths)
    elif command == "verify":
        result = failure_probe(paths) if args.case == "failure" else verify_task_one(paths)
    elif command == "render":
        result = _render(paths, cast(Path, args.out))
    else:
        return AuditResult("INVALID_ARGUMENT", f"unknown command: {command}"), 2
    successful = result.status in {"AUDIT_COMPLETE", "EXPECTED_FAILURE_CONFIRMED"}
    return result, 0 if successful else 2


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_DEFAULT)
    parser.add_argument("--external-root", type=Path, default=None)
    _ = parser.add_argument("--asset-root", type=Path, default=None)
    _ = parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--live-root", type=Path, default=None)
    parser.add_argument("--staged-root", type=Path, default=None)
    parser.add_argument("--aggregate", type=Path, default=None)
    parser.add_argument("--manifest-sha256", default=None)


def _parser() -> JsonArgumentParser:
    parser = JsonArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "analyze"):
        _add_common(subcommands.add_parser(name))
    verify = subcommands.add_parser("verify")
    verify.add_argument("--case", choices=("happy", "failure"), required=True)
    _add_common(verify)
    render = subcommands.add_parser("render")
    render.add_argument("--out", type=Path, required=True)
    _add_common(render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except CliArgumentError as error:
        return _emit(AuditResult("INVALID_ARGUMENT", str(error)), 2)
    result, code = _result(arguments)
    return _emit(result, code)


if __name__ == "__main__":
    raise SystemExit(main())
