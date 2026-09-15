"""Pinned acquisition of external benchmark sources; never imported by verification.

Run from the repository root, for example::

    python -m scale.experiments.nonlatent_iclr.tasks.asset_acquisition fetch --dry-run

This is the only Task4 module permitted to touch the network. The verification path
(``external_assets``, ``qualification_adapters``, ``registry_task``) must never import it;
those modules use the network-free ``pinned_sources`` table instead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

from .pinned_sources import (
    ALLOWED_HOSTS, ALLOWED_LICENSES, BUNDLE_NAME, DEFAULT_ASSET_ROOT, MANIFEST_NAME,
    PINNED_SOURCES, SUITE_DIRECTORIES, PinnedFile, PinnedSource, PinnedSourceError,
    is_pinned_commit, source_for,
)

PROXY_VARIABLES: Final = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy")

__all__ = [
    "ALLOWED_HOSTS", "ALLOWED_LICENSES", "BUNDLE_NAME", "DEFAULT_ASSET_ROOT", "MANIFEST_NAME",
    "PINNED_SOURCES", "PROXY_VARIABLES", "SUITE_DIRECTORIES", "AcquisitionConflict",
    "AcquisitionError", "AcquisitionReceipt", "PinnedFile", "PinnedSource", "acquire",
    "build_bundle", "bundle_bytes", "fetch_file", "is_pinned_commit", "main", "raw_url",
    "scrubbed_network_env", "verify_acquired",
]


@dataclass(frozen=True, slots=True)
class AcquisitionReceipt:
    """Bound record of one acquisition attempt."""

    requirement: str
    commit: str
    license: str
    files_written: int
    files_unchanged: int
    files_archived: int
    bundle_sha256: str
    network_env_scrubbed: bool
    dry_run: bool


class AcquisitionError(RuntimeError):
    """Raised when a pinned source cannot be acquired safely."""


class AcquisitionConflict(RuntimeError):
    """Raised instead of silently replacing an existing declaration."""


def scrubbed_network_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Remove every proxy variable so the pinned fetch cannot be silently proxied."""
    env = dict(os.environ if base is None else base)
    for name in PROXY_VARIABLES:
        _ = env.pop(name, None)
    entries = [item for item in env.get("no_proxy", "").split(",") if item]
    if "*" not in entries:
        entries.append("*")
    for name in ("no_proxy", "NO_PROXY"):
        env[name] = ",".join(entries)
    return env


def raw_url(source: PinnedSource, entry: PinnedFile) -> str:
    """Derive a commit-pinned URL, rejecting unpinned or non-permissive sources."""
    if not is_pinned_commit(source.commit):
        raise AcquisitionError(f"commit is not a full 40-hex sha: {source.commit}")
    if source.host not in ALLOWED_HOSTS:
        raise AcquisitionError(f"host is not allowlisted: {source.host}")
    if source.license not in ALLOWED_LICENSES:
        raise AcquisitionError(f"license is not allowlisted: {source.license}")
    return f"https://{source.host}/{source.repo}/{source.commit}/{entry.upstream_path}"


def fetch_file(source: PinnedSource, entry: PinnedFile, timeout: float = 60.0) -> bytes:
    """Fetch one pinned file over direct HTTPS with the proxy environment scrubbed."""
    url = raw_url(source, entry)
    completed = subprocess.run(
        ["curl", "--noproxy", "*", "-fsSL", "--max-time", str(int(timeout)), url],
        capture_output=True, env=scrubbed_network_env(), check=False,
    )
    if completed.returncode != 0:
        raise AcquisitionError(f"fetch failed for {url}: curl exit {completed.returncode}")
    return completed.stdout


def build_bundle(source: PinnedSource) -> dict[str, object]:
    """Describe the pin; the bundle's own bytes become the manifest's trusted root."""
    if not source.files:
        raise AcquisitionError(f"pinned source declares no files: {source.requirement}")
    _ = raw_url(source, source.files[0])
    return {
        "schema_version": 1,
        "suite": SUITE_DIRECTORIES[source.requirement],
        "upstream": {"host": source.host, "repo": source.repo, "commit": source.commit, "license": source.license},
        "files": [
            {"dest": entry.dest, "upstream_path": entry.upstream_path, "sha256": entry.sha256}
            for entry in sorted(source.files, key=lambda item: item.dest)
        ],
        "generators": sorted(source.generators),
        "generator_config": source.generator_config,
        "acquisition": "per-file HTTPS at a pinned commit; proxy environment scrubbed",
    }


def bundle_bytes(bundle: Mapping[str, object]) -> bytes:
    """Canonical serialization so the bound hash is reproducible."""
    return (json.dumps(dict(bundle), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _source_for(requirement: str) -> PinnedSource:
    try:
        return source_for(requirement)
    except PinnedSourceError as error:
        raise AcquisitionError(str(error)) from error


def _write_pinned_file(path: Path, data: bytes) -> str:
    """Write pinned bytes, archiving rather than discarding any differing local copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        _ = path.write_bytes(data)
        return "created"
    existing = path.read_bytes()
    if existing == data:
        return "unchanged"
    archive = path.with_name(f"{path.name}.archive-{sha256(existing).hexdigest()[:16]}")
    if not archive.exists():
        _ = archive.write_bytes(existing)
    _ = path.write_bytes(data)
    return "archived"


def _write_declaration(path: Path, data: bytes) -> bool:
    """Publish a declaration once; a conflicting existing declaration is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() == data:
            return False
        raise AcquisitionConflict(f"declaration already exists with different content: {path}")
    _ = path.write_bytes(data)
    return True


def acquire(
    requirement: str,
    dest_root: Path,
    *,
    dry_run: bool = False,
    fetch: Callable[[PinnedSource, PinnedFile], bytes] | None = None,
    source: PinnedSource | None = None,
) -> AcquisitionReceipt:
    """Fetch, verify, and pin one suite; nothing is written unless every byte matches.

    ``source`` overrides the frozen table for hermetic tests; production callers omit it.
    """
    pinned = _source_for(requirement) if source is None else source
    suite = dest_root / SUITE_DIRECTORIES[requirement]
    transport = fetch_file if fetch is None else fetch
    payloads: dict[str, bytes] = {}
    for entry in pinned.files:
        data = transport(pinned, entry)
        if sha256(data).hexdigest() != entry.sha256:
            raise AcquisitionError(f"fetched bytes do not match the frozen pin for {entry.dest}")
        payloads[entry.dest] = data
    bundle = build_bundle(pinned)
    bundle_digest = sha256(bundle_bytes(bundle)).hexdigest()
    if dry_run:
        return AcquisitionReceipt(requirement, pinned.commit, pinned.license, 0, 0, 0, bundle_digest, True, True)
    written = unchanged = archived = 0
    for dest in sorted(payloads):
        match _write_pinned_file(suite / dest, payloads[dest]):
            case "created":
                written += 1
            case "unchanged":
                unchanged += 1
            case _:
                archived += 1
    _ = _write_declaration(suite / BUNDLE_NAME, bundle_bytes(bundle))
    manifest = {
        "version": f"{SUITE_DIRECTORIES[requirement]}@{pinned.commit}",
        "license": pinned.license,
        "sha256": bundle_digest,
        "asset_path": BUNDLE_NAME,
        "upstream_repo": pinned.repo,
        "upstream_commit": pinned.commit,
        "network_env_scrubbed": True,
    }
    _ = _write_declaration(suite / MANIFEST_NAME, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return AcquisitionReceipt(requirement, pinned.commit, pinned.license, written, unchanged, archived, bundle_digest, True, False)


def verify_acquired(requirement: str, dest_root: Path, *, source: PinnedSource | None = None) -> tuple[str, ...]:
    """Offline check that local bytes still satisfy the frozen pin and declarations."""
    pinned = _source_for(requirement) if source is None else source
    suite = dest_root / SUITE_DIRECTORIES[requirement]
    problems: list[str] = []
    for entry in pinned.files:
        path = suite / entry.dest
        if not path.is_file():
            problems.append(f"missing vendored file {entry.dest}")
        elif sha256(path.read_bytes()).hexdigest() != entry.sha256:
            problems.append(f"vendored file does not match the frozen pin: {entry.dest}")
    for name in (BUNDLE_NAME, MANIFEST_NAME):
        if not (suite / name).is_file():
            problems.append(f"missing declaration {name}")
    return tuple(problems)


def _write_receipt(evidence_root: Path, receipt: AcquisitionReceipt) -> Path:
    attempt = evidence_root / f"acquisition-{uuid.uuid4().hex}"
    attempt.mkdir(parents=True, exist_ok=False)
    path = attempt / "result.json"
    _ = path.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class CliOptions:
    """Typed view of one parsed command line."""

    command: str
    requirements: tuple[str, ...]
    asset_root: Path
    evidence_root: Path | None
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("command", choices=("fetch", "verify", "list"))
    _ = parser.add_argument("--requirement", default=None, help="comma-separated requirement names")
    _ = parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    _ = parser.add_argument("--evidence-root", type=Path, default=None)
    _ = parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_options(argv: tuple[str, ...] | None) -> CliOptions:
    """Convert the argparse namespace to declared types once, at the boundary."""
    raw: dict[str, object] = dict(vars(_build_parser().parse_args(argv)))
    declared = raw.get("requirement")
    selected = tuple(part.strip() for part in str(declared).split(",") if part.strip()) if declared is not None else ()
    evidence = raw.get("evidence_root")
    return CliOptions(
        command=str(raw.get("command", "")),
        requirements=selected,
        asset_root=Path(str(raw.get("asset_root", DEFAULT_ASSET_ROOT))),
        evidence_root=None if evidence is None else Path(str(evidence)),
        dry_run=bool(raw.get("dry_run", False)),
    )


def main(argv: tuple[str, ...] | None = None) -> int:
    """Offline-plannable CLI: fetch|verify|list with an explicit asset root."""
    options = _parse_options(argv)
    selected = options.requirements or tuple(source.requirement for source in PINNED_SOURCES)
    if options.command == "list":
        for source in PINNED_SOURCES:
            print(f"{source.requirement} {source.repo}@{source.commit[:12]} {source.license} files={len(source.files)}")
        return 0
    if options.command == "verify":
        failed = False
        for requirement in selected:
            problems = verify_acquired(requirement, options.asset_root)
            for problem in problems:
                print(f"{requirement}: {problem}")
            failed = failed or bool(problems)
            print(f"{requirement}: {'MISMATCH' if problems else 'PINNED'}")
        return 2 if failed else 0
    for requirement in selected:
        receipt = acquire(requirement, options.asset_root, dry_run=options.dry_run)
        print(f"{requirement}: written={receipt.files_written} unchanged={receipt.files_unchanged} archived={receipt.files_archived} bundle={receipt.bundle_sha256[:16]}")
        if options.evidence_root is not None and not options.dry_run:
            print(f"receipt={_write_receipt(options.evidence_root, receipt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
