from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .contract import METADATA_PATHS, PANEL_PATHS, SNAPSHOT_ARTIFACTS

SOURCES = SNAPSHOT_ARTIFACTS
PANEL_NAMES = PANEL_PATHS
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_PANEL_BYTES = 512 * 1024 * 1024
MAX_PANEL_FILES = 512
RAW_SUFFIXES = frozenset({".json", ".csv"})


@dataclass(frozen=True, slots=True)
class Artifact:
    source: str
    origin: str
    path: str
    kind: str
    identity: str
    sha256: str | None
    bytes: int | None
    file_count: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class View:
    source: str
    state: str
    detail: str


class FileLimitError(OSError):
    def __init__(self, size: int) -> None:
        super().__init__("file byte limit exceeded")
        self.size = size


class SourceChangedError(OSError):
    pass


def _safe_target(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe inventory path: {relative}")
    target = root.joinpath(*path.parts)
    if not target.resolve(strict=False).is_relative_to(root.resolve()):
        raise ValueError(f"inventory path escapes root: {relative}")
    return target


def _hash_regular(path: Path, limit: int | None) -> tuple[str, int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("source is not regular")
        if limit is not None and before.st_size > limit:
            raise FileLimitError(before.st_size)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise OSError("source changed while hashing")
        return digest.hexdigest(), before.st_size
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    return _hash_regular(path, None)[0]


def file_artifact(
    root: Path,
    source: str,
    origin: str,
    relative: str,
    kind: str,
    hashed_reason: str | None = None,
) -> Artifact:
    target = _safe_target(root, relative)
    try:
        digest, size = _hash_regular(target, MAX_FILE_BYTES)
    except FileLimitError as error:
        return Artifact(source, origin, relative, kind, "unverified", None, error.size, 1, "per-file byte limit exceeded")
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return Artifact(source, origin, relative, kind, "missing", None, None, 0, "absent or non-regular")
    except OSError:
        if target.is_symlink():
            return Artifact(source, origin, relative, kind, "missing", None, None, 0, "unsafe symlink")
        size = target.stat().st_size if target.is_file() else None
        count = 1 if size is not None else 0
        return Artifact(source, origin, relative, kind, "unverified", None, size, count, "regular file could not be hashed")
    return Artifact(source, origin, relative, kind, "sha256", digest, size, 1, hashed_reason)


def _panel_files(target: Path) -> tuple[list[Path], bool]:
    files: list[Path] = []
    unsafe = False
    for directory, directories, names in os.walk(target, topdown=True, followlinks=False):
        base = Path(directory)
        safe_directories: list[str] = []
        for name in sorted(directories):
            candidate = base / name
            if candidate.is_symlink():
                unsafe = True
            else:
                safe_directories.append(name)
        directories[:] = safe_directories
        for name in sorted(names):
            candidate = base / name
            if candidate.suffix not in RAW_SUFFIXES:
                continue
            if candidate.is_symlink() or not candidate.is_file():
                unsafe = True
            else:
                files.append(candidate)
    return sorted(files), unsafe


def panel_artifact(root: Path, relative: str, max_files: int = MAX_PANEL_FILES) -> Artifact:
    target = _safe_target(root, relative)
    if target.is_file() and not target.is_symlink():
        return file_artifact(root, "R3", "external", relative, "raw_record", "bounded raw file")
    if target.is_symlink() or not target.is_dir():
        return Artifact("R3", "external", relative, "raw_record", "missing", None, None, 0, "named panel absent or unsafe")
    files, unsafe = _panel_files(target)
    try:
        sizes = tuple(path.stat().st_size for path in files)
    except OSError as error:
        raise SourceChangedError(f"panel changed during enumeration: {relative}") from error
    total = sum(sizes)
    if not files:
        return Artifact("R3", "external", relative, "raw_record", "missing", None, None, 0, "panel has no JSON/CSV records")
    reason: str | None = None
    if unsafe:
        reason = "panel contains unsafe entries"
    elif len(files) > max_files:
        reason = f"panel file limit exceeded ({max_files})"
    elif total > MAX_PANEL_BYTES or any(size > MAX_FILE_BYTES for size in sizes):
        reason = "panel byte limit exceeded"
    if reason is not None:
        return Artifact("R3", "external", relative, "raw_record", "unverified", None, total, len(files), reason)
    manifest = hashlib.sha256()
    try:
        for path, size in zip(files, sizes, strict=True):
            digest, current_size = _hash_regular(path, MAX_FILE_BYTES)
            relative_file = path.relative_to(target).as_posix()
            manifest.update(f"{relative_file}\0{current_size}\0{digest}\n".encode())
            if current_size != size:
                raise SourceChangedError(f"panel changed while hashing: {relative}")
    except SourceChangedError:
        raise
    except FileNotFoundError as error:
        raise SourceChangedError(f"panel changed while hashing: {relative}") from error
    except OSError:
        return Artifact("R3", "external", relative, "raw_record", "unverified", None, total, len(files), "panel changed or could not be hashed")
    return Artifact("R3", "external", relative, "raw_record", "sha256", manifest.hexdigest(), total, len(files), f"bounded manifest of {len(files)} files")


def build_artifacts(snapshot_root: Path, external_root: Path, max_panel_files: int) -> tuple[Artifact, ...]:
    snapshot = tuple(file_artifact(snapshot_root, source, "snapshot", path, kind) for source, path, kind in SOURCES)
    panels = tuple(panel_artifact(external_root, path, max_panel_files) for path in PANEL_NAMES)
    metadata = tuple(
        file_artifact(external_root, "R6", "external", path, "metadata", "metadata identity; payload inference blocked")
        for path in METADATA_PATHS
    )
    return snapshot + panels + metadata


def observe_artifact(artifact: Artifact, snapshot_root: Path, external_root: Path, max_panel_files: int) -> Artifact:
    root = snapshot_root if artifact.origin == "snapshot" else external_root
    if artifact.origin == "external" and artifact.kind == "raw_record" and artifact.path in PANEL_NAMES:
        return panel_artifact(root, artifact.path, max_panel_files)
    reason = "metadata identity; payload inference blocked" if artifact.kind == "metadata" else None
    return file_artifact(root, artifact.source, artifact.origin, artifact.path, artifact.kind, reason)


def build_views(snapshot_root: Path, live_root: Path, staged_root: Path) -> tuple[View, ...]:
    def observe(name: str, root: Path) -> View:
        if root.is_symlink():
            return View(name, "unresolved", f"unsafe symlink: {root}")
        if root.is_dir():
            return View(name, "observed", f"present directory: {root}")
        return View(name, "unresolved", f"absent directory: {root}")

    return tuple(observe(name, root) for name, root in (("snapshot", snapshot_root), ("live", live_root), ("staged", staged_root)))
