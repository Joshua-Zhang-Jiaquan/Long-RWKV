from __future__ import annotations

import errno
import os
from pathlib import Path
from uuid import uuid4


class UnsafePathError(ValueError):
    pass


def _parts(relative: Path) -> tuple[str, ...]:
    parts = relative.parts
    if relative.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise UnsafePathError(f"unsafe relative destination: {relative}")
    return parts


def _root_fd(root: Path) -> int:
    if not root.is_absolute() or root.is_symlink():
        raise UnsafePathError(f"unsafe publication root: {root}")
    try:
        root.mkdir(parents=True, exist_ok=True)
        return os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except (FileExistsError, NotADirectoryError) as error:
        raise UnsafePathError(f"unsafe publication root: {root}") from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise UnsafePathError(f"unsafe publication root: {root}") from error
        raise


def _directory_fd(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o755, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise UnsafePathError(f"unsafe publication parent: {name}") from error
        raise


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError("short publication write")
        remaining = remaining[written:]


def write_once(root: Path, relative: Path, data: bytes) -> bool:
    parts = _parts(relative)
    descriptors = [_root_fd(root)]
    temporary = f".{parts[-1]}.{uuid4().hex}.tmp"
    try:
        for part in parts[:-1]:
            descriptors.append(_directory_fd(descriptors[-1], part))
        parent_fd = descriptors[-1]
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            _write_all(temporary_fd, data)
            os.fsync(temporary_fd)
        finally:
            os.close(temporary_fd)
        try:
            os.link(
                temporary,
                parts[-1],
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            return False
        os.fsync(parent_fd)
        return True
    finally:
        try:
            os.unlink(temporary, dir_fd=descriptors[-1])
        except FileNotFoundError:
            pass
        for descriptor in reversed(descriptors):
            os.close(descriptor)
