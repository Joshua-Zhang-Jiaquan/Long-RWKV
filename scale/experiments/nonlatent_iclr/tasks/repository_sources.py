"""Pinned permissively-licensed repositories used as the source of repository tasks.

Pure data plus a fetching step, mirroring ``pinned_sources``: the table is frozen in code,
the selection rule is frozen in code, and the resulting bytes are hash-bound in the bundle
that the qualification adapter replays. Like ``asset_acquisition`` this module touches the
network and must not be imported by the verification path.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from typing import Final

from .asset_acquisition import scrubbed_network_env
from .pinned_sources import ALLOWED_LICENSES

MAX_SOURCE_FILE_BYTES: Final = 12_000
MAX_FILES_PER_REPOSITORY: Final = 1500
MAX_CANDIDATES_PER_ENTRY: Final = 60
TARGET_TASK_COUNT: Final = 200
EXCLUDED_PATH_MARKERS: Final = ("test", "conftest", "setup.py", "docs/", "example", "__main__", "benchmark")
_SHA: Final = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class PinnedRepository:
    """One commit-pinned, license-cleared upstream repository."""

    repository: str
    commit: str
    license: str
    subtree: str

    @property
    def family(self) -> str:
        """Stable family id for split assignment and held-out grouping."""
        from scale.data.split_assign import family_id

        return family_id(self.repository)


_ALGORITHMS_PYTHON: Final = "TheAlgorithms/Python"
_ALGORITHMS_COMMIT: Final = "3f8772c07acb9efe60d105dfc311fc25926bcd04"
_ALGORITHMS_SUBTREES: Final = (
    "maths", "strings", "conversions", "bit_manipulation", "boolean_algebra", "ciphers",
    "compression", "dynamic_programming", "greedy_methods", "hashes", "knapsack",
    "linear_algebra", "matrix", "physics", "scheduling", "searches", "sorts", "graphs",
    "data_structures", "digital_image_processing", "geometry", "fractals", "quantum",
    "project_euler", "other", "cellular_automata", "electronics", "financial",
    "fuzzy_logic", "genetic_algorithm", "geodesy", "graphics", "machine_learning",
    "neural_network", "networking_flow", "computer_vision",
)

PINNED_REPOSITORIES: Final[tuple[PinnedRepository, ...]] = (
    *(PinnedRepository(_ALGORITHMS_PYTHON, _ALGORITHMS_COMMIT, "MIT", subtree) for subtree in _ALGORITHMS_SUBTREES),
    PinnedRepository("keon/algorithms", "7f71a911232b2ef4f5396f16f5ea704d4315ce0c", "MIT", "algorithms"),
    PinnedRepository("pytoolz/toolz", "568c2b8393973cd172a466546c9d95779c452438", "BSD-3-Clause", "toolz"),
    PinnedRepository("mahmoud/boltons", "961dcff3f42e73b245aef65e377fe82763b257bb", "BSD-3-Clause", "boltons"),
    PinnedRepository("pallets/itsdangerous", "672971d66a2ef9f85151e53283113f33d642dabd", "BSD-3-Clause", "src"),
    PinnedRepository("python-jsonschema/jsonschema", "865c27fc3df08a082740d7743583795235fdfe81", "MIT", "jsonschema"),
    PinnedRepository("jd/tenacity", "3e58094d3bc414975aad9eadf343a32bdb3b89b3", "Apache-2.0", "tenacity"),
    PinnedRepository("more-itertools/more-itertools", "9ed3dbb0ae527230cd156d91d0af305478558fba", "MIT", "more_itertools"),
    PinnedRepository("OmkarPathak/pygorithm", "cee38ed47e671171c3a30d12d5ec288f19c206d8", "MIT", ""),
    PinnedRepository("prabhupant/python-ds", "35d3556a992ceccc5b925afa892fae3ba01a0e81", "MIT", ""),
    PinnedRepository("geekcomputers/Python", "808584709a9b710d7e0ece36a6ea4ffcf109b537", "MIT", ""),
    PinnedRepository("rougier/numpy-100", "8ca28aaefca69a32ad3ccbbd01c7c9a785c2d5c1", "MIT", ""),
)


class RepositorySourceError(RuntimeError):
    """Raised when a pinned repository cannot be listed or fetched safely."""


MAX_ARCHIVE_BYTES: Final = 200 * 1024 * 1024


def archive_url(repository: PinnedRepository) -> str:
    """Commit-pinned tarball URL. Codeload is not rate-limited like the JSON API."""
    if repository.license not in ALLOWED_LICENSES:
        raise RepositorySourceError(f"license is not allowlisted: {repository.license}")
    if not _SHA.fullmatch(repository.commit):
        raise RepositorySourceError(f"commit is not a full 40-hex sha: {repository.commit}")
    return f"https://codeload.github.com/{repository.repository}/tar.gz/{repository.commit}"


def _fetch_archive_bytes(repository: str, commit: str, license_id: str, timeout: float) -> bytes:
    return fetch_archive(PinnedRepository(repository, commit, license_id, ""), timeout)


def fetch_archive(repository: PinnedRepository, timeout: float = 300.0) -> bytes:
    """Download one pinned repository tarball over direct HTTPS with proxies scrubbed."""
    url = archive_url(repository)
    completed = subprocess.run(
        ["curl", "--noproxy", "*", "-fsSL", "--max-time", str(int(timeout)), url],
        capture_output=True, env=scrubbed_network_env(), check=False,
    )
    if completed.returncode != 0:
        raise RepositorySourceError(f"archive fetch failed for {url}: curl exit {completed.returncode}")
    if len(completed.stdout) > MAX_ARCHIVE_BYTES:
        raise RepositorySourceError(f"archive exceeds the size cap for {repository.repository}")
    return completed.stdout


def select_sources(repository: PinnedRepository, archive: bytes) -> tuple[tuple[str, bytes], ...]:
    """Apply the frozen selection rule to the pinned tree; never writes to disk."""
    import io
    import tarfile

    selected: list[tuple[str, bytes]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                parts = member.name.split("/", 1)
                if len(parts) != 2:
                    continue
                path, size = parts[1], member.size
                if not path.endswith(".py") or not path.startswith(repository.subtree):
                    continue
                if size > MAX_SOURCE_FILE_BYTES:
                    continue
                if any(marker in path.lower() for marker in EXCLUDED_PATH_MARKERS):
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                selected.append((path, handle.read()))
    except tarfile.TarError as error:
        raise RepositorySourceError(f"archive is not a readable tarball: {error}") from error
    return tuple(sorted(selected)[:MAX_FILES_PER_REPOSITORY])


@lru_cache(maxsize=8)
def _cached_archive(repository: str, commit: str, license_id: str, timeout: float) -> bytes:
    return _fetch_archive_bytes(repository, commit, license_id, timeout)


def fetch_sources(repository: PinnedRepository, timeout: float = 300.0) -> tuple[tuple[str, bytes], ...]:
    """Fetch one repository once and return its selected source files.

    Archives are cached per pinned commit, so several subtree entries over one repository
    cost a single download.
    """
    archive = _cached_archive(repository.repository, repository.commit, repository.license, timeout)
    return select_sources(repository, archive)


def file_sha256(data: bytes) -> str:
    return sha256(data).hexdigest()
