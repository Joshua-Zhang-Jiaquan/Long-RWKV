"""Frozen pin table for external suites: pure data, no network and no I/O.

Both the acquisition module (which fetches) and the qualification adapters (which
replay) import this module, so the pin a fetch writes is the same pin a verifier
checks against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ALLOWED_LICENSES: Final = frozenset({"MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause"})
ALLOWED_HOSTS: Final = frozenset({"raw.githubusercontent.com", "huggingface.co"})
SUITE_DIRECTORIES: Final = {
    "ruler_development_suite": "ruler",
    "longbench_development_suite": "longbench",
}
BUNDLE_NAME: Final = "source_bundle.json"
MANIFEST_NAME: Final = "development_manifest.json"
DEFAULT_ASSET_ROOT: Final = Path("DAN/nonlatent_iclr/task4_assets")

_COMMIT: Final = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class PinnedFile:
    """One upstream file frozen by destination path and content hash."""

    dest: str
    upstream_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PinnedSource:
    """A commit-pinned, license-cleared external suite."""

    requirement: str
    host: str
    repo: str
    commit: str
    license: str
    files: tuple[PinnedFile, ...]
    generators: tuple[str, ...]
    generator_config: str | None = None


class PinnedSourceError(LookupError):
    """Raised when no frozen pin exists for a declared requirement."""


# Every hash below was verified byte-identical against the live upstream commit before
# being frozen; the acquisition module re-checks them on every fetch.
PINNED_SOURCES: Final[tuple[PinnedSource, ...]] = (
    PinnedSource(
        requirement="ruler_development_suite",
        host="raw.githubusercontent.com",
        repo="NVIDIA/RULER",
        commit="c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
        license="Apache-2.0",
        files=(
            PinnedFile("source/LICENSE", "LICENSE", "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"),
            PinnedFile("source/synthetic.yaml", "scripts/synthetic.yaml", "34bc71dcacdc41a829a170f04b528fbf48d62c616338005ab4991680fbf8cb0b"),
            PinnedFile("source/variable_tracking.py", "scripts/data/synthetic/variable_tracking.py", "9aac483420e158d116ab63fc43b9606bdb284ac0c053288c30776d5c365530e5"),
        ),
        generators=("source/variable_tracking.py",),
        generator_config="source/synthetic.yaml",
    ),
    PinnedSource(
        requirement="longbench_development_suite",
        host="raw.githubusercontent.com",
        repo="THUDM/LongBench",
        commit="2e00731f8d0bff23dc4325161044d0ed8af94c1e",
        license="MIT",
        files=(
            PinnedFile("source/LICENSE", "LICENSE", "bf7f05f274d65931bdf14f060b6855f1c359b8424eb74fd3091d25de89d700c3"),
            PinnedFile("source/task_config.json", "LongBench/config/dataset2prompt.json", "56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30"),
        ),
        generators=("source/task_config.json",),
        generator_config="source/task_config.json",
    ),
)


def is_pinned_commit(commit: str) -> bool:
    """Report whether a revision is a full commit sha rather than a floating branch name."""
    return _COMMIT.fullmatch(commit) is not None


def source_for(requirement: str) -> PinnedSource:
    """Return the frozen pin for a declared requirement, or fail closed."""
    for source in PINNED_SOURCES:
        if source.requirement == requirement:
            return source
    raise PinnedSourceError(f"no frozen pin for requirement: {requirement}")
