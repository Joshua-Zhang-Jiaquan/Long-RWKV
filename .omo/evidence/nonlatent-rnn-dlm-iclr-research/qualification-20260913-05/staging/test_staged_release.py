from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[5]
SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "scale/experiments/nonlatent_iclr/qualification/runtime_manifest.json"
)
RELEASE_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-05-global-92fbf9af"
)
QUALIFICATION_RELATIVE = Path(
    "scale/experiments/nonlatent_iclr/qualification"
)
LAUNCHER_RELATIVE = QUALIFICATION_RELATIVE / "run_qualification.sh"
MANIFEST_RELATIVE = QUALIFICATION_RELATIVE / "runtime_manifest.json"
EXPECTED_PAYLOAD_RELATIVES = frozenset(
    {
        Path("scale/__init__.py"),
        Path("scale/experiments/__init__.py"),
        Path("scale/experiments/nonlatent_iclr/__init__.py"),
        Path("scale/experiments/nonlatent_iclr/contract.py"),
        Path("scale/experiments/nonlatent_iclr/inventory.py"),
        Path("scale/experiments/nonlatent_iclr/models.py"),
        Path("scale/experiments/nonlatent_iclr/publication.py"),
        Path("scale/experiments/nonlatent_iclr/schema.py"),
        QUALIFICATION_RELATIVE / "__init__.py",
        QUALIFICATION_RELATIVE / "cli.py",
        QUALIFICATION_RELATIVE / "contracts.py",
        QUALIFICATION_RELATIVE / "controller_receipt.py",
        QUALIFICATION_RELATIVE / "controls.py",
        QUALIFICATION_RELATIVE / "evidence_contracts.py",
        QUALIFICATION_RELATIVE / "manifest.py",
        QUALIFICATION_RELATIVE / "model_checks.py",
        QUALIFICATION_RELATIVE / "probe.py",
        QUALIFICATION_RELATIVE / "runtime_checks.py",
        QUALIFICATION_RELATIVE / "runtime_config.py",
        QUALIFICATION_RELATIVE / "runtime_identity.py",
        LAUNCHER_RELATIVE,
    }
)

type ManifestRole = Literal[
    "payload",
    "staged_source",
    "package_source",
    "hf_config",
    "hf_index",
    "hf_shard",
    "checkpoint_meta",
    "checkpoint_model",
]


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    role: ManifestRole
    sha256: str
    size_bytes: int


class PackageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution: str
    version: str


class RuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    payload_id: str
    expected_image: str
    files: tuple[ManifestFile, ...]
    packages: tuple[PackageIdentity, ...]


def _load_manifest(path: Path) -> RuntimeManifest:
    return RuntimeManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _release_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(RELEASE_ROOT),
        }
    )
    return environment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_payload_inventory_has_exact_transitive_import_closure() -> None:
    # Given: the immutable release manifest used by the worker launcher.
    manifest = _load_manifest(RELEASE_ROOT / MANIFEST_RELATIVE)

    # When: payload paths are interpreted relative to the release root.
    payload_relatives = frozenset(
        entry.path.relative_to(RELEASE_ROOT)
        for entry in manifest.files
        if entry.role == "payload"
    )

    # Then: every eager parent import and qualification module is bound exactly once.
    assert payload_relatives == EXPECTED_PAYLOAD_RELATIVES


def test_manifest_rebases_only_payload_paths_to_worker_release() -> None:
    # Given: the reviewed project manifest and its worker-visible successor.
    source = _load_manifest(SOURCE_MANIFEST)
    release = _load_manifest(RELEASE_ROOT / MANIFEST_RELATIVE)

    # When: non-payload records are compared as ordered typed values.
    source_external = tuple(entry for entry in source.files if entry.role != "payload")
    release_external = tuple(entry for entry in release.files if entry.role != "payload")

    # Then: staged source, package API, HF, checkpoint, and package identities are unchanged.
    assert release_external == source_external
    assert release.packages == source.packages
    assert release.expected_image == source.expected_image


def test_manifest_has_valid_worker_visible_root_mapping() -> None:
    # Given: every path bound by the staged runtime manifest.
    manifest = _load_manifest(RELEASE_ROOT / MANIFEST_RELATIVE)
    project_prefix = str(PROJECT_ROOT) + "/"

    # When: absolute path ownership is partitioned by manifest role.
    payload_paths = tuple(entry.path for entry in manifest.files if entry.role == "payload")

    # Then: payloads live below the release and no bound path retains the absent project root.
    assert all(path.is_absolute() and path.is_relative_to(RELEASE_ROOT) for path in payload_paths)
    assert all(not str(entry.path).startswith(project_prefix) for entry in manifest.files)
    assert RELEASE_ROOT not in tuple(entry.path for entry in manifest.files)


def test_source_blobs_remain_identical_except_diagnostic_launcher() -> None:
    # Given: reviewed project bytes and their staged payload counterparts.
    ordinary_relatives = EXPECTED_PAYLOAD_RELATIVES - {LAUNCHER_RELATIVE}

    # When: every source blob that needs no worker bootstrap adjustment is hashed.
    digest_pairs = tuple(
        (_sha256(PROJECT_ROOT / relative), _sha256(RELEASE_ROOT / relative))
        for relative in sorted(ordinary_relatives)
    )

    # Then: all ordinary Python/package bytes are exact copies.
    assert all(source_digest == release_digest for source_digest, release_digest in digest_pairs)


def test_stale_rebased_payload_hash_is_rejected_from_release_only(
    tmp_path: Path,
) -> None:
    # Given: a release manifest copy whose first payload digest is deliberately stale.
    manifest = _load_manifest(RELEASE_ROOT / MANIFEST_RELATIVE)
    first, *remaining = manifest.files
    assert first.role == "payload"
    stale = manifest.model_copy(
        update={
            "files": (
                first.model_copy(update={"sha256": "0" * 64}),
                *remaining,
            )
        }
    )
    stale_path = tmp_path / "runtime-manifest-stale.json"
    stale_path.write_text(stale.model_dump_json(indent=2) + "\n", encoding="utf-8")
    stale_digest = _sha256(stale_path)
    command = (
        "from pathlib import Path; "
        "from scale.experiments.nonlatent_iclr.qualification.contracts "
        "import QualificationInputError; "
        "from scale.experiments.nonlatent_iclr.qualification.manifest "
        "import verify_runtime_manifest; "
        "\ntry:\n"
        f" verify_runtime_manifest(Path({str(stale_path)!r}), expected_sha256={stale_digest!r})\n"
        "except QualificationInputError as error:\n"
        " print(str(error)); raise SystemExit(7) from None\n"
    )

    # When: public CPU-only verification runs with only the release on PYTHONPATH.
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: the relocated payload hash mismatch fails before external large-file hashing.
    assert result.returncode == 7
    assert result.stdout.strip() == "manifest_file_sha256_mismatch:__init__.py"


def test_cpu_entrypoint_imports_from_global_release_without_project_path(
    tmp_path: Path,
) -> None:
    # Given: an isolated process whose only application PYTHONPATH is the global release.
    arguments = [
        sys.executable,
        "-m",
        "scale.experiments.nonlatent_iclr.qualification.cli",
        "validate",
        "--output-dir",
        str(tmp_path / "unused-output"),
        "--nonce",
        "bad",
    ]

    # When: the real CPU argument boundary runs outside the project checkout.
    result = subprocess.run(
        arguments,
        cwd=tmp_path,
        env=_release_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: it returns structured input evidence without CUDA or the project mount.
    assert result.returncode == 2
    assert json.loads(result.stdout) == {
        "detail": "nonce_must_be_32_lower_hex_characters",
        "status": "INVALID_ARGUMENT",
    }


def test_launcher_relative_repo_root_is_the_frozen_release() -> None:
    # Given: the exact launcher path to be handed to the independent controller.
    launcher = RELEASE_ROOT / LAUNCHER_RELATIVE

    # When: its reviewed four-level relative root calculation is evaluated.
    computed_root = launcher.parent.parents[3]

    # Then: PYTHONPATH's repository component is the release, not the absent project mount.
    assert computed_root == RELEASE_ROOT
    assert 'export PYTHONPATH="${REPO_ROOT}:${STAGED_ROOT}"' in launcher.read_text(
        encoding="utf-8"
    )


def test_missing_release_manifest_publishes_concrete_early_diagnostic(
    tmp_path: Path,
) -> None:
    # Given: a disposable launcher copy with a missing colocated manifest and private output root.
    launcher = tmp_path / LAUNCHER_RELATIVE
    launcher.parent.mkdir(parents=True)
    shutil.copy2(RELEASE_ROOT / LAUNCHER_RELATIVE, launcher)
    output_root = tmp_path / "global-output"
    text = launcher.read_text(encoding="utf-8")
    launcher.write_text(
        text.replace(
            "readonly DEFAULT_OUTPUT_ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification",
            f"readonly DEFAULT_OUTPUT_ROOT={output_root}",
        ),
        encoding="utf-8",
    )
    run_id = "qualification-20260912-01-staging-red"
    environment = _release_environment()
    environment.update(
        {
            "QUALIFICATION_JOB_RECEIPT": str(tmp_path / "controller-receipt.json"),
            "QUALIFICATION_MANIFEST_SHA256": "a" * 64,
            "QUALIFICATION_NONCE": "b" * 32,
            "QUALIFICATION_RUN_ID": run_id,
            "QUALIFICATION_WORKFLOW_INNER": "1",
        }
    )

    # When: startup reaches the absent source-manifest boundary before Python validation.
    result = subprocess.run(
        ["/usr/bin/bash", str(launcher)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: a concrete terminal diagnostic exists under shared output without project writes.
    diagnostic = json.loads(
        (output_root / run_id / "early-failure.json").read_text(encoding="utf-8")
    )
    assert result.returncode == 2
    assert diagnostic == {
        "detail": "source_manifest_unavailable",
        "phase": "startup",
        "run_id": run_id,
        "schema_version": 1,
        "status": "FAILED",
    }
