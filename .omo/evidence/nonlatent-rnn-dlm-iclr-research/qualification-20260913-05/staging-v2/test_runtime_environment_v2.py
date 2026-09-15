from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[5]
RELEASE = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260913-05-global-ffaa464d-v2"
)
QUALIFICATION = Path("scale/experiments/nonlatent_iclr/qualification")
HELPER = QUALIFICATION / "runtime_environment.py"
MANIFEST = QUALIFICATION / "runtime_manifest.json"
LAUNCHER = QUALIFICATION / "run_qualification.sh"
ALIAS = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "m2_baseline_triangle/m4loop_endpoint_ckpt"
)
STAGED_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "qz_stage_traj4096_v7/scale"
)
MODEL_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "models/RWKV7-Goose-World3-2.9B-HF"
)
MAX_PACKAGE_SOURCE_BYTES = 1_048_576


class PackageObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution: str
    expected_version: str
    observed_version: str | None
    status: Literal["MATCH", "MISSING", "VERSION_MISMATCH"]


class SourceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_path: Path
    resolved_path: Path | None
    expected_size_bytes: int
    observed_size_bytes: int | None
    expected_sha256: str
    observed_sha256: str | None
    status: Literal[
        "MATCH",
        "MISSING",
        "SYMLINK",
        "NOT_REGULAR",
        "TOO_LARGE",
        "SIZE_MISMATCH",
        "SHA256_MISMATCH",
        "UNREADABLE",
    ]


class RootObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["release", "staged_source", "checkpoint", "model", "manifest"]
    requested_path: Path
    resolved_path: Path
    exists: bool
    is_directory: bool
    is_symlink: bool


class RuntimeEnvironmentDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    status: Literal["MATCH", "MISMATCH"]
    manifest_sha256: str
    python_executable: Path
    python_version: str
    cwd: Path
    sys_path: tuple[str, ...]
    runtime_modules_loaded: tuple[str, ...]
    packages: tuple[PackageObservation, ...]
    package_sources: tuple[SourceObservation, ...]
    roots: tuple[RootObservation, ...]


def _environment(release: Path = RELEASE) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{release}:{STAGED_ROOT}",
        }
    )
    return environment


def _helper_arguments(root: Path, manifest: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(root / HELPER),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--release-root",
        str(root),
        "--staged-root",
        str(STAGED_ROOT),
        "--checkpoint-root",
        str(ALIAS),
        "--model-root",
        str(MODEL_ROOT),
    ]


def _run_helper(
    root: Path, manifest: Path, output: Path, cwd: Path
) -> tuple[subprocess.CompletedProcess[str], RuntimeEnvironmentDiagnostic]:
    result = subprocess.run(
        _helper_arguments(root, manifest, output),
        cwd=cwd,
        env=_environment(root),
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostic = RuntimeEnvironmentDiagnostic.model_validate_json(
        output.read_text(encoding="utf-8")
    )
    return result, diagnostic


def _write_synthetic_manifest(
    path: Path, distribution: str, source_path: Path
) -> None:
    payload = {
        "files": [
            {
                "path": str(source_path),
                "role": "package_source",
                "sha256": "0" * 64,
                "size_bytes": 1,
            }
        ],
        "packages": [{"distribution": distribution, "version": "expected-v1"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_real_diagnostic_records_six_packages_and_small_source_hashes(
    tmp_path: Path,
) -> None:
    # Given: the real versioned manifest and a fresh diagnostic destination.
    output = tmp_path / "runtime_environment.json"

    # When: the standalone stdlib helper observes metadata and small source files.
    result, diagnostic = _run_helper(RELEASE, RELEASE / MANIFEST, output, tmp_path)

    # Then: file/stdout agree, six correct names are observed, and no runtime library loads.
    assert result.returncode == 0
    assert json.loads(result.stdout) == json.loads(output.read_text(encoding="utf-8"))
    assert tuple(item.distribution for item in diagnostic.packages) == (
        "torch",
        "flash-linear-attention",
        "fla-core",
        "transformers",
        "safetensors",
        "pytorch-triton",
    )
    assert len(diagnostic.package_sources) == 8
    assert all(
        item.observed_size_bytes is not None
        and item.observed_size_bytes <= MAX_PACKAGE_SOURCE_BYTES
        for item in diagnostic.package_sources
    )
    assert diagnostic.runtime_modules_loaded == ()


def test_missing_distribution_is_recorded_without_import_or_auto_bless(
    tmp_path: Path,
) -> None:
    # Given: a manifest naming one definitely absent distribution and an existing small file.
    manifest = tmp_path / "missing-package.json"
    _write_synthetic_manifest(manifest, "qualification-definitely-missing", Path(__file__))
    output = tmp_path / "runtime_environment.json"

    # When: metadata-only observation runs without importing that distribution.
    result, diagnostic = _run_helper(RELEASE, manifest, output, tmp_path)

    # Then: absence is explicit, expected remains separate, and diagnostic status is not PASS.
    package = diagnostic.packages[0]
    assert result.returncode == 0
    assert diagnostic.status == "MISMATCH"
    assert package.expected_version == "expected-v1"
    assert package.observed_version is None
    assert package.status == "MISSING"
    assert diagnostic.runtime_modules_loaded == ()


def test_missing_package_source_is_recorded_without_hashing_other_roles(
    tmp_path: Path,
) -> None:
    # Given: a present distribution with a missing package-source path.
    manifest = tmp_path / "missing-source.json"
    _write_synthetic_manifest(manifest, "pydantic", tmp_path / "absent.py")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["packages"][0]["version"] = metadata.version("pydantic")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "runtime_environment.json"

    # When: bounded source observation runs.
    result, diagnostic = _run_helper(RELEASE, manifest, output, tmp_path)

    # Then: package metadata matches while the missing source remains a mismatch.
    assert result.returncode == 0
    assert diagnostic.packages[0].status == "MATCH"
    assert diagnostic.package_sources[0].status == "MISSING"
    assert diagnostic.package_sources[0].observed_sha256 is None
    assert diagnostic.status == "MISMATCH"


def test_oversized_package_source_is_not_hashed(tmp_path: Path) -> None:
    # Given: a package-source candidate one byte above the diagnostic hash bound.
    oversized = tmp_path / "oversized.py"
    oversized.write_bytes(b"x" * (MAX_PACKAGE_SOURCE_BYTES + 1))
    manifest = tmp_path / "oversized-source.json"
    _write_synthetic_manifest(manifest, "pydantic", oversized)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["packages"][0]["version"] = metadata.version("pydantic")
    payload["files"][0]["size_bytes"] = MAX_PACKAGE_SOURCE_BYTES + 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "runtime_environment.json"

    # When: the stdlib observer reaches that path.
    _, diagnostic = _run_helper(RELEASE, manifest, output, tmp_path)

    # Then: it records the bound violation without producing a digest.
    assert diagnostic.package_sources[0].status == "TOO_LARGE"
    assert diagnostic.package_sources[0].observed_sha256 is None


def test_launcher_publishes_environment_before_strict_manifest_failure(
    tmp_path: Path,
) -> None:
    # Given: a disposable complete release with a private output root and stale pin.
    copied_release = tmp_path / "release"
    shutil.copytree(RELEASE, copied_release)
    launcher = copied_release / LAUNCHER
    launcher.chmod(0o700)
    output_root = tmp_path / "global-output"
    launcher.write_text(
        launcher.read_text(encoding="utf-8").replace(
            "readonly DEFAULT_OUTPUT_ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification",
            f"readonly DEFAULT_OUTPUT_ROOT={output_root}",
        ),
        encoding="utf-8",
    )
    run_id = "qualification-20260912-01-envdiag1"
    environment = _environment(copied_release)
    environment.update(
        {
            "QUALIFICATION_JOB_RECEIPT": str(tmp_path / "controller.json"),
            "QUALIFICATION_MANIFEST_SHA256": "0" * 64,
            "QUALIFICATION_NONCE": "a" * 32,
            "QUALIFICATION_RUN_ID": run_id,
            "QUALIFICATION_WORKFLOW_INNER": "1",
        }
    )

    # When: the launcher reaches strict validation with a stale manifest pin.
    result = subprocess.run(
        ["/usr/bin/bash", str(launcher)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output_dir = output_root / run_id
    diagnostic = RuntimeEnvironmentDiagnostic.model_validate_json(
        (output_dir / "runtime_environment.json").read_text(encoding="utf-8")
    )
    preflight = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))

    # Then: diagnostic evidence precedes and does not override strict manifest failure.
    assert result.returncode == 1
    assert json.loads(result.stdout) == diagnostic.model_dump(mode="json")
    assert preflight == {
        "detail": "source_identity:manifest_digest_mismatch",
        "status": "FAILED",
    }
    assert diagnostic.runtime_modules_loaded == ()
