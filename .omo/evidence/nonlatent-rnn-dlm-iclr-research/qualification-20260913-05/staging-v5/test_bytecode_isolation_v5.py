from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from bytecode_test_support import (
    RELEASE_ROOT,
    V4_RELEASE,
    audit_import,
    guard_path,
    output_value,
    owned_bytecode,
    parse_observation,
    qualification_dir,
    run_cache_lifecycle,
    run_guard_scan,
)


V4_REVIEW = Path(
    "/inspire/hdd/project/multimodal-diffusion-language-model/"
    "zhangjiaquan-253108540222/DiffRWKV-RELAY/.omo/evidence/"
    "nonlatent-rnn-dlm-iclr-research/qualification-20260913-05/"
    "staging-v4-review/cache-audit.json"
)


class CacheOpenSet(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    opened_release_pyc: tuple[Path, ...]


class RuntimeReproduction(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    default_first: CacheOpenSet


class CacheAuditReceipt(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    runtime_reproduction: RuntimeReproduction


def test_v4_reproduction_executes_unmanifested_release_bytecode() -> None:
    # Given: the forensic v4 release and the independent review's actual-open receipt.
    receipt = CacheAuditReceipt.model_validate_json(
        V4_REVIEW.read_text(encoding="utf-8")
    )

    # When: a fresh interpreter imports the same release without cache isolation.
    result = audit_import(V4_RELEASE, isolated=False)
    assert result.returncode == 0, result.stderr
    observation = parse_observation(result.stdout)

    # Then: existing release bytecode is opened and unmarshalled despite dont-write mode.
    assert observation.dont_write_bytecode == "1"
    assert observation.opened_release_pyc
    assert observation.loaded_release_pyc
    reviewed = receipt.runtime_reproduction.default_first.opened_release_pyc
    assert set(observation.opened_release_pyc).issubset(reviewed)
    assert set(observation.loaded_release_pyc).issubset(reviewed)


def test_source_clean_release_remains_bytecode_free_after_launcher_help() -> None:
    # Given: the candidate's complete owned bytecode inventory.
    before = owned_bytecode(RELEASE_ROOT)
    launcher = qualification_dir(RELEASE_ROOT) / "run_qualification.sh"

    # When: the public shell-only help path exits normally.
    result = subprocess.run(
        (str(launcher), "--help"),
        env={"LC_ALL": "C", "PATH": "/usr/local/bin:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )
    after = owned_bytecode(RELEASE_ROOT)

    # Then: the release starts clean and help neither creates nor retains bytecode.
    assert result.returncode == 0, result.stderr
    assert before == ()
    assert after == ()


def test_isolated_child_uses_launcher_environment_without_bytecode_io() -> None:
    # Given: the candidate helper and source-clean release.
    before = owned_bytecode(RELEASE_ROOT)

    # When: a real child imports the qualification CLI through the cache guard.
    result = audit_import(RELEASE_ROOT, isolated=True)
    assert result.returncode == 0, result.stderr
    observation = parse_observation(result.stdout)
    cache_prefix = Path(output_value(result.stdout, "CACHE_PREFIX"))

    # Then: exact owned paths are active, no release bytecode is loaded or written,
    # and normal exit removes only the fresh temporary prefix.
    assert observation.cwd == RELEASE_ROOT
    assert observation.pythonpath == (
        f"{RELEASE_ROOT}:{RELEASE_ROOT / 'model_source/scale'}"
    )
    assert observation.safe_path is True
    assert observation.dont_write_bytecode == "1"
    assert observation.cache_prefix == cache_prefix
    assert str(cache_prefix).startswith("/tmp/nonlatent-qualification-pycache.")
    assert observation.opened_release_pyc == ()
    assert observation.loaded_release_pyc == ()
    assert output_value(result.stdout, "CACHE_ENTRY") == ""
    assert not cache_prefix.exists()
    assert before == ()
    assert owned_bytecode(RELEASE_ROOT) == ()


@pytest.mark.parametrize(
    "relative_path",
    (Path("__pycache__"), Path("stale.pyc"), Path("nested/stale.pyo")),
)
def test_preimport_guard_rejects_every_bytecode_artifact(
    tmp_path: Path, relative_path: Path
) -> None:
    # Given: clean owned roots with one injected cache directory or bytecode file.
    payload_root = tmp_path / "payload"
    model_root = tmp_path / "model"
    payload_root.mkdir()
    model_root.mkdir()
    artifact = payload_root / relative_path
    if artifact.name == "__pycache__":
        artifact.mkdir()
    else:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"stale-bytecode")

    # When: the production pre-import inventory guard scans only those roots.
    result = run_guard_scan(RELEASE_ROOT, payload_root, model_root)

    # Then: startup fails closed and identifies the injected owned artifact.
    assert result.returncode != 0
    assert "BYTECODE_PRECHECK_REJECTED" in result.stderr
    assert str(artifact) in result.stderr


@pytest.mark.parametrize(
    ("mode", "expected_returncode"), (("failure", 7), ("timeout", 124))
)
def test_cache_prefix_is_removed_after_failure_or_timeout(
    mode: Literal["failure", "timeout"], expected_returncode: int
) -> None:
    # Given: the production cache lifecycle running in a failing or timed-out shell.
    # When: the shell terminates through the selected non-success path.
    result = run_cache_lifecycle(RELEASE_ROOT, mode)
    assert result.returncode == expected_returncode, result.stderr
    cache_prefix = Path(output_value(result.stdout, "CACHE_PREFIX"))

    # Then: the requested status is visible and the helper-owned prefix is gone.
    assert str(cache_prefix).startswith("/tmp/nonlatent-qualification-pycache.")
    assert not cache_prefix.exists()


def test_launcher_installs_guard_before_every_python_child() -> None:
    # Given: the candidate launcher and its single shell guard helper.
    launcher = qualification_dir(RELEASE_ROOT) / "run_qualification.sh"
    helper = guard_path(RELEASE_ROOT)
    source = launcher.read_text(encoding="utf-8")
    assert helper.is_file()
    helper_source = helper.read_text(encoding="utf-8")

    # When: startup ordering and cleanup ownership are inspected.
    inner = source.index('if [[ "${QUALIFICATION_WORKFLOW_INNER:-0}" != 1 ]]')
    guard_import = source.index('source "${PAYLOAD_DIR}/bytecode_guard.sh"')
    cache_setup = source.index("qualification_python_cache_setup")
    cache_scan = source.index("qualification_assert_source_clean")
    first_python = source.index('"${PAYLOAD_DIR}/runtime_environment.py"')
    torchrun = source.index("/usr/local/bin/torchrun")

    # Then: isolation and inventory precede Python/torchrun and every exit cleans it.
    assert inner < guard_import < cache_setup < cache_scan < first_python < torchrun
    assert 'cd -- "${REPO_ROOT}"' in source
    assert "export PYTHONSAFEPATH=1" in source
    assert 'export PYTHONPATH="${REPO_ROOT}:${STAGED_ROOT}"' in source
    assert "PYTHONDONTWRITEBYTECODE=1" in helper_source
    assert "PYTHONPYCACHEPREFIX" in helper_source
    assert source.count("qualification_python_cache_cleanup") >= 2
