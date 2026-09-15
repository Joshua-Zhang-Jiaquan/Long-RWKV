from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Final

from origin_test_support import (
    PYTHON,
    python_environment,
    run_python,
    write_fake_manifest,
    write_fake_models,
)


V4_RELEASE: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/"
    "payloads/qualification-20260913-05-global-ffaa464d-origin-v4"
)
RELEASE_ROOT: Final = Path(
    os.environ.get("QUALIFICATION_RELEASE_UNDER_TEST", str(V4_RELEASE))
)
OWNED_MODEL_ROOT: Final = RELEASE_ROOT / "model_source/scale"
QUALIFICATION_DIR: Final = (
    RELEASE_ROOT / "scale/experiments/nonlatent_iclr/qualification"
)
LAUNCHER: Final = QUALIFICATION_DIR / "run_qualification.sh"
ORIGINAL_CWD: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "qz_stage_traj4096_v7/scale"
)
EXPECTED_CLI_HELP: Final = """usage: nonlatent-qualification [-h] {validate,aggregate}

positional arguments:
  {validate,aggregate}

options:
  -h, --help            show this help message and exit
"""
CONSTRUCT_SETUP: Final = """
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from scale.experiments.nonlatent_iclr.qualification.contracts import QualificationRuntimeError
from scale.experiments.nonlatent_iclr.qualification.model_checks import construct_model

owned = Path(os.environ["OWNED_ROOT"])
foreign = Path(os.environ["FOREIGN_ROOT"])
manifest = Path(os.environ["FAKE_MANIFEST"])
sentinel = Path(os.environ["CONSTRUCTOR_SENTINEL"])
config = SimpleNamespace(
    staged_root=owned,
    hf_model_root=Path("/unused-hf-root"),
    manifest_path=manifest,
)
torch_module = SimpleNamespace(bfloat16="bfloat16")
"""


def _fake_environment(
    tmp_path: Path,
    *,
    constructor_sentinel: Path | None = None,
    include_helper: bool = True,
) -> tuple[dict[str, str], tuple[Path, Path, Path]]:
    owned_files = write_fake_models(tmp_path / "owned", "owned")
    write_fake_models(tmp_path / "foreign", "foreign", constructor_sentinel)
    staged_sources = owned_files if include_helper else (owned_files[0], owned_files[2])
    manifest = write_fake_manifest(tmp_path / "manifest.json", staged_sources)
    environment = dict(python_environment(RELEASE_ROOT, safe_path=True))
    environment.update(
        {
            "OWNED_ROOT": str(tmp_path / "owned"),
            "FOREIGN_ROOT": str(tmp_path / "foreign"),
            "FAKE_MANIFEST": str(manifest),
            "CONSTRUCTOR_SENTINEL": str(
                constructor_sentinel or tmp_path / "constructor-unused"
            ),
        }
    )
    return environment, owned_files


def test_owned_root_is_promoted_when_already_later_in_sys_path(
    tmp_path: Path,
) -> None:
    # Given: foreign source precedes an owned staged root already present in sys.path.
    environment, _ = _fake_environment(tmp_path)
    script = CONSTRUCT_SETUP + """
sys.path.insert(0, str(owned))
sys.path.insert(0, str(foreign))

model = construct_model(config, torch_module, "cpu")

print(model.origin)
print(sys.path[0])
"""

    # When: the production construction seam selects its model module.
    result = run_python(script, cwd=tmp_path, environment=environment)

    # Then: the owned package wins even though its path was already present later.
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["owned", str(tmp_path / "owned")]


def test_foreign_preloaded_models_package_is_rejected_before_construction(
    tmp_path: Path,
) -> None:
    # Given: a foreign models package and entrypoint are already cached.
    sentinel = tmp_path / "constructed"
    environment, _ = _fake_environment(tmp_path, constructor_sentinel=sentinel)
    script = CONSTRUCT_SETUP + """
sys.path.insert(0, str(owned))
sys.path.insert(0, str(foreign))
import models.birwkv7_diffusion

try:
    construct_model(config, torch_module, "cpu")
except QualificationRuntimeError as error:
    assert str(error) == "foreign_preloaded_model_module:models"
    assert not sentinel.exists()
else:
    raise AssertionError("foreign preloaded models package was reused")
"""

    # When: construction reaches the ownership boundary.
    result = run_python(script, cwd=tmp_path, environment=environment)

    # Then: it fails closed before invoking the foreign constructor.
    assert result.returncode == 0, result.stderr


def test_foreign_preloaded_models_submodule_is_rejected_before_construction(
    tmp_path: Path,
) -> None:
    # Given: the package is owned but its canonical entrypoint cache is foreign.
    sentinel = tmp_path / "constructed"
    environment, _ = _fake_environment(tmp_path, constructor_sentinel=sentinel)
    script = CONSTRUCT_SETUP + """
import importlib.util

sys.path.insert(0, str(owned))
import models
import models.helper

spec = importlib.util.spec_from_file_location(
    "models.birwkv7_diffusion", foreign / "models/birwkv7_diffusion.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules["models.birwkv7_diffusion"] = module
spec.loader.exec_module(module)

try:
    construct_model(config, torch_module, "cpu")
except QualificationRuntimeError as error:
    assert str(error) == "foreign_preloaded_model_module:models.birwkv7_diffusion"
    assert not sentinel.exists()
else:
    raise AssertionError("foreign preloaded models submodule was reused")
"""

    # When: construction reaches the ownership boundary.
    result = run_python(script, cwd=tmp_path, environment=environment)

    # Then: it rejects the foreign canonical submodule without deleting or reusing it.
    assert result.returncode == 0, result.stderr


def test_owned_preloaded_model_origins_are_accepted(tmp_path: Path) -> None:
    # Given: the exact owned package and entrypoint are already cached.
    environment, _ = _fake_environment(tmp_path)
    script = CONSTRUCT_SETUP + """
sys.path.insert(0, str(owned))
import models.birwkv7_diffusion

model = construct_model(config, torch_module, "cpu")

print(model.origin)
print(sys.path[0])
"""

    # When: construction validates the preloaded modules and proceeds.
    result = run_python(script, cwd=tmp_path, environment=environment)

    # Then: owned, manifest-covered origins remain legitimate.
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["owned", str(tmp_path / "owned")]


def test_import_rejects_loaded_model_helper_missing_from_manifest(
    tmp_path: Path,
) -> None:
    # Given: an owned entrypoint imports an owned helper omitted from the manifest.
    environment, _ = _fake_environment(tmp_path, include_helper=False)
    script = CONSTRUCT_SETUP + """
sys.path.insert(0, str(owned))

try:
    construct_model(config, torch_module, "cpu")
except QualificationRuntimeError as error:
    assert str(error) == "unmanifested_loaded_model_module:models.helper"
else:
    raise AssertionError("unmanifested loaded model helper was accepted")
"""

    # When: the entrypoint import completes at the construction boundary.
    result = run_python(script, cwd=tmp_path, environment=environment)

    # Then: post-import closure validation rejects the missing manifest binding.
    assert result.returncode == 0, result.stderr


def test_wrong_original_cwd_resolves_owned_models_with_launcher_environment() -> None:
    # Given: the reviewer's mutable original CWD and the launcher's actual safe-path choice.
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    safe_path = "export PYTHONSAFEPATH=1" in launcher_source
    environment = python_environment(RELEASE_ROOT, safe_path=safe_path)
    script = """
import importlib.machinery
import importlib.util

package = importlib.util.find_spec("models")
assert package is not None and package.origin is not None
assert package.submodule_search_locations is not None
entrypoint = importlib.machinery.PathFinder.find_spec(
    "models.birwkv7_diffusion", package.submodule_search_locations
)
assert entrypoint is not None and entrypoint.origin is not None
print(package.origin)
print(entrypoint.origin)
"""

    # When: Python starts exactly from the hostile CWD without importing model code.
    result = run_python(script, cwd=ORIGINAL_CWD, environment=environment)

    # Then: both resolutions point into the immutable owned package.
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(OWNED_MODEL_ROOT / "models/__init__.py"),
        str(OWNED_MODEL_ROOT / "models/birwkv7_diffusion.py"),
    ]


def test_launcher_controls_cwd_and_safe_path_before_all_python_processes() -> None:
    # Given: the global launcher inherited by validate, torchrun, and aggregate.
    launcher_source = LAUNCHER.read_text(encoding="utf-8")

    # When: its process-boundary controls are inspected.
    controlled_cwd = launcher_source.index('cd -- "${REPO_ROOT}"')
    safe_path = launcher_source.index("export PYTHONSAFEPATH=1")
    fixed_pythonpath = launcher_source.index(
        'export PYTHONPATH="${REPO_ROOT}:${STAGED_ROOT}"'
    )
    torchrun = launcher_source.index("/usr/local/bin/torchrun")

    # Then: owned paths are authoritative before direct Python or torchrun children.
    assert controlled_cwd < safe_path < fixed_pythonpath < torchrun
    assert "${PYTHONPATH:-}" not in launcher_source


def test_global_cli_help_is_exact_from_untrusted_cwd() -> None:
    # Given: an isolated safe-path environment launched from the mutable original CWD.
    environment = python_environment(RELEASE_ROOT, safe_path=True)

    # When: the global-only CPU CLI help is requested without importing model code.
    result = subprocess.run(
        (
            str(PYTHON),
            "-P",
            "-m",
            "scale.experiments.nonlatent_iclr.qualification.cli",
            "--help",
        ),
        cwd=ORIGINAL_CWD,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    # Then: the exact stable command surface is available from the hostile CWD.
    assert result.returncode == 0, result.stderr
    assert result.stdout == EXPECTED_CLI_HELP
