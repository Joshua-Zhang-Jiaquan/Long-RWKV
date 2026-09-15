from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os
import subprocess
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict


PYTHON: Final = Path("/usr/bin/python")
BASH: Final = Path("/bin/bash")
TIMEOUT: Final = Path("/usr/bin/timeout")
V4_RELEASE: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/"
    "payloads/qualification-20260913-05-global-ffaa464d-origin-v4"
)
V5_RELEASE: Final = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/"
    "payloads/qualification-20260913-05-global-ffaa464d-bytecode-v5"
)
RELEASE_ROOT: Final = Path(
    os.environ.get("QUALIFICATION_RELEASE_UNDER_TEST", str(V5_RELEASE))
)

AUDIT_SCRIPT: Final = r"""
import json
import os
from pathlib import Path
import sys

release = Path(os.environ["RELEASE_ROOT"]).resolve()
opened = []
loaded = []
pending = None

def observe(event, args):
    global pending
    if event == "open" and args:
        raw_path = args[0]
        if isinstance(raw_path, (str, bytes)):
            path = Path(os.fsdecode(raw_path))
            try:
                resolved = path.resolve()
            except OSError:
                return
            if (
                resolved.suffix in {".pyc", ".pyo"}
                and resolved.is_relative_to(release)
                and resolved.is_file()
            ):
                pending = str(resolved)
                opened.append(pending)
    elif event == "marshal.loads" and pending is not None:
        loaded.append(pending)
        pending = None

sys.addaudithook(observe)
import scale.experiments.nonlatent_iclr.qualification.cli

print(json.dumps({
    "cache_prefix": os.environ.get("PYTHONPYCACHEPREFIX"),
    "cwd": os.getcwd(),
    "dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
    "loaded_release_pyc": sorted(set(loaded)),
    "opened_release_pyc": sorted(set(opened)),
    "pythonpath": os.environ.get("PYTHONPATH"),
    "safe_path": sys.flags.safe_path,
}, sort_keys=True))
"""


class ImportObservation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    cache_prefix: Path | None
    cwd: Path
    dont_write_bytecode: str | None
    loaded_release_pyc: tuple[Path, ...]
    opened_release_pyc: tuple[Path, ...]
    pythonpath: str | None
    safe_path: bool


def qualification_dir(release: Path) -> Path:
    return release / "scale/experiments/nonlatent_iclr/qualification"


def guard_path(release: Path) -> Path:
    return qualification_dir(release) / "bytecode_guard.sh"


def fixed_environment(release: Path) -> Mapping[str, str]:
    return {
        "LC_ALL": "C",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            (str(release), str(release / "model_source/scale"))
        ),
        "PYTHONSAFEPATH": "1",
        "RELEASE_ROOT": str(release),
    }


def owned_bytecode(release: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in release.rglob("*")
                if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
            ),
            key=str,
        )
    )


def audit_import(release: Path, *, isolated: bool) -> subprocess.CompletedProcess[str]:
    environment = fixed_environment(release)
    if not isolated:
        return subprocess.run(
            (str(PYTHON), "-P", "-c", AUDIT_SCRIPT),
            cwd=release,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    script = r"""
set -Eeuo pipefail
source "$1"
cd -- "$2"
export PYTHONSAFEPATH=1
export PYTHONPATH="$2:$3"
export RELEASE_ROOT="$2"
qualification_python_cache_setup
trap 'qualification_python_cache_cleanup' EXIT
trap 'exit 143' INT TERM
qualification_assert_source_clean "$2/scale" "$3"
"$4" -P -c "$5"
cache_entry="$(/usr/bin/find "${PYTHONPYCACHEPREFIX}" -mindepth 1 -print -quit)"
printf 'CACHE_PREFIX=%s\n' "${PYTHONPYCACHEPREFIX}"
printf 'CACHE_ENTRY=%s\n' "${cache_entry}"
"""
    return subprocess.run(
        (
            str(BASH),
            "-c",
            script,
            "bytecode-audit",
            str(guard_path(release)),
            str(release),
            str(release / "model_source/scale"),
            str(PYTHON),
            AUDIT_SCRIPT,
        ),
        env={"LC_ALL": "C", "PATH": "/usr/local/bin:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )


def parse_observation(stdout: str) -> ImportObservation:
    line = next(item for item in stdout.splitlines() if item.startswith("{"))
    return ImportObservation.model_validate_json(line)


def output_value(stdout: str, name: str) -> str:
    prefix = f"{name}="
    return next(line.removeprefix(prefix) for line in stdout.splitlines() if line.startswith(prefix))


def run_guard_scan(
    release: Path, payload_root: Path, model_root: Path
) -> subprocess.CompletedProcess[str]:
    script = 'set -Eeuo pipefail; source "$1"; qualification_assert_source_clean "$2" "$3"'
    return subprocess.run(
        (
            str(BASH),
            "-c",
            script,
            "bytecode-scan",
            str(guard_path(release)),
            str(payload_root),
            str(model_root),
        ),
        env={"LC_ALL": "C", "PATH": "/usr/local/bin:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )


def run_cache_lifecycle(
    release: Path, mode: Literal["normal", "failure", "timeout"]
) -> subprocess.CompletedProcess[str]:
    script = r"""
set -Eeuo pipefail
source "$1"
qualification_python_cache_setup
trap 'qualification_python_cache_cleanup' EXIT
trap 'exit 143' INT TERM
printf 'CACHE_PREFIX=%s\n' "${PYTHONPYCACHEPREFIX}"
case "$2" in
    normal) exit 0 ;;
    failure) exit 7 ;;
    timeout) while :; do :; done ;;
esac
"""
    command = (str(BASH), "-c", script, "cache-lifecycle", str(guard_path(release)), mode)
    if mode == "timeout":
        command = (
            str(TIMEOUT),
            "--signal=TERM",
            "--kill-after=1s",
            "0.2s",
            *command,
        )
    return subprocess.run(
        command,
        env={"LC_ALL": "C", "PATH": "/usr/local/bin:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        check=False,
    )
