from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Final


PYTHON: Final = Path("/usr/bin/python")
EXPECTED_IMAGE: Final = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
REQUIRED_NONSTAGED_ROLES: Final = (
    "payload",
    "package_source",
    "hf_config",
    "hf_index",
    "hf_shard",
    "checkpoint_meta",
    "checkpoint_model",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fake_models(
    root: Path, marker: str, constructor_sentinel: Path | None = None
) -> tuple[Path, Path, Path]:
    package = root / "models"
    package.mkdir(parents=True)
    package_init = package / "__init__.py"
    helper = package / "helper.py"
    entrypoint = package / "birwkv7_diffusion.py"
    package_init.write_text("", encoding="utf-8")
    helper.write_text(f"ORIGIN = {marker!r}\n", encoding="utf-8")
    sentinel_statement = ""
    if constructor_sentinel is not None:
        sentinel_statement = (
            f"        Path({str(constructor_sentinel)!r}).write_text"
            "('constructed', encoding='utf-8')\n"
        )
    entrypoint.write_text(
        "from pathlib import Path\n"
        "from . import helper\n\n"
        "class BiRWKV7ForMaskedDiffusion:\n"
        "    def __init__(self):\n"
        "        self.origin = helper.ORIGIN\n\n"
        "    @classmethod\n"
        "    def from_hf_pretrained(cls, _root, **_kwargs):\n"
        f"{sentinel_statement}"
        "        return cls()\n\n"
        "    def to(self, _device):\n"
        "        return self\n\n"
        "    def eval(self):\n"
        "        return self\n",
        encoding="utf-8",
    )
    return package_init, helper, entrypoint


def write_fake_manifest(path: Path, staged_sources: Sequence[Path]) -> Path:
    placeholders = path.parent / "manifest-placeholders"
    placeholders.mkdir()
    files: list[dict[str, str | int]] = []
    for role in REQUIRED_NONSTAGED_ROLES:
        placeholder = placeholders / f"{role}.txt"
        placeholder.write_text(role, encoding="utf-8")
        files.append(
            {
                "path": str(placeholder),
                "role": role,
                "sha256": sha256_file(placeholder),
                "size_bytes": placeholder.stat().st_size,
            }
        )
    for source in staged_sources:
        files.append(
            {
                "path": str(source),
                "role": "staged_source",
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "payload_id": "qualification-20260912-01",
                "expected_image": EXPECTED_IMAGE,
                "files": files,
                "packages": [{"distribution": "fake", "version": "1"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def python_environment(release_root: Path, *, safe_path: bool) -> Mapping[str, str]:
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            (str(release_root), str(release_root / "model_source/scale"))
        ),
    }
    if safe_path:
        environment["PYTHONSAFEPATH"] = "1"
    return environment


def run_python(
    script: str, *, cwd: Path, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (str(PYTHON), "-c", script),
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
