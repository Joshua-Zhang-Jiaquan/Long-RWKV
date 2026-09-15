"""Acquisition tests: pinned provenance, proxy scrubbing, and fail-closed writes."""

from __future__ import annotations

import ast
import json
import os
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from typing import Final, cast

import pytest

from scale.experiments.nonlatent_iclr.tasks import asset_acquisition as a
from scale.experiments.nonlatent_iclr.tasks import external_assets as e

ASSET_ROOT: Final = Path("DAN/nonlatent_iclr/task4_assets")
PROXY_NAMES: Final = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy")
REQUIREMENT: Final = "ruler_development_suite"
VERIFICATION_MODULES: Final = (
    Path("scale/experiments/nonlatent_iclr/tasks/external_assets.py"),
    Path("scale/experiments/nonlatent_iclr/tasks/qualification_adapters.py"),
    Path("scale/experiments/nonlatent_iclr/registry_task.py"),
    Path("scale/experiments/nonlatent_iclr/task_registry.py"),
)
BODIES: Final = {
    "source/LICENSE": b"license bytes\n",
    "source/generator.py": b"def f():\n    return 1\n",
}


def _fake_source(requirement: str = REQUIREMENT) -> tuple[a.PinnedSource, dict[str, bytes]]:
    """Build a pinned source plus the bytes a transport should serve; touches no disk."""
    files = tuple(
        a.PinnedFile(dest, f"upstream/{dest}", sha256(body).hexdigest())
        for dest, body in sorted(BODIES.items())
    )
    source = a.PinnedSource(
        requirement=requirement, host="raw.githubusercontent.com", repo="owner/repo",
        commit="a" * 40, license="MIT", files=files,
        generators=("source/generator.py",), generator_config="source/generator.py",
    )
    return source, dict(BODIES)


def _transport(bodies: dict[str, bytes]):
    """Serve declared bytes for whichever pinned source is being acquired."""

    def fetch(_source: a.PinnedSource, entry: a.PinnedFile) -> bytes:
        return bodies[entry.dest]

    return fetch


def test_scrubbed_environment_when_every_proxy_spelling_is_present() -> None:
    # Given: a caller environment carrying every proxy spelling the harness uses.
    base = {name: "http://127.0.0.1:7890" for name in PROXY_NAMES}
    base.update({"no_proxy": "localhost,127.0.0.1", "PATH": "/usr/bin", "HOME": "/root"})
    # When: the acquisition environment is prepared.
    env = a.scrubbed_network_env(base)
    # Then: no proxy variable survives and direct egress is forced.
    assert not [name for name in PROXY_NAMES if name in env]
    assert env["no_proxy"] == "localhost,127.0.0.1,*"
    assert (env["PATH"], env["HOME"]) == ("/usr/bin", "/root")


def test_scrubbed_environment_when_absent_still_forces_direct() -> None:
    # Given: a bare environment without any proxy or no_proxy entry.
    env = a.scrubbed_network_env({"PATH": "/usr/bin"})
    # When / Then: the result still bypasses any ambient proxy.
    assert env["no_proxy"] == "*"


def test_pinned_table_when_compared_with_declared_requirements() -> None:
    # Given: the frozen pin table and the declared external requirements.
    declared = {requirement.requirement for requirement in e.REQUIREMENTS}
    # When / Then: exactly the two network-pinned suites are frozen and pass the allowlists.
    assert {source.requirement for source in a.PINNED_SOURCES} == {
        "ruler_development_suite", "longbench_development_suite",
    }
    assert {source.requirement for source in a.PINNED_SOURCES} <= declared
    for source in a.PINNED_SOURCES:
        assert source.license in a.ALLOWED_LICENSES
        assert source.host in a.ALLOWED_HOSTS
        assert a.is_pinned_commit(source.commit), source.commit
        assert source.requirement in a.SUITE_DIRECTORIES
        assert source.generators
        dests = [entry.dest for entry in source.files]
        assert len(dests) == len(set(dests))
        assert all(dest.startswith("source/") for dest in dests)


def test_pinned_table_when_compared_with_vendored_bytes_on_disk() -> None:
    # Given: the accepted vendored files already present under the real asset root.
    # When / Then: each frozen hash still describes the bytes on disk.
    for source in a.PINNED_SOURCES:
        suite = ASSET_ROOT / a.SUITE_DIRECTORIES[source.requirement]
        for entry in source.files:
            actual = sha256((suite / entry.dest).read_bytes()).hexdigest()
            assert actual == entry.sha256, entry.dest


def test_raw_url_when_source_is_pinned() -> None:
    # Given: a pinned source entry.
    entry = a.PinnedFile("source/LICENSE", "LICENSE", "c" * 64)
    source = a.PinnedSource("ruler_development_suite", "raw.githubusercontent.com", "NVIDIA/RULER", "b" * 40, "Apache-2.0", (entry,), ("source/LICENSE",))
    # When: its fetch URL is derived.
    # Then: the URL is commit-pinned rather than branch-floating.
    assert a.raw_url(source, entry) == f"https://raw.githubusercontent.com/NVIDIA/RULER/{'b' * 40}/LICENSE"


def test_raw_url_rejects_when_host_is_not_allowlisted() -> None:
    # Given: a source pointing at an undeclared host.
    entry = a.PinnedFile("source/LICENSE", "LICENSE", "c" * 64)
    source = a.PinnedSource("ruler_development_suite", "evil.example", "NVIDIA/RULER", "b" * 40, "Apache-2.0", (entry,), ("source/LICENSE",))
    # When / Then: URL derivation fails closed before any request.
    with pytest.raises(a.AcquisitionError, match="host"):
        _ = a.raw_url(source, entry)


def test_raw_url_rejects_when_license_is_not_allowlisted() -> None:
    # Given: a source under a non-permissive license.
    entry = a.PinnedFile("source/LICENSE", "LICENSE", "c" * 64)
    source = a.PinnedSource("ruler_development_suite", "raw.githubusercontent.com", "NVIDIA/RULER", "b" * 40, "GPL-3.0", (entry,), ("source/LICENSE",))
    # When / Then: acquisition refuses the source entirely.
    with pytest.raises(a.AcquisitionError, match="license"):
        _ = a.raw_url(source, entry)


def test_raw_url_rejects_when_commit_is_not_a_full_sha() -> None:
    # Given: a branch name standing in for a commit.
    entry = a.PinnedFile("source/LICENSE", "LICENSE", "c" * 64)
    source = a.PinnedSource("ruler_development_suite", "raw.githubusercontent.com", "NVIDIA/RULER", "main", "Apache-2.0", (entry,), ("source/LICENSE",))
    # When / Then: a floating reference cannot be acquired.
    with pytest.raises(a.AcquisitionError, match="commit"):
        _ = a.raw_url(source, entry)


def test_acquire_when_fetch_succeeds_binds_manifest_to_bundle(tmp_path: Path) -> None:
    # Given: a pinned source whose bytes are served from a fake transport.
    source, bodies = _fake_source()
    # When: acquisition runs offline against that transport.
    receipt = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=_transport(bodies))
    # Then: every pinned file is published and the manifest is a real root of trust.
    suite = tmp_path / a.SUITE_DIRECTORIES[REQUIREMENT]
    manifest = cast("dict[str, object]", json.loads((suite / "development_manifest.json").read_text(encoding="utf-8")))
    assert receipt.files_written == len(source.files)
    assert receipt.files_archived == 0
    assert manifest["sha256"] == sha256((suite / str(manifest["asset_path"])).read_bytes()).hexdigest()
    assert manifest["version"] == f"{a.SUITE_DIRECTORIES[REQUIREMENT]}@{source.commit}"
    assert manifest["license"] == source.license
    assert receipt.bundle_sha256 == manifest["sha256"]
    assert receipt.network_env_scrubbed is True
    # Then: the declared asset passes every shape check, the missing adapter aside.
    status = next(s for s in e.inventory_external_assets(tmp_path) if s.requirement == REQUIREMENT)
    assert "missing declared local asset" not in status.reason
    assert "not bound to local asset bytes" not in status.reason


def test_acquire_when_repeated_leaves_identical_bytes_and_no_archive(tmp_path: Path) -> None:
    # Given: an already acquired suite.
    source, bodies = _fake_source()
    transport = _transport(bodies)
    before = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=transport)
    # When: the same pinned bytes are acquired again.
    after = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=transport)
    # Then: nothing is rewritten and no archive is manufactured.
    assert before.files_written == len(source.files)
    assert after.files_written == 0
    assert after.files_unchanged == len(source.files)
    assert after.files_archived == 0
    assert after.bundle_sha256 == before.bundle_sha256
    assert not list((tmp_path / a.SUITE_DIRECTORIES[REQUIREMENT]).rglob("*.archive-*"))


def test_acquire_when_vendored_bytes_change_archives_instead_of_overwriting(tmp_path: Path) -> None:
    # Given: an acquired suite whose vendored generator is then replaced out of band.
    source, bodies = _fake_source()
    suite = tmp_path / a.SUITE_DIRECTORIES[REQUIREMENT]
    transport = _transport(bodies)
    _ = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=transport)
    target = suite / "source/generator.py"
    _ = target.write_bytes(b"def f():\n    return 2\n")
    # When: acquisition restores the pinned bytes.
    receipt = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=transport)
    # Then: the altered bytes are archived byte-for-byte and the pin is restored.
    archives = list(suite.rglob("*.archive-*"))
    assert receipt.files_archived == 1
    assert len(archives) == 1
    assert archives[0].read_bytes() == b"def f():\n    return 2\n"
    assert target.read_bytes() == b"def f():\n    return 1\n"


def test_acquire_when_fetched_bytes_do_not_match_the_pin_writes_nothing(tmp_path: Path) -> None:
    # Given: a transport serving bytes that differ from the frozen hash.
    source, bodies = _fake_source()
    tampered = dict(bodies)
    tampered["source/LICENSE"] = b"tampered upstream bytes\n"
    # When: acquisition is attempted.
    # Then: it fails closed and no partial suite is published.
    with pytest.raises(a.AcquisitionError, match="pin"):
        _ = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=_transport(tampered))
    suite = tmp_path / a.SUITE_DIRECTORIES[REQUIREMENT]
    assert not (suite / "development_manifest.json").exists()
    assert not (suite / "source_bundle.json").exists()
    assert not (suite / "source/LICENSE").exists()


def test_acquire_when_dry_run_writes_nothing(tmp_path: Path) -> None:
    # Given: a pinned source and an empty destination root.
    source, bodies = _fake_source()
    # When: acquisition is planned only.
    receipt = a.acquire(REQUIREMENT, tmp_path, dry_run=True, source=source, fetch=_transport(bodies))
    # Then: the plan is reported without touching the destination.
    assert receipt.dry_run is True
    assert receipt.files_written == 0
    assert receipt.bundle_sha256 == sha256(a.bundle_bytes(a.build_bundle(source))).hexdigest()
    assert not (tmp_path / a.SUITE_DIRECTORIES[REQUIREMENT] / "development_manifest.json").exists()


def test_acquire_when_manifest_conflicts_raises_without_replacing(tmp_path: Path) -> None:
    # Given: an acquired suite whose manifest is then replaced out of band.
    source, bodies = _fake_source()
    suite = tmp_path / a.SUITE_DIRECTORIES[REQUIREMENT]
    transport = _transport(bodies)
    _ = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=transport)
    manifest = suite / "development_manifest.json"
    _ = manifest.write_text(json.dumps({"version": "forged"}), encoding="utf-8")
    # When: acquisition runs again.
    # Then: the conflicting declaration is reported, not silently overwritten.
    with pytest.raises(a.AcquisitionConflict):
        _ = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=transport)
    preserved = cast("dict[str, object]", json.loads(manifest.read_text(encoding="utf-8")))
    assert preserved == {"version": "forged"}


def test_verify_acquired_when_vendored_bytes_are_swapped(tmp_path: Path) -> None:
    # Given: an acquired suite whose vendored file no longer matches the frozen pin.
    source, bodies = _fake_source()
    suite = tmp_path / a.SUITE_DIRECTORIES[REQUIREMENT]
    _ = a.acquire(REQUIREMENT, tmp_path, source=source, fetch=_transport(bodies))
    assert a.verify_acquired(REQUIREMENT, tmp_path, source=source) == ()
    _ = (suite / "source/LICENSE").write_bytes(b"swapped\n")
    # When: the offline verifier inspects the suite.
    problems = a.verify_acquired(REQUIREMENT, tmp_path, source=source)
    # Then: the swap is reported here. Binding vendored files at inventory level is the
    # trusted adapter's job, so the inventory is only required to stay unready here.
    assert problems
    status = next(s for s in e.inventory_external_assets(tmp_path) if s.requirement == REQUIREMENT)
    assert status.ready is False


def _imported_modules(path: Path) -> set[str]:
    """Collect every imported module name, including imports made inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_verification_modules_never_import_acquisition() -> None:
    # Given: the modules on the Task4 verification path.
    # When / Then: none of their real imports may reach the network-capable module.
    for module in VERIFICATION_MODULES:
        reached = [name for name in _imported_modules(module) if name.endswith("asset_acquisition")]
        assert not reached, f"{module} imports {reached}"


def test_verify_registry_when_network_is_unavailable() -> None:
    # Given: an interpreter with every socket entry point disabled.
    program = (
        "import socket\n"
        "def _blocked(*args, **kwargs):\n"
        "    raise RuntimeError('network access attempted')\n"
        "socket.socket = _blocked\n"
        "socket.create_connection = _blocked\n"
        "from scale.experiments.nonlatent_iclr.task_registry import build_registry, verify_registry\n"
        "print(verify_registry(build_registry()).exit_code)\n"
    )
    # When: the Task4 verification path runs there.
    process = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False, timeout=300,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    # Then: verification completes offline with no hidden fetch. The documented codes are
    # 0 (every declared asset qualified by replay) and 2 (something is still unqualified);
    # this test asserts only that the offline path completes, not which stage it reached.
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() in {"0", "2"}
