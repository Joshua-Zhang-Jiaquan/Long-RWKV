"""Adapter tests: qualification must replay evidence, never trust a declaration."""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from pathlib import Path
from typing import Final, cast

import pytest

from scale.experiments.nonlatent_iclr.tasks import external_assets as e
from scale.experiments.nonlatent_iclr.tasks import qualification_adapters as q
from scale.experiments.nonlatent_iclr.tasks.pinned_sources import (
    ALLOWED_HOSTS, ALLOWED_LICENSES, BUNDLE_NAME, MANIFEST_NAME, PINNED_SOURCES,
    SUITE_DIRECTORIES, PinnedSource, is_pinned_commit, source_for,
)

ASSET_ROOT: Final = Path("DAN/nonlatent_iclr/task4_assets")
RULER: Final = "ruler_development_suite"
LONGBENCH: Final = "longbench_development_suite"


def _status(root: Path, requirement: str) -> e.AssetStatus:
    return next(status for status in e.inventory_external_assets(root) if status.requirement == requirement)


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


def _write_suite(
    root: Path,
    source: PinnedSource,
    *,
    commit: str | None = None,
    license_id: str | None = None,
    omit_generators: bool = False,
) -> Path:
    """Build an internally self-consistent suite: bundle hash binds, per-file hashes bind."""
    suite = root / SUITE_DIRECTORIES[source.requirement]
    entries: list[dict[str, str]] = []
    for entry in source.files:
        body = f"vendored:{entry.dest}\n".encode()
        path = suite / entry.dest
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(body)
        entries.append({"dest": entry.dest, "upstream_path": entry.upstream_path, "sha256": sha256(body).hexdigest()})
    bundle: dict[str, object] = {
        "schema_version": 1,
        "suite": SUITE_DIRECTORIES[source.requirement],
        "upstream": {
            "host": source.host, "repo": source.repo,
            "commit": commit or source.commit, "license": license_id or source.license,
        },
        "files": entries,
        "generators": [] if omit_generators else sorted(source.generators),
        "generator_config": source.generator_config,
    }
    data = (json.dumps(bundle, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _ = (suite / BUNDLE_NAME).write_bytes(data)
    manifest = {
        "version": f"{SUITE_DIRECTORIES[source.requirement]}@{commit or source.commit}",
        "license": license_id or source.license,
        "sha256": sha256(data).hexdigest(),
        "asset_path": BUNDLE_NAME,
    }
    _ = (suite / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return suite


def test_adapter_registry_covers_every_declared_requirement() -> None:
    # Given: the declared external requirements and the dispatch table.
    declared = {requirement.requirement for requirement in e.REQUIREMENTS}
    # When / Then: every declared requirement has an adapter and none is unhandled.
    assert set(q.ADAPTERS) == declared


def test_every_declared_requirement_is_qualified_by_replay() -> None:
    # Given: the pinned suites acquired from upstream at the frozen commits, the accepted
    # tokenizer, the published isolation evidence and the derived repository-task corpus.
    statuses = {status.requirement: status for status in e.inventory_external_assets(ASSET_ROOT)}
    # When / Then: each of the five declared requirements qualifies on replay alone.
    assert set(statuses) == {requirement.requirement for requirement in e.REQUIREMENTS}
    for requirement, status in statuses.items():
        assert status.ready is True, f"{requirement}: {status.reason}"
        assert status.reason.startswith("QUALIFIED"), requirement
    assert "NVIDIA/RULER@c3f5e3b4f87f" in statuses[RULER].reason
    assert "THUDM/LongBench@2e00731f8d0b" in statuses[LONGBENCH].reason
    assert "replayed" in statuses["rwkv_tokenizer"].reason
    assert "process_sandbox_not_externally_managed" in statuses["isolated_repository_evaluator"].reason
    assert "held out" in statuses["rights_cleared_repository_tasks"].reason


def test_forged_bundle_with_an_arbitrary_commit_is_rejected(tmp_path: Path) -> None:
    # Given: a bundle that is internally consistent but pins a commit we never verified.
    _ = _write_suite(tmp_path, source_for(RULER), commit="b" * 40)
    # When: the adapter replays it against the frozen pin.
    status = _status(tmp_path, RULER)
    # Then: self-consistency alone cannot qualify a stranger's revision.
    assert status.ready is False
    assert "commit" in status.reason


def test_swapped_vendored_file_is_rejected(tmp_path: Path) -> None:
    # Given: a suite whose declared per-file hash still names the original bytes.
    source = source_for(RULER)
    suite = _write_suite(tmp_path, source)
    _ = (suite / source.files[0].dest).write_bytes(b"swapped after qualification\n")
    # When: the adapter recomputes the file hashes from disk.
    status = _status(tmp_path, RULER)
    # Then: the declaration is not accepted as evidence for bytes it does not describe.
    assert status.ready is False
    assert source.files[0].dest in status.reason


def test_license_outside_the_allowlist_is_rejected(tmp_path: Path) -> None:
    # Given: a self-consistent suite declaring a non-permissive license.
    _ = _write_suite(tmp_path, source_for(RULER), license_id="GPL-3.0")
    # When / Then: the allowlist is enforced on the bundle, not just the manifest.
    status = _status(tmp_path, RULER)
    assert status.ready is False
    assert "license" in status.reason


def test_missing_generators_are_rejected(tmp_path: Path) -> None:
    # Given: a self-consistent suite that pins no data generator.
    _ = _write_suite(tmp_path, source_for(RULER), omit_generators=True)
    # When / Then: pinning a suite without its generator is not qualified.
    status = _status(tmp_path, RULER)
    assert status.ready is False
    assert "generator" in status.reason


def test_manifest_hash_not_bound_to_bundle_is_rejected(tmp_path: Path) -> None:
    # Given: a suite whose manifest digest names bytes that are not the bundle on disk.
    source = source_for(RULER)
    suite = _write_suite(tmp_path, source)
    manifest = cast("dict[str, object]", json.loads((suite / MANIFEST_NAME).read_text(encoding="utf-8")))
    manifest["sha256"] = "0" * 64
    _ = (suite / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # When / Then: the bound-asset check rejects it before any adapter runs.
    status = _status(tmp_path, RULER)
    assert status.ready is False
    assert "not bound to local asset bytes" in status.reason


def test_tokenizer_adapter_is_fail_closed_without_a_replay_sidecar(tmp_path: Path) -> None:
    # Given: a tokenizer manifest that satisfies the shape checks but has no replay sidecar.
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir(parents=True)
    body = b'{"results": "placeholder"}\n'
    _ = (tokenizer / "rwkv_tokenizer_results.json").write_bytes(body)
    manifest = {
        "version": "1", "license": "local", "sha256": sha256(body).hexdigest(),
        "results_path": "rwkv_tokenizer_results.json",
        "qualification_procedure": "replay",
    }
    _ = (tokenizer / "rwkv_tokenizer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # When / Then: absent replay inputs cannot qualify the tokenizer.
    status = _status(tmp_path, "rwkv_tokenizer")
    assert status.ready is False
    assert "no trusted" not in status.reason


def test_adapter_never_imports_the_network_module() -> None:
    # Given: the replay path that must stay offline.
    module = Path(q.__file__)
    # When / Then: its real imports depend on the frozen table, never the fetching module.
    assert not [name for name in _imported_modules(module) if name.endswith("asset_acquisition")]


def test_planted_tampering_still_rejected_after_adapter_dispatch(tmp_path: Path) -> None:
    # Given: a fake pinned suite with a manifest whose digest is not bound to its asset.
    suite = tmp_path / "ruler"
    suite.mkdir(parents=True)
    _ = (suite / "suite.json").write_text("altered", encoding="utf-8")
    _ = (suite / "development_manifest.json").write_text(
        json.dumps({"version": "1", "license": "cleared", "sha256": "0" * 64, "asset_path": "suite.json"}),
        encoding="utf-8",
    )
    # When / Then: the production tampering control still fails closed.
    assert _status(tmp_path, RULER).ready is False


def _source_id(source: PinnedSource) -> str:
    return source.requirement


@pytest.mark.parametrize("source", PINNED_SOURCES, ids=_source_id)
def test_frozen_pins_are_well_formed(source: PinnedSource) -> None:
    # Given: each frozen pin.
    # When / Then: it declares an allowlisted host, license, a full commit, and generators.
    assert source.host in ALLOWED_HOSTS
    assert source.license in ALLOWED_LICENSES
    assert is_pinned_commit(source.commit)
    assert source.generators
    declared = {entry.dest for entry in source.files}
    assert set(source.generators) <= declared


def _evaluator_root(tmp_path: Path, *, scope: str | None = None) -> Path:
    """Publish genuine isolation evidence, optionally overriding the declared scope."""
    from scale.experiments.nonlatent_iclr.tasks import repository_evaluator as rev

    root = tmp_path / "repository_evaluator"
    manifest_path, evidence_path = rev.write_isolation_evidence(root)
    if scope is not None:
        manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest["scope"] = scope
        _ = manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert evidence_path.is_file()
    return tmp_path


def test_evaluator_adapter_qualifies_when_the_live_probe_matches(tmp_path: Path) -> None:
    # Given: published isolation evidence produced by a real probe of this host.
    root = _evaluator_root(tmp_path)
    # When / Then: the evaluator requirement is reachable by replay, with its scope disclosed.
    status = _status(root, "isolated_repository_evaluator")
    assert status.ready is True, status.reason
    assert "process_sandbox_not_externally_managed" in status.reason


def test_evaluator_adapter_rejects_a_claim_of_external_management(tmp_path: Path) -> None:
    # Given: a declaration claiming an externally managed VM this host does not provide.
    root = _evaluator_root(tmp_path, scope="externally_managed_vm")
    # When / Then: the false scope is refused rather than accepted at face value.
    status = _status(root, "isolated_repository_evaluator")
    assert status.ready is False
    assert "externally managed" in status.reason


def test_evaluator_adapter_rejects_evidence_that_no_longer_matches_the_host(tmp_path: Path) -> None:
    # Given: genuine evidence whose recorded capability set is then altered.
    root = _evaluator_root(tmp_path)
    assert _status(root, "isolated_repository_evaluator").ready is True
    evidence_path = root / "repository_evaluator" / "isolation_evidence.json"
    evidence = cast("dict[str, object]", json.loads(evidence_path.read_text(encoding="utf-8")))
    observations = cast("dict[str, object]", evidence["observations"])
    observations["network_namespace"] = {"observed": True, "detail": "forged as available"}
    _ = evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = root / "repository_evaluator" / "isolation_manifest.json"
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    manifest["sha256"] = sha256(evidence_path.read_bytes()).hexdigest()
    _ = manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # When: the adapter replays the probe; the digest still binds, so only replay can catch it.
    status = _status(root, "isolated_repository_evaluator")
    # Then: the mismatch with the real host is detected.
    assert status.ready is False
    assert "no longer matches" in status.reason
