"""Repository-evaluator tests: isolation claims must match what the host really does."""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from pathlib import Path
from typing import Final, cast

import pytest

from scale.experiments.nonlatent_iclr.tasks import repository_evaluator as r

SQUARE: Final = "def square(n):\n    return n * n\n"
SQUARE_CHECK: Final = "assert square(3) == 9\nassert square(0) == 0\n"
WEAK_CHECK: Final = "assert True\n"


def _observations(evidence: dict[str, object]) -> dict[str, object]:
    block = evidence.get("observations")
    assert isinstance(block, dict)
    return cast("dict[str, object]", block)


def _entry(block: dict[str, object], name: str) -> dict[str, object]:
    value = block.get(name)
    assert isinstance(value, dict), name
    return cast("dict[str, object]", value)


def test_probe_reports_measured_capabilities_rather_than_assumed_ones() -> None:
    # Given: the actual host.
    evidence = r.probe_isolation()
    # When: isolation is probed.
    block = _observations(evidence)
    # Then: every mechanism is reported with a concrete observation, never a bare claim.
    for name in (
        "python_isolated_mode", "environment_scrubbed", "fresh_working_directory",
        "rlimits", "network_namespace", "mount_namespace", "readonly_bind_mount",
        "cgroup_limits", "socket_stub",
    ):
        assert "observed" in _entry(block, name), name
    assert evidence["scope"] == r.SCOPE
    assert evidence["backend"] == r.BACKEND


def test_network_namespace_is_reported_unavailable_with_a_reason() -> None:
    # Given: this host, where unshare is not permitted.
    block = _observations(r.probe_isolation())
    # When: the namespace probes are inspected.
    network = _entry(block, "network_namespace")
    mount = _entry(block, "mount_namespace")
    # Then: an unavailable mechanism carries the real error, not a silent omission.
    for probe in (network, mount):
        assert probe["observed"] is False
        assert str(probe["detail"]).strip()
    assert "Operation not permitted" in str(network["detail"])


def test_socket_stub_is_declared_bypassable_rather_than_presented_as_isolation() -> None:
    # Given: the fallback network path in force on this host.
    stub = _entry(_observations(r.probe_isolation()), "socket_stub")
    # When / Then: it is disclosed as bypassable and never as a network boundary.
    assert stub["bypassable"] is True
    assert stub["observed"] is True
    assert "not a network namespace" in str(stub["detail"])


def test_rlimits_are_read_back_from_inside_the_sandboxed_child() -> None:
    # Given: a child started with the evaluator's own resource limiter.
    rlimits = _entry(_observations(r.probe_isolation()), "rlimits")
    # When / Then: the enforced values are observed, not merely configured.
    assert rlimits["observed"] is True, rlimits
    values = cast("dict[str, object]", rlimits["values"])
    assert set(values) == {"RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NPROC"}
    assert values["RLIMIT_NPROC"] == [0, 0]
    assert values["RLIMIT_FSIZE"] == [1024 * 1024, 1024 * 1024]


def test_evidence_records_the_unmet_plan_wording_explicitly() -> None:
    # Given: the plan's requirement for an externally managed evaluator.
    evidence = r.probe_isolation()
    # When / Then: the gap is stated rather than hidden, and no container claim is made.
    assert "externally managed" in r.UNMET_PLAN_WORDING
    assert evidence["unmet_plan_wording"] == r.UNMET_PLAN_WORDING
    claims = evidence["claims_not_made"]
    assert isinstance(claims, list)
    assert any("container or VM isolation" == claim for claim in cast("list[object]", claims))


def test_stable_projection_excludes_host_versions_but_keeps_capabilities() -> None:
    # Given: two probes that differ only in host build metadata.
    evidence = r.probe_isolation()
    drifted = dict(evidence)
    drifted["host"] = {"kernel": "9.9.9-other", "python": "3.13.0"}
    # When / Then: the projection is stable, and a changed capability is not.
    assert r.stable_projection(evidence) == r.stable_projection(drifted)
    downgraded = cast("dict[str, object]", json.loads(json.dumps(evidence)))
    cast("dict[str, object]", downgraded["observations"])["network_namespace"] = {"observed": True, "detail": "gained"}
    assert r.stable_projection(downgraded) != r.stable_projection(evidence)


def test_verify_task_accepts_discriminating_check_and_rejects_a_weak_one() -> None:
    # Given: a real check and a check that cannot fail.
    # When / Then: discrimination is required, not just a passing baseline.
    assert r.verify_task(SQUARE, SQUARE_CHECK, entry_point="square") == (
        True, "baseline passed and both mutants were rejected",
    )
    passed, reason = r.verify_task(SQUARE, WEAK_CHECK, entry_point="square")
    assert passed is False
    assert "mutant" in reason


def test_evaluate_source_reports_a_failing_baseline_without_raising() -> None:
    # Given: a check that contradicts the source.
    outcome = r.evaluate_source(SQUARE, "assert square(3) == 10\n", entry_point="square")
    # When / Then: the failure is reported as an outcome, not an exception.
    assert outcome.passed is False
    assert outcome.timed_out is False
    assert outcome.isolation_mode in {"unshare", "socket_stub"}


def test_write_isolation_evidence_binds_the_manifest_to_the_evidence_bytes(tmp_path: Path) -> None:
    # Given: an empty output root.
    manifest_path, evidence_path = r.write_isolation_evidence(tmp_path)
    # When: both declarations are read back.
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    evidence_bytes = evidence_path.read_bytes()
    # Then: the manifest's digest describes exactly those bytes and the scope stays honest.
    assert manifest["sha256"] == sha256(evidence_bytes).hexdigest()
    assert manifest["isolation_evidence_path"] == r.EVIDENCE_NAME
    assert manifest["scope"] == r.SCOPE
    assert manifest["isolation_backend"] == r.BACKEND
    assert manifest["network_isolation"] == "socket_stub_best_effort_bypassable"
    assert "externally managed" in str(manifest["unmet_plan_wording"])


def test_write_isolation_evidence_is_idempotent_but_never_silently_replaces(tmp_path: Path) -> None:
    # Given: published evidence.
    manifest_path, evidence_path = r.write_isolation_evidence(tmp_path)
    original = evidence_path.read_bytes()
    # When: it is published again unchanged, then the evidence is tampered with.
    _ = r.write_isolation_evidence(tmp_path)
    assert evidence_path.read_bytes() == original
    _ = evidence_path.write_bytes(b'{"forged": true}\n')
    # Then: a conflicting rewrite is refused rather than quietly overwritten.
    with pytest.raises(RuntimeError, match="already exists"):
        _ = r.write_isolation_evidence(tmp_path)
    assert evidence_path.read_bytes() == b'{"forged": true}\n'
    assert manifest_path.is_file()


def test_evaluator_module_never_imports_the_network_module() -> None:
    # Given: the evaluator source.
    tree = ast.parse(Path(r.__file__).read_text(encoding="utf-8"))
    # When: its real imports are collected.
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    # Then: it cannot reach the acquisition path.
    assert not [name for name in modules if name.endswith("asset_acquisition")]
