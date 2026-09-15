from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import cast

from scale.experiments.nonlatent_iclr.registry_task import IMPLEMENTATION_SOURCES
from scale.experiments.nonlatent_iclr.registry_task import EXTERNAL_SUITES_NOT_CONSUMED
from scale.experiments.nonlatent_iclr.registry_task import EXTERNAL_SUITES_STATUS_SCOPE
from scale.experiments.nonlatent_iclr.registry_task import main
from scale.experiments.nonlatent_iclr.registry_task import implementation_source_hashes
from scale.experiments.nonlatent_iclr.registry_task import receipt_is_current
from scale.experiments.nonlatent_iclr.registry_task import load_accepted_length_qualification
from scale.experiments.nonlatent_iclr.task_registry import build_registry
from scale.experiments.nonlatent_iclr.task_registry import registry_document
from scale.experiments.nonlatent_iclr.task_registry import unbound_length_qualification
from scale.experiments.nonlatent_iclr.task_registry import JsonValue
from scale.experiments.nonlatent_iclr.task_registry import verify_registry
from scale.experiments.nonlatent_iclr.tasks.length_qualification import REPRESENTATIVE_ARTIFACT
from scale.experiments.nonlatent_iclr.tasks.length_qualification import SUMMARY_NAME


def test_registry_declares_full_matrix_lazily_when_built() -> None:
    # Given: the CPU task registry
    registry = build_registry()

    # When: its cells are inspected without materializing instances
    cells = registry.cells

    # Then: all declared axes and 200 instances per cell are retained as lazy declarations
    assert registry.data_seeds == (101, 102, 103, 104, 105)
    assert {cell.length for cell in cells} == {4096, 8192, 16384, 32768, 65536}
    assert {cell.position_fraction for cell in cells} == {10, 50, 90}
    assert {cell.load for cell in cells} == {1, 8, 32, 128}
    assert {cell.distractor for cell in cells} == {"none", "random", "similar"}
    assert all(cell.instances_per_cell == 200 for cell in cells)
    assert not registry.materialized_instances


def test_grouped_variants_remain_in_one_split_when_assigned() -> None:
    # Given: related paraphrase, location, and load views of the same source family
    registry = build_registry()

    # When: provenance is checked before derived views
    violations = registry.split_violations()

    # Then: no family can straddle a split
    assert violations == ()


def test_external_assets_fail_closed_when_local_paths_are_absent(tmp_path: Path) -> None:
    # Given: an intentionally empty local asset root
    registry = build_registry(asset_root=tmp_path)

    # When: benchmark readiness is verified
    report = verify_registry(registry)

    # Then: every unmet source and isolated evaluator requirement blocks completion
    assert report.ready is False
    assert report.exit_code == 2
    assert {item.requirement for item in report.blocked} == {
        "ruler_development_suite",
        "longbench_development_suite",
        "rights_cleared_repository_tasks",
        "isolated_repository_evaluator",
        "rwkv_tokenizer",
    }


def test_standalone_failure_verification_confirms_rejections(tmp_path: Path) -> None:
    # Given: a clean output root and unavailable external assets
    output = tmp_path / "out"

    # When: the standalone entrypoint runs its planted-failure checks
    exit_code = main(("verify", "--case", "failure", "--output-root", str(output), "--asset-root", str(tmp_path)))

    # Then: each negative condition is rejected, without claiming benchmark readiness
    assert exit_code == 0
    assert (output / "task04_failure_verification.json").is_file()


def test_placeholder_manifests_cannot_unlock_benchmark_readiness(tmp_path: Path) -> None:
    # Given: structurally plausible but empty-provenance manifests at every declared path
    registry = build_registry(asset_root=tmp_path)
    for requirement in (
        "ruler/development_manifest.json",
        "longbench/development_manifest.json",
        "repository_tasks/manifest.json",
        "repository_evaluator/isolation_manifest.json",
        "tokenizer/rwkv_tokenizer_manifest.json",
    ):
        path = tmp_path / requirement
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(json.dumps({"version": "", "license": "", "sha256": "", "isolation": "external_vm"}), encoding="utf-8")

    # When: production asset readiness is evaluated
    report = verify_registry(registry)

    # Then: placeholders cannot make the registry READY
    assert report.ready is False
    assert len(report.blocked) == 5


def test_nonempty_arbitrary_manifest_strings_cannot_unlock_readiness(tmp_path: Path) -> None:
    # Given: nonempty but unbound manifest claims, including an evaluator backend name
    registry = build_registry(asset_root=tmp_path)
    for requirement in (
        "ruler/development_manifest.json",
        "longbench/development_manifest.json",
        "repository_tasks/manifest.json",
        "repository_evaluator/isolation_manifest.json",
        "tokenizer/rwkv_tokenizer_manifest.json",
    ):
        path = tmp_path / requirement
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(json.dumps({"version": "invented", "license": "invented", "sha256": "a" * 64, "isolation": "external_vm"}), encoding="utf-8")

    # When: the production readiness check parses the claims
    report = verify_registry(registry)

    # Then: no unbound declaration can qualify a real asset or evaluator
    assert report.ready is False
    assert len(report.blocked) == 5


def test_report_accounts_for_feasible_and_infeasible_cells(tmp_path: Path) -> None:
    # Given: the full lazy matrix with unavailable external assets
    report = verify_registry(build_registry(asset_root=tmp_path))

    # When: representative cell construction is assessed
    # Then: each declared cell is counted exactly once without corpus materialization
    assert report.successful_cells + report.infeasible_cells == 720
    assert report.successful_cells > 0
    assert report.infeasible_cells > 0


def test_failure_verify_records_production_validator_rejections(tmp_path: Path) -> None:
    # Given: a fresh standalone failure-verification output root
    output = tmp_path / "out"

    # When: production validators receive isolated altered inputs
    exit_code = main(("verify", "--case", "failure", "--output-root", str(output), "--asset-root", str(tmp_path)))
    result = cast("dict[str, object]", json.loads((output / "task04_failure_verification.json").read_text(encoding="utf-8")))

    # Then: every rejection is captured and only genuine rejection success returns zero
    assert exit_code == 0
    assert result == {
        "answer_field_rejected": True,
        "fork_leakage_rejected": True,
        "silent_truncation_rejected": True,
        "test_tampering_rejected": True,
    }


def test_receipts_require_current_implementation_source_hashes(tmp_path: Path) -> None:
    # Given: an isolated copy of each implementation source tracked by a receipt
    source_root = tmp_path / "implementation"
    for relative in IMPLEMENTATION_SOURCES:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(relative, encoding="utf-8")
    hashes: dict[str, JsonValue] = {name: digest for name, digest in implementation_source_hashes(source_root).items()}
    receipt: dict[str, JsonValue] = {"registry_sha256": "r" * 64, "implementation_source_hashes": hashes}

    # When: one tracked source changes, or a legacy receipt lacks source hashes
    _ = (source_root / IMPLEMENTATION_SOURCES[0]).write_text("changed", encoding="utf-8")
    legacy: dict[str, JsonValue] = {"registry_sha256": "r" * 64}

    # Then: source freshness is mandatory and legacy unbound receipts are ineligible
    assert receipt_is_current(receipt, source_root) is False
    assert receipt_is_current(legacy, source_root) is False


def test_registry_document_without_a_bound_summary_stays_conservative() -> None:
    # Given: a registry whose asset root holds no accepted length-qualification summary
    registry = build_registry(asset_root=Path("/nonexistent-task4-root"))
    report = verify_registry(registry)

    # When: the document is rendered
    document = registry_document(registry, report)

    # Then: token qualification is explicitly unbound rather than asserted
    assert document["length_qualification"] == unbound_length_qualification()
    assert load_accepted_length_qualification(Path("/nonexistent-task4-root")) is None


def test_load_accepted_summary_requires_the_artifact_hash_to_bind(tmp_path: Path) -> None:
    # Given: a summary whose declared artifact digest does not describe the artifact on disk
    artifact = tmp_path / REPRESENTATIVE_ARTIFACT
    _ = artifact.write_bytes(b"real artifact bytes\n")
    summary = {
        "artifact": REPRESENTATIVE_ARTIFACT,
        "artifact_sha256": "0" * 64,
        "source_hashes": {"a.py": "1" * 64},
        "qualified_representative_cells": 689,
    }
    _ = (tmp_path / SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")

    # When / Then: an unbound or hash-mismatched summary is refused
    assert load_accepted_length_qualification(tmp_path) is None


def test_prepare_preserves_a_bound_length_qualification(tmp_path: Path) -> None:
    # Given: an asset root carrying a correctly bound summary and its artifact
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    artifact = asset_root / REPRESENTATIVE_ARTIFACT
    _ = artifact.write_bytes(b"real artifact bytes\n")
    summary = {
        "artifact": REPRESENTATIVE_ARTIFACT,
        "artifact_sha256": sha256(artifact.read_bytes()).hexdigest(),
        "source_hashes": {"a.py": "1" * 64},
        "qualified_representative_cells": 689,
        "matrix_token_lengths_qualified": False,
    }
    _ = (asset_root / SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "out"

    # When: the registry is prepared
    exit_code = main(("prepare", "--output-root", str(output), "--evidence-root", str(tmp_path / "ev"), "--asset-root", str(asset_root)))

    # Then: the accepted token evidence survives regeneration instead of being downgraded
    assert exit_code == 2
    document = cast("dict[str, object]", json.loads((output / "task_registry.json").read_text(encoding="utf-8")))
    emitted = document["length_qualification"]
    assert isinstance(emitted, dict)
    assert emitted["qualified_representative_cells"] == 689
    assert "source_hashes" not in emitted


def test_registry_scopes_the_length_artifact_external_suites_status(tmp_path: Path) -> None:
    # Given: a bound summary repeating the length artifact's own "external suites blocked" wording
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    artifact = asset_root / REPRESENTATIVE_ARTIFACT
    _ = artifact.write_bytes(b"real artifact bytes\n")
    summary = {
        "artifact": REPRESENTATIVE_ARTIFACT,
        "artifact_sha256": sha256(artifact.read_bytes()).hexdigest(),
        "source_hashes": {"a.py": "1" * 64},
        "external_suites_status": "BLOCKED; not read or qualified",
    }
    _ = (asset_root / SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")

    # When: the registry block is composed
    emitted = cast("dict[str, object]", load_accepted_length_qualification(asset_root))

    # Then: the artifact-local wording cannot be read as a global Task4 blocker, and the scope is named
    assert emitted["external_suites_status"] == EXTERNAL_SUITES_NOT_CONSUMED
    assert emitted["external_suites_status_scope"] == EXTERNAL_SUITES_STATUS_SCOPE
    assert "BLOCKED" not in str(emitted["external_suites_status"])


def test_prepare_archives_an_enriched_registry_before_replacing_it(tmp_path: Path) -> None:
    # Given: a previous registry carrying an enriched block but no bound summary now
    output = tmp_path / "out"
    output.mkdir()
    previous = {"schema_version": 1, "status": "BLOCKED", "length_qualification": {"qualified_representative_cells": 689}}
    _ = (output / "task_registry.json").write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()

    # When: the registry is regenerated without an accepted summary
    _ = main(("prepare", "--output-root", str(output), "--evidence-root", str(tmp_path / "ev"), "--asset-root", str(asset_root)))

    # Then: the superseded bytes are archived rather than silently lost
    archives = list(output.glob("task_registry.archive-*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text(encoding="utf-8")) == previous
