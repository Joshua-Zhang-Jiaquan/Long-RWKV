from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import pytest

from scale.experiments.nonlatent_iclr import inventory, service
from scale.experiments.nonlatent_iclr.service import AuditPaths, prepare_task_one, verify_task_one


METADATA_PATHS = (
    "outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/meta.json",
    "outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00006000_probe_copy/meta.json",
    "models/RWKV7-Goose-World3-2.9B-HF/tokenizer_config.json",
)
LM1B_RECORD = (
    "cap_lm_m4loop_s4750/lm1b/merged/"
    "merged_m4loop_endpoint_ckpt_nll-ppl-bits.json"
)
WIKITEXT_RECORD = (
    "cap_lm_m4loop_s4750/wikitext103/merged/"
    "merged_m4loop_endpoint_ckpt_nll-ppl-bits.json"
)
SAMPLER_RECORD = (
    "sampler_gate_m4_loop_s4750/merged/"
    "merged_m4loop_endpoint_ckpt_em-tau-residue-max_run_frac-distinct_frac.json"
)


class ArtifactJson(TypedDict):
    source: str
    origin: str
    path: str
    kind: str
    identity: str
    sha256: str | None
    bytes: int | None
    file_count: int
    reason: str | None


class ClaimJson(TypedDict):
    claim_id: str
    source: str
    state: str
    detail: str
    records: list[str]
    fields: list[str]


class ViewJson(TypedDict):
    source: str
    state: str
    detail: str


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _paths(
    tmp_path: Path, *, live: bool = False, staged: bool = False,
    missing_panel: str | None = None,
) -> AuditPaths:
    repo = tmp_path / "repo"
    snapshot = repo / "DAN/v7_arch_round"
    for _, relative, _ in service.SOURCES:
        _write(snapshot / relative, f"snapshot:{relative}\n")

    external = tmp_path / "external"
    for panel in service.PANEL_NAMES:
        if panel == missing_panel:
            continue
        if panel.endswith(".csv"):
            _write(external / panel, "4750,4980736000.0,1.222e+20,47.00\n")
        elif panel == "cap_lm_m4loop_s4750":
            _write(external / LM1B_RECORD, '{"record_count": 1, "metric_schema": ["nll", "ppl", "bits"]}\n')
            _write(external / WIKITEXT_RECORD, '{"record_count": 1, "metric_schema": ["nll", "ppl", "bits"]}\n')
        elif panel == "sampler_gate_m4_loop_s4750":
            _write(external / SAMPLER_RECORD, '{"record_count": 1, "metric_schema": ["em", "tau", "max_run_frac"]}\n')
        else:
            _write(external / panel / "merged/record.json", '{"record_count": 1}\n')
    for relative in METADATA_PATHS:
        _write(external / relative, '{"step": 4750, "tokens_seen": 4980736000.0}\n')

    live_root = tmp_path / "live-scale"
    staged_root = tmp_path / "staged-scale"
    if live:
        live_root.mkdir()
    if staged:
        staged_root.mkdir()
    return AuditPaths.from_roots(
        repo,
        tmp_path / "evidence",
        external_root=external,
        live_root=live_root,
        staged_root=staged_root,
    )


def _ledger(paths: AuditPaths) -> dict[str, object]:
    return cast(dict[str, object], json.loads(paths.ledger_path.read_text(encoding="utf-8")))


def _artifacts(paths: AuditPaths) -> list[ArtifactJson]:
    return cast(list[ArtifactJson], _ledger(paths)["artifacts"])


def _claims(paths: AuditPaths) -> list[ClaimJson]:
    return cast(list[ClaimJson], _ledger(paths)["claims"])


def _views(paths: AuditPaths) -> list[ViewJson]:
    return cast(list[ViewJson], _ledger(paths)["views"])


def test_inventory_has_explicit_roots_and_exact_panel_identities(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    ledger = _ledger(paths)

    assert ledger["source_roots"] == {
        "snapshot": str((paths.repo_root / "DAN/v7_arch_round").resolve()),
        "external": str(paths.external_root),
        "live": str(paths.live_root),
        "staged": str(paths.staged_root),
    }
    external_raw = [
        item for item in _artifacts(paths)
        if item["origin"] == "external" and item["kind"] == "raw_record"
    ]
    assert {item["path"] for item in external_raw} == set(service.PANEL_NAMES)
    assert len(external_raw) == 13
    for item in external_raw:
        assert item["identity"] == "sha256"
        assert item["sha256"] is not None
        assert len(item["sha256"]) == 64
        assert type(item["bytes"]) is int and item["bytes"] > 0
        assert type(item["file_count"]) is int and item["file_count"] > 0

    metadata = {
        item["path"]: item for item in _artifacts(paths)
        if item["kind"] == "metadata"
    }
    assert set(metadata) == set(METADATA_PATHS)
    assert all(item["origin"] == "external" for item in metadata.values())
    assert all(item["identity"] == "sha256" for item in metadata.values())


@pytest.mark.parametrize("relative", (SAMPLER_RECORD, METADATA_PATHS[0]))
def test_external_hashed_source_drift_is_stale(tmp_path: Path, relative: str) -> None:
    paths = _paths(tmp_path)
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    _write(paths.external_root / relative, "changed\n")
    assert verify_task_one(paths).status == "STALE"


def test_missing_named_panel_remains_explicit_and_unresolved(tmp_path: Path) -> None:
    missing = "sampler_gate_m4_loop_s4750_reps4"
    paths = _paths(tmp_path, missing_panel=missing)
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    entry = next(item for item in _artifacts(paths) if item["path"] == missing)
    assert entry["identity"] == "missing"
    assert entry["file_count"] == 0
    assert entry["sha256"] is None


def test_historical_claims_bind_exact_records_and_fields(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    claims = _claims(paths)
    for term in ("LM1B", "WikiText103", "4.98B", "max_run_frac", "masked-token accuracy", "tau", "r100"):
        claim = next(item for item in claims if term in item["detail"])
        assert claim["state"] in {"claimed", "unresolved"}
        assert claim["records"]
        assert claim["fields"]
        assert "task2" in claim["detail"]
    assert LM1B_RECORD in next(item for item in claims if "LM1B" in item["detail"])["records"]
    assert WIKITEXT_RECORD in next(item for item in claims if "WikiText103" in item["detail"])["records"]
    assert SAMPLER_RECORD in next(item for item in claims if "max_run_frac" in item["detail"])["records"]


@pytest.mark.parametrize(("live", "staged"), ((False, False), (True, False), (False, True), (True, True)))
def test_views_derive_state_and_detail_from_actual_roots(
    tmp_path: Path, live: bool, staged: bool
) -> None:
    paths = _paths(tmp_path, live=live, staged=staged)
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    views = {item["source"]: item for item in _views(paths)}
    assert views["snapshot"]["state"] == "observed"
    for name, exists in (("live", live), ("staged", staged)):
        assert views[name]["state"] == ("observed" if exists else "unresolved")
        assert ("present" if exists else "absent") in views[name]["detail"]


def test_view_presence_drift_is_stale(tmp_path: Path) -> None:
    paths = _paths(tmp_path, staged=False)
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    paths.staged_root.mkdir()
    assert verify_task_one(paths).status == "STALE"


def test_panel_file_limit_is_visible_not_silently_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(service, "MAX_PANEL_FILES", 1)
    assert prepare_task_one(paths).status == "AUDIT_COMPLETE"
    entry = next(
        item for item in _artifacts(paths)
        if item["path"] == "cap_lm_m4loop_s4750"
    )
    assert entry["identity"] == "unverified"
    assert entry["file_count"] == 2
    assert entry["reason"] is not None
    assert "file limit" in entry["reason"]


def test_panel_disappearing_during_initial_inventory_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a panel file that is removed immediately after directory listing.
    paths = _paths(tmp_path)
    original_panel_files = inventory._panel_files

    def remove_after_listing(target: Path) -> tuple[list[Path], bool]:
        files, unsafe = original_panel_files(target)
        if target.name == "cap_lm_n2_s4000" and files:
            files[0].unlink()
        return files, unsafe

    monkeypatch.setattr(inventory, "_panel_files", remove_after_listing)
    # When: initial preparation observes that source race.
    result = prepare_task_one(paths)
    # Then: the service returns a structured failure without publishing a ledger.
    assert result.status == "SOURCE_CHANGED"
    assert not paths.ledger_path.exists()
