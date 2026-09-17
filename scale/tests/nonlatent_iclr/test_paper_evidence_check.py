"""The paper's own correspondence claim, made checkable.

The manuscript asserts its numbers "are machine-verifiable rather than asserted".
These tests pin the checker that makes that true, and they exist mostly to pin the
FALSE-PASS guards, because a check that reports a pass it did not earn is worse
than no check at all:

* a bare substring match is not evidence -- on the real tree ``56.60`` occurs in
  ``length_matrix.json`` as part of an unrelated token count and ``200`` occurs
  inside an id like ``rt-9fb7bdda200f-014``;
* a correctly ROUNDED transcription is not missing evidence -- the ledger holds
  ``886.659...`` and the paper writes ``886.7``;
* absence has to be reported as absence, not as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr import paper_evidence_check as pec


def _inventory(tmp_path: Path, claims: list[dict]) -> Path:
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({"schema": pec.SCHEMA, "claims": claims}), encoding="utf-8")
    return path


def _source(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def _claim(**over) -> dict:
    base = {"claim_id": "c", "value": "1", "source": "s.json", "kind": "measured",
            "near": r'"x":\s*[0-9.]+'}
    base.update(over)
    return base


# --------------------------------------------------------------------------
# the inventory is checked, not trusted
# --------------------------------------------------------------------------


def test_a_wrong_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"schema": "other", "claims": [_claim()]}))
    with pytest.raises(pec.EvidenceCheckRefusal, match="is not"):
        pec.load_inventory(path)


def test_a_missing_field_is_refused(tmp_path: Path) -> None:
    bad = _claim()
    del bad["source"]
    with pytest.raises(pec.EvidenceCheckRefusal, match="missing 'source'"):
        pec.load_inventory(_inventory(tmp_path, [bad]))


def test_a_duplicate_claim_id_is_refused(tmp_path: Path) -> None:
    with pytest.raises(pec.EvidenceCheckRefusal, match="duplicate claim_id"):
        pec.load_inventory(_inventory(tmp_path, [_claim(), _claim()]))


def test_an_empty_inventory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(pec.EvidenceCheckRefusal, match="no claims"):
        pec.load_inventory(_inventory(tmp_path, []))


# --------------------------------------------------------------------------
# the verdicts
# --------------------------------------------------------------------------


def test_a_value_present_in_context_verifies(tmp_path: Path) -> None:
    _source(tmp_path, "s.json", '{"x": 42}')
    result = pec.check_claim(pec.Claim("c", "42", "s.json", "measured",
                                       near=r'"x":\s*[0-9.]+'), tmp_path)
    assert result.verdict == "verified"


def test_a_coincidental_substring_does_not_verify(tmp_path: Path) -> None:
    """The false-positive guard, on the real tree's own shape."""
    # Given: a file where the value occurs ONLY outside any labelled context --
    # exactly how 56.60 occurs inside an unrelated token count.
    _source(tmp_path, "s.json", '{"token_max": 56.60, "id": "rt-9fb7bdda200f"}')
    # When: the claim names a context the value does not appear in.
    result = pec.check_claim(pec.Claim("c", "56.60", "s.json", "measured",
                                       near=r'"obqa":\s*[0-9.]+'), tmp_path)
    # Then: it is NOT verified. A bare substring search would have passed it.
    assert result.verdict == "value_not_found"


def test_a_claim_without_a_context_cannot_verify(tmp_path: Path) -> None:
    # Given: a claim that names no context for its value.
    _source(tmp_path, "s.json", '{"x": 42}')
    # When/Then: refused as unverifiable rather than greped, because without a
    # context a match cannot be told from a coincidence.
    result = pec.check_claim(pec.Claim("c", "42", "s.json", "measured"), tmp_path)
    assert result.verdict == "value_not_found"
    assert "no `near` context" in result.detail


def test_a_correctly_rounded_value_is_not_missing_evidence(tmp_path: Path) -> None:
    """The ledger's 886.659... against the paper's 886.7."""
    # Given: the artifact's full-precision value.
    _source(tmp_path, "s.json", '{"tokens_per_second": 886.6594840890932}')
    # When: the paper cites it rounded.
    result = pec.check_claim(pec.Claim("c", "886.7", "s.json", "measured",
                                       near=r'"tokens_per_second":\s*[0-9.]+'), tmp_path)
    # Then: rounding_ok, not value_not_found -- calling a correct rounding "missing
    # evidence" would train a reader to ignore the check.
    assert result.verdict == "rounding_ok"
    assert "rounded to 1 dp" in result.detail


def test_a_misrounded_value_is_still_caught(tmp_path: Path) -> None:
    # Given: the same artifact.
    _source(tmp_path, "s.json", '{"tokens_per_second": 886.6594840890932}')
    # When: the paper cites a number that is NOT its rounding.
    result = pec.check_claim(pec.Claim("c", "887.7", "s.json", "measured",
                                       near=r'"tokens_per_second":\s*[0-9.]+'), tmp_path)
    # Then: caught -- the tolerance separates rounding from a wrong number.
    assert result.verdict == "value_not_found"


def test_a_missing_file_is_reported_as_missing(tmp_path: Path) -> None:
    result = pec.check_claim(pec.Claim("c", "1", "absent.json", "measured",
                                       near=r'"x":\s*[0-9.]+'), tmp_path)
    assert result.verdict == "path_missing"


def test_external_is_a_disclosure_not_a_pass(tmp_path: Path) -> None:
    # Given: a claim whose source the repo does not ship.
    result = pec.check_claim(pec.Claim("c", "56.60", "/cluster/x.csv", "external",
                                       appears_in="the cluster CSV only"), tmp_path)
    # When/Then: reported as external with its reason, and the report counts it
    # separately from verified.
    assert result.verdict == "external"
    assert "cluster CSV" in result.detail
    report = pec.check_all([pec.Claim("c", "56.60", "/x", "external",
                                      appears_in="cluster only")], tmp_path)
    assert report["counts"]["verified"] == 0
    assert report["disclosed"] == ["c"]


def test_a_derived_value_cannot_be_greped_and_is_disclosed(tmp_path: Path) -> None:
    # Given: an arithmetic aggregate over artifacted inputs.
    result = pec.check_claim(pec.Claim("c", "11,567", "s.json", "derived",
                                       appears_in="sum of the per-arm rows"), tmp_path)
    # When/Then: disclosed as derived rather than reported missing.
    assert result.verdict == "derived"


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


def test_the_report_is_incomplete_while_anything_is_missing(tmp_path: Path) -> None:
    # Given: one verified claim and one whose value is absent.
    _source(tmp_path, "s.json", '{"x": 1}')
    report = pec.check_all([
        pec.Claim("good", "1", "s.json", "measured", near=r'"x":\s*[0-9.]+'),
        pec.Claim("bad", "9", "s.json", "measured", near=r'"x":\s*[0-9.]+'),
    ], tmp_path)
    # When/Then: incomplete, and the bad one is listed rather than summarised away.
    assert report["complete"] is False
    assert report["counts"]["verified"] == 1
    assert report["counts"]["value_not_found"] == 1


def test_a_report_with_only_disclosures_is_complete(tmp_path: Path) -> None:
    report = pec.check_all([pec.Claim("e", "1", "/x", "external", appears_in="elsewhere")],
                           tmp_path)
    # Then: external and derived are disclosures, not failures -- but they are
    # named so a reader can see what the check could not reach.
    assert report["complete"] is True
    assert report["disclosed"] == ["e"]


# --------------------------------------------------------------------------
# the real inventory
# --------------------------------------------------------------------------


def test_the_real_inventory_complete_and_contextual() -> None:
    """The shipped inventory must be satisfiable and must use contexts."""
    repo = Path(__file__).resolve().parents[3]
    inventory = repo / "DAN" / "nonlatent_iclr" / "paper_claims.json"
    if not inventory.is_file():
        pytest.skip(f"inventory absent: {inventory}")
    claims = pec.load_inventory(inventory)
    assert len(claims) >= 20
    # every non-external/derived claim must name the context it should appear in,
    # or it could only ever be greped
    for claim in claims:
        if claim.kind not in ("external", "derived"):
            assert claim.near, f"{claim.claim_id} has no context to check against"
    report = pec.check_all(claims, repo)
    assert report["counts"]["value_not_found"] == 0, [
        r for r in report["results"] if r["verdict"] == "value_not_found"]
    assert report["complete"] is True


def test_the_disclosure_classes_are_named_in_the_report() -> None:
    """A reader must be able to see what the check could not reach."""
    repo = Path(__file__).resolve().parents[3]
    inventory = repo / "DAN" / "nonlatent_iclr" / "paper_claims.json"
    if not inventory.is_file():
        pytest.skip(f"inventory absent: {inventory}")
    report = pec.check_all(pec.load_inventory(inventory), repo)
    # the ledger totals are sums and cannot be greped; they are the only
    # disclosures left now that the descriptive readings are recorded
    assert report["counts"]["derived"] >= 1
    assert report["counts"]["external"] == 0
