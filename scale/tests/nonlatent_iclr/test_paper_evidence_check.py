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
    # named so a reader can see what the check could not reach.  `complete` is
    # False here only because no manuscript was supplied, which is a different
    # statement from "the claims are wrong" and is reported as such.
    assert report["verdict"] == "coverage_not_checked"
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


# --------------------------------------------------------------------------
# coverage of the PAPER, not of the inventory
# --------------------------------------------------------------------------


def test_latex_thousands_separators_are_not_split() -> None:
    """`{,}` is how LaTeX writes a thousands separator, and splitting it invents gaps.

    A naive digit-run extractor turns `\\textbf{11{,}567}` into `11` and `567`, both
    of which then read as uncatalogued numbers that do not exist. A coverage report
    full of phantom gaps trains a reader to ignore it, which is the failure this
    whole module is written against.
    """
    # Given: a manuscript line with a LaTeX-separated number.
    text = r"The ledger holds \textbf{11{,}567} GPU-hours and 3{,}600 units."
    # When: its numbers are extracted.
    found = pec.paper_numbers(text)
    # Then: the numbers are whole, not fragments.
    assert "11,567" in found
    assert "3,600" in found
    assert "567" not in found
    assert "600" not in found


def test_citations_and_years_are_not_treated_as_claims() -> None:
    # Given: a line carrying an arXiv id and a year.
    text = r"See \cite{x} (2605.03042, 2026) for the 4096-token canvas."
    found = pec.paper_numbers(text)
    # When/Then: the identifier and the year are excluded, the measurement is not.
    assert "2605" not in found
    assert "2026" not in found
    assert "4096" in found


def test_a_number_no_claim_covers_is_reported() -> None:
    """The hole this was written for: the paper cited a step no claim held."""
    # Given: a manuscript citing a number and an inventory that does not hold it.
    text = "stopping near step 20{,}600 at its cap"
    claims = [pec.Claim("other", "12345", "x.json", "measured", near=r"x")]
    # When: coverage is checked.
    gaps = pec.uncatalogued_numbers(text, claims)
    # Then: the number IS reported -- the old verdict was "complete" while the
    # manuscript cited a number nothing carried.
    assert "20,600" in gaps


def test_a_declared_axis_written_as_a_set_is_not_a_gap() -> None:
    r"""The load axis reads as a comma-grouped number, and that is genuinely ambiguous.

    "\{1,8,32,128\}" ends in ",128", so the pair (32, 128) is character-for-character
    the thousands-grouped 32,128. The ambiguity cannot be resolved from the token, so
    coverage accepts either reading -- the whole token if catalogued, or every part
    if they all are. Rejecting a correctly-declared axis would be a false gap, which
    is the failure that makes a check ignorable.
    """
    # Given: the axis written as a set, with all its members declared.
    text = r"loads $\{1,8,32,128\}$"
    claims = [pec.Claim("a", "32", "x.json", "measured", near=r"x"),
              pec.Claim("b", "128", "x.json", "measured", near=r"x")]
    # When/Then: no gap -- both readings are covered.
    assert pec.uncatalogued_numbers(text, claims) == []


def test_an_ambiguous_token_withan_undeclared_part_is_still_a_gap() -> None:
    """The ambiguity must not become a loophole.

    If only one part of "32,128" is declared, the token is NOT covered: accepting it
    anyway would let an undeclared number hide behind a declared neighbour.
    """
    text = r"loads $\{1,8,32,128\}$"
    claims = [pec.Claim("b", "128", "x.json", "measured", near=r"x")]
    assert pec.uncatalogued_numbers(text, claims) == ["32,128"]


def test_a_declared_non_claim_is_exempt_but_named() -> None:
    # Given: an inventory that declares a protocol constant structural.
    inventory = {"non_claims": [{"value": "1000", "reason": "the protocol's request floor"}]}
    declared = pec.declared_non_claims(inventory)
    # When: coverage runs.
    gaps = pec.uncatalogued_numbers("at least 1{,}000 requests", [], declared)
    # Then: exempt -- but the exemption is a NAMED reason, not a silent skip.
    assert gaps == []
    assert "request floor" in declared["1000"]


def test_the_real_inventory_covers_the_real_manuscript() -> None:
    """The verdict must be about the paper, not about the inventory.

    This is the check that would have caught a cited number no claim held: it
    extracts every substantive number from main.tex and requires each to be
    catalogued with a source or declared structural with a reason.
    """
    repo = Path(__file__).resolve().parents[3]
    inventory_path = repo / "DAN" / "nonlatent_iclr" / "paper_claims.json"
    paper = repo / "paper" / "main.tex"
    if not (inventory_path.is_file() and paper.is_file()):
        pytest.skip("inventory or manuscript absent")
    inventory = json.loads(inventory_path.read_text())
    claims = pec.load_inventory(inventory_path)
    gaps = pec.uncatalogued_numbers(paper.read_text(), claims,
                                    pec.declared_non_claims(inventory))
    assert gaps == [], f"numbers in the manuscript that nothing covers: {gaps}"
    report = pec.check_all(claims, repo, paper=paper, inventory=inventory)
    assert report["complete"] is True
    assert report["counts"]["value_not_found"] == 0


def test_coverage_runs_with_no_inventory_declared(tmp_path: Path) -> None:
    """`inventory` is optional, and passing only `paper` must not crash.

    It did: check_all passed None into declared_non_claims, which called .get on
    it. A caller who wanted coverage without exemptions got an AttributeError
    instead of a coverage report.
    """
    # Given: a manuscript and a claim set, but no inventory document.
    paper = tmp_path / "m.tex"
    paper.write_text("the ledger holds 11{,}567 GPU-hours", encoding="utf-8")
    claims = [pec.Claim("c", "11,567", "x.json", "measured", near=r"x")]
    # When: coverage is checked with paper but no inventory.
    report = pec.check_all(claims, tmp_path, paper=paper, inventory=None)
    # Then: a report, not a crash -- and the number IS found, so no gap.
    assert report["complete"] is False or report["uncatalogued"] == []
    assert report["uncatalogued"] == []


def test_a_short_declared_constant_does_not_exempt_longer_numbers() -> None:
    """The containment direction, which was backwards and produced a FALSE PASS.

    The load axis declares "1". An earlier version exempted any paper number that
    CONTAINED a catalogue entry, so "193" and "511" passed as covered -- because
    they contain "1" -- while nothing catalogued them. A check that exempts
    whatever it is shown is worse than no check, because it reports coverage.
    """
    # Given: a manuscript citing 193 and 511, and a catalogue holding only "1".
    text = "of the 193 completions the longest run is 511"
    claims = [pec.Claim("load", "1", "x.json", "measured", near=r"x")]
    # When: coverage runs.
    gaps = pec.uncatalogued_numbers(text, claims)
    # Then: both are reported, because "1" is not evidence for either.
    assert "193" in gaps
    assert "511" in gaps


def test_a_fragment_of_a_longer_catalogued_number_is_still_exempt() -> None:
    """The direction that IS wanted: a fragment of a catalogued number."""
    # Given: "s8000" yielding "800", where "8000" is declared structural.
    text = "at s8000 the endpoint is 9{,}500"
    declared = {"8000": "a checkpoint step label", "9500": "a checkpoint step label"}
    # When/Then: neither fragment is reported.
    assert pec.uncatalogued_numbers(text, [], declared) == []


def test_pinned_git_revisions_are_not_numbers() -> None:
    """A commit pin like LongBench@2e00731f must not yield "00731"."""
    # Given: a bibliography line with a pinned revision.
    text = r"\bibitem{longbench} THUDM LongBench, \texttt{THUDM/LongBench@2e00731f}."
    # When/Then: the hex fragment is not extracted as a claim.
    assert not any(tok.lstrip("0") in {"731"} or tok == "00731"
                   for tok in pec.paper_numbers(text))


def test_a_new_number_is_needs_review_not_missing_evidence() -> None:
    r"""Failing to cover a number is not the same as finding the paper wrong.

    A gate that reports a NEW legitimate number as a failure is indistinguishable
    from the negative result it destroys -- and unlike a false pass, it surfaces
    only after the work it was meant to certify. So the two states are named
    separately: `missing_evidence` means the check looked and found a problem,
    `needs_review` means the check has not been extended to that number yet.
    """
    # Given: a manuscript citing a number the inventory does not yet carry.
    text = "the new measurement is 4271 tokens"
    # an external claim verifies without touching disk, so the fixture cannot
    # itself introduce a path_missing that masks what this test is measuring
    claims = [pec.Claim("known", "100", "/elsewhere", "external", appears_in="stated")]
    # When: coverage runs.
    report = pec.check_all(claims, Path("."), paper_text=text)
    # Then: review is needed, but nothing was found to be WRONG.
    assert report["verdict"] == "needs_review"
    assert report["counts"]["value_not_found"] == 0
    assert report["uncatalogued"] == ["4271"]


def test_a_missing_value_is_missing_evidence() -> None:
    # Given: a claim whose cited source does not exist.
    report = pec.check_all(
        [pec.Claim("bad", "9", "absent.json", "measured", near=r'"x":\s*[0-9.]+')],
        Path("."))
    # When/Then: that IS a finding, and it is named as one.
    assert report["verdict"] == "missing_evidence"
    assert report["complete"] is False


def test_a_clean_check_reports_ok_only_when_coverage_ran() -> None:
    """`ok` claims two things: the claims verify AND the paper was covered.

    It is reachable only when both were actually done, so the state is not
    inferred from an absence of findings.
    """
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    # coverage run over a manuscript that cites nothing substantive
    report = pec.check_all(claims, Path("."), paper_text="")
    assert report["coverage_checked"] is True
    assert report["verdict"] == "ok"
    assert report["complete"] is True
    # and without it, the same claims do NOT earn `ok`
    assert pec.check_all(claims, Path("."))["verdict"] == "coverage_not_checked"


def test_a_verdict_without_coverage_says_so() -> None:
    """The verdict must not claim what it never tested.

    Run without a manuscript, coverage is never RUN, and every assertion about
    the OUTCOME stays green: the claims verify, nothing is missing, so a naive
    verdict says `ok` and `complete` -- asserting the paper is covered on the
    strength of not having looked at it. That is the same defect as a null that
    prints a formatted nan and calls the score "NOT separable": the decision can
    be right while the CLAIM is wrong, and a test that reads only the outcome
    cannot see the difference. So these tests read the justification.
    """
    # Given: a clean set of claims and NO manuscript.
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    # When: the check runs.
    report = pec.check_all(claims, Path("."))
    # Then: it does NOT claim coverage.
    assert report["coverage_checked"] is False
    assert report["verdict"] == "coverage_not_checked"
    assert report["complete"] is False
    # and the state is named rather than left to be inferred from an absence
    assert report["verdict"] != "ok"


def test_coverage_actually_run_is_reported_as_run() -> None:
    # Given: the same claims WITH a manuscript.
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    # When/Then: coverage runs, and the verdict may then say so.
    report = pec.check_all(claims, Path("."), paper_text="the ledger holds 11{,}567")
    assert report["coverage_checked"] is True
    assert report["verdict"] == "needs_review"      # 11,567 is not catalogued here
    assert "11,567" in report["uncatalogued"]
