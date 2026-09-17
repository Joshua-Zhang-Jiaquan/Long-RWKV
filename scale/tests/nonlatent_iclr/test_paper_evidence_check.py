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
    # Then: it is NOT verified. A bare substring search would have passed it -- and
    # the finding says the value is present but outside the declared context, which
    # is more accurate than "the artifact does not carry it" and sends the reader to
    # the right place.
    assert result.verdict == "value_outside_context"


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


def test_a_report_with_only_disclosures_is_a_pass_once_coverage_runs(tmp_path: Path) -> None:
    claims = [pec.Claim("e", "1", "/x", "external", appears_in="elsewhere")]
    # Then: external and derived are disclosures, not failures -- but they are
    # named so a reader can see what the check could not reach.
    report = pec.check_all(claims, tmp_path, paper_text="")
    assert report["verdict"]["status"] == "pass"
    assert report["disclosed"] == ["e"]
    # and WITHOUT coverage the same claims are withheld, not passed: the pass is
    # earned by running, not inferred from the absence of findings
    withheld = pec.check_all(claims, tmp_path)
    assert withheld["verdict"]["status"] == "withheld"
    assert withheld["disclosed"] == ["e"]


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
    # Then: the verdict is WITHHELD -- not a finding, and not a pass.
    assert report["verdict"]["status"] == "withheld"
    assert report["counts"]["value_not_found"] == 0
    assert report["uncatalogued"] == ["4271"]
    # and the reason names the remedy, because a withhold with no route out is
    # just a block with better manners
    assert "4271" in report["verdict"]["withheld_reason"]
    assert "paper_claims.json" in report["verdict"]["remedy"]


def test_a_missing_value_is_missing_evidence() -> None:
    # Given: a claim whose cited source does not exist.
    report = pec.check_all(
        [pec.Claim("bad", "9", "absent.json", "measured", near=r'"x":\s*[0-9.]+')],
        Path("."))
    # When/Then: that IS a finding, and it is named as one -- a finding needs no
    # reason or remedy, because the finding IS the statement.
    assert report["verdict"]["status"] == "finding"
    assert report["verdict"]["withheld_reason"] is None
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
    assert report["verdict"]["status"] == "pass"
    assert report["complete"] is True
    # and without it, the same claims do NOT earn a pass
    assert pec.check_all(claims, Path("."))["verdict"]["status"] == "withheld"


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
    assert report["verdict"]["status"] == "withheld"
    assert report["complete"] is False
    # and the state is named rather than inferred from an absence, with a route out
    assert "never run" in report["verdict"]["withheld_reason"]
    assert "--paper" in report["verdict"]["remedy"]


def test_coverage_actually_run_is_reported_as_run() -> None:
    # Given: the same claims WITH a manuscript.
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    # When/Then: coverage runs, and the verdict may then say so.
    report = pec.check_all(claims, Path("."), paper_text="the ledger holds 11{,}567")
    assert report["coverage_checked"] is True
    assert report["verdict"]["status"] == "withheld"   # 11,567 is not catalogued here
    assert "11,567" in report["uncatalogued"]


def test_the_three_statuses_are_the_product_of_two_axes() -> None:
    """`withheld` is one state with many reasons, not many states.

    "Did the check run?" and "if it ran, what did it find?" are orthogonal, so
    every verdict is a point in their product: ran+clean, ran+problem,
    not-run+why. Naming each not-run reason as its own state duplicates the axis
    and loses it, and it grows one exit per failure discovered. The reason is
    DATA; the state is `withheld`.
    """
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    # Given: two different reasons the check cannot run.
    no_paper = pec.check_all(claims, Path("."))
    uncatalogued = pec.check_all(claims, Path("."), paper_text="the value is 4271")
    # When/Then: the same STATUS, different reasons -- which is the point.
    assert no_paper["verdict"]["status"] == uncatalogued["verdict"]["status"] == "withheld"
    assert no_paper["verdict"]["withheld_reason"] != uncatalogued["verdict"]["withheld_reason"]
    # and each carries a remedy, so neither is a dead end
    for report in (no_paper, uncatalogued):
        assert report["verdict"]["remedy"]


def test_a_withheld_verdict_is_never_counted_as_a_pass() -> None:
    """The failure this whole line of work started from."""
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    for report in (pec.check_all(claims, Path(".")),
                   pec.check_all(claims, Path("."), paper_text="the value is 4271")):
        assert report["verdict"]["status"] != "pass"
        assert report["complete"] is False


# --------------------------------------------------------------------------
# a finding must name WHY, not merely that
# --------------------------------------------------------------------------


def _finding(claim: pec.Claim, root: Path) -> dict:
    return pec.check_all([claim], root, paper_text="")["verdict"]


def test_a_reader_can_tell_which_cause_from_the_output_alone(tmp_path: Path) -> None:
    """The test their note implies: not "did it block", but "can you tell why".

    A `finding` has two causes needing different responses -- a value absent from
    an artifact that exists (the claim or the artifact is wrong) versus an
    artifact the repository does not ship (the paper depends on something missing).
    Reporting both as a bare `finding` forces the reader to go and read the
    per-claim list, and the withhold reads complete once it exits non-zero.
    """
    # Given: one claim of each kind.
    (tmp_path / "present.json").write_text('{"x": 1}', encoding="utf-8")
    absent_value = pec.Claim("v", "99999", "present.json", "measured", near=r'"x":\s*[0-9]+')
    absent_file = pec.Claim("f", "1", "absent.json", "measured", near=r'"x":\s*[0-9]+')
    # When: each is checked.
    value_verdict = _finding(absent_value, tmp_path)
    file_verdict = _finding(absent_file, tmp_path)
    # Then: the STATUS is the same...
    assert value_verdict["status"] == file_verdict["status"] == "finding"
    # ...and the CAUSE differs, from the output alone, without consulting counts.
    assert value_verdict["finding_cause"][0]["cause"] == "value_not_found"
    assert file_verdict["finding_cause"][0]["cause"] == "path_missing"
    assert value_verdict["finding_cause"] != file_verdict["finding_cause"]
    # and each names the claim it came from and what it means
    assert value_verdict["finding_cause"][0]["claims"] == ["v"]
    assert "does not carry the cited value" in value_verdict["finding_cause"][0]["description"]
    assert "not present in this tree" in file_verdict["finding_cause"][0]["description"]


def test_both_causes_are_listed_when_both_occur(tmp_path: Path) -> None:
    """A finding that reports one cause when there are two under-describes it."""
    (tmp_path / "present.json").write_text('{"x": 1}', encoding="utf-8")
    report = pec.check_all(
        [pec.Claim("v", "99999", "present.json", "measured", near=r'"x":\s*[0-9]+'),
         pec.Claim("f", "1", "absent.json", "measured", near=r'"x":\s*[0-9]+')],
        tmp_path, paper_text="")
    causes = {c["cause"] for c in report["verdict"]["finding_cause"]}
    assert causes == {"value_not_found", "path_missing"}


def test_a_non_finding_carries_no_cause(tmp_path: Path) -> None:
    """The cause field exists only where there is a cause.

    A pass or a withhold with a stale `finding_cause` would be the
    three-states-in-one-field problem again, one level down.
    """
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    for report in (pec.check_all(claims, tmp_path, paper_text=""),   # pass
                   pec.check_all(claims, tmp_path)):                  # withheld
        assert report["verdict"]["status"] != "finding"
        assert report["verdict"]["finding_cause"] is None


def test_the_two_axes_hold_for_findings_too() -> None:
    """`finding` is one state with many causes, exactly as `withheld` is.

    Enumerating causes as states is the same mistake in the other half of the
    product: it would grow one exit per cause discovered, which is how a single
    bool became four exits on the peer's gate.
    """
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    report = pec.check_all(claims, Path("."), paper_text="")
    # the vocabulary is closed and small; the causes live in a field
    assert report["verdict"]["status"] in {"pass", "finding", "withheld"}
    assert set(report["verdict"]) == {"status", "withheld_reason", "remedy", "finding_cause"}


# --------------------------------------------------------------------------
# the diagnosis must name the PROBABLE failure, not the one that prompted it
# --------------------------------------------------------------------------


def test_a_value_outside_its_declared_context_is_a_different_finding(tmp_path: Path) -> None:
    r"""A `near` pattern narrower than the artifact's formatting is the LIKELY defect.

    It produces exactly the same symptom as a genuinely absent value, and reporting
    both as "the artifact does not carry it" names the improbable case and sends the
    reader to inspect the artifact -- which is the one thing that is fine. Every real
    case in this file was this one: the LaTeX thousands separator, the 886.7 rounding,
    the `problem`-versus-`question` field. A diagnosis tested only on the case that
    prompted it has encoded the example, not the mechanism.
    """
    # Given: an artifact that carries the value, and a pattern too narrow to reach it.
    _source(tmp_path, "s.json", '{"arm": "A1", "tokens_per_second": 799.1303}')
    narrow = pec.Claim("t", "799.1", "s.json", "measured",
                       near=r'"arm":\s*"A1"[\s\S]{0,1}?"tokens_per_second":\s*[0-9.]+')
    # When: it is checked.
    result = pec.check_claim(narrow, tmp_path)
    # Then: the finding names the PATTERN as the suspect, not the artifact.
    assert result.verdict == "value_outside_context"
    assert "IS present" in result.detail
    assert "the pattern is too narrow" in result.detail
    assert "the artifact itself is not the suspect" in result.detail
    # and it is a distinct finding from a genuinely absent value
    absent = pec.Claim("t", "987654321", "s.json", "measured", near=r'"arm"')
    assert pec.check_claim(absent, tmp_path).verdict == "value_not_found"


def test_the_two_artifact_causes_are_distinguishable_in_the_report(tmp_path: Path) -> None:
    _source(tmp_path, "s.json", '{"arm": "A1", "tokens_per_second": 799.1303}')
    report = pec.check_all([
        pec.Claim("outside", "799.1", "s.json", "measured", near=r'"arm":\s*"A1"[\s\S]{0,1}?x'),
        pec.Claim("absent", "987654321", "s.json", "measured", near=r'"arm"'),
    ], tmp_path, paper_text="")
    causes = {c["cause"] for c in report["verdict"]["finding_cause"]}
    assert causes == {"value_outside_context", "value_not_found"}
    # each says what to do about it, and they differ
    by_cause = {c["cause"]: c["description"] for c in report["verdict"]["finding_cause"]}
    # the two descriptions must point at different suspects
    assert "pattern is the suspect" in by_cause["value_outside_context"]
    assert "does not carry" in by_cause["value_not_found"]
    assert by_cause["value_outside_context"] != by_cause["value_not_found"]


def test_the_real_inventory_is_unaffected_by_the_new_cause() -> None:
    """The distinction must not reclassify the shipped inventory's clean claims."""
    repo = Path(__file__).resolve().parents[3]
    inventory = repo / "DAN" / "nonlatent_iclr" / "paper_claims.json"
    if not inventory.is_file():
        pytest.skip("inventory absent")
    report = pec.check_all(pec.load_inventory(inventory), repo,
                           paper=repo / "paper" / "main.tex")
    assert report["counts"]["value_outside_context"] == 0
    assert report["counts"]["value_not_found"] == 0


# --------------------------------------------------------------------------
# a skip is a claim
# --------------------------------------------------------------------------


def test_an_extractor_that_matched_nothing_cannot_report_a_pass() -> None:
    r"""The extractor's own skip is a claim, and it is the one nobody audits.

    If the number pattern silently matches nothing -- a broken regex, a changed
    document class, an encoding change -- then every downstream count is zero,
    `uncatalogued` is empty, and the verdict reads `pass`. A confident finding
    produced by a filter that skipped everything. The denominator is the thing
    that shows it, and the suspect is the EXTRACTOR, not the paper.
    """
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    manuscript = "the ledger holds 11{,}567 GPU-hours and 3{,}600 units"
    # Given: the extractor working, then silently matching nothing.
    good = pec.check_all(claims, Path("."), paper_text=manuscript)
    assert good["numbers_extracted"] == 2
    assert good["verdict"]["status"] == "withheld"     # 11,567 is not catalogued
    broken = pec.check_all(claims, Path("."), paper_text=manuscript,
                           extractor=lambda _t: [])
    # When/Then: the broken extractor does NOT pass, and names itself as the suspect.
    assert broken["verdict"]["status"] == "withheld"
    assert broken["numbers_extracted"] == 0
    # the suspect is named in the REASON and the route out in the REMEDY
    assert "the suspect is the extraction pattern" in broken["verdict"]["withheld_reason"]
    assert "vacuous" in broken["verdict"]["withheld_reason"]
    assert "paper_numbers()" in broken["verdict"]["remedy"]


def test_the_extracted_count_is_always_reported() -> None:
    """A count with no denominator is how a skip hides."""
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    for text in ("", "no numbers here at all", "the value is 4271"):
        report = pec.check_all(claims, Path("."), paper_text=text)
        assert "numbers_extracted" in report
        assert "manuscript_chars" in report
        assert report["manuscript_chars"] == len(text)


def test_an_empty_manuscript_is_not_treated_as_a_broken_extractor() -> None:
    """Zero numbers from a zero-length manuscript is a vacuous input, not a bug.

    The distinction matters: withholding on an empty input would be another
    over-block, and the over-block's failure mode is a correct result destroyed
    by a gate that could not tell the two apart.
    """
    claims = [pec.Claim("e", "1", "/elsewhere", "external", appears_in="stated")]
    report = pec.check_all(claims, Path("."), paper_text="")
    assert report["numbers_extracted"] == 0
    assert report["verdict"]["status"] == "pass"       # nothing to cover, and that is fine
