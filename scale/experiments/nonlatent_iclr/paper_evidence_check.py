"""Check every cited number in the paper against the artifact that should carry it.

The paper claims its correspondence with its evidence is "machine-verifiable
rather than asserted".  That claim was itself only asserted: nobody had run a
check.  This does, deterministically, with no model in the loop.

The method is the one ARIS's ``result-to-claim`` Step 1.5 prescribes -- enumerate
each cited value as ``{id, value, source}`` and require the literal string to
appear in the source file -- and the reason it is worth mechanising is that it
needs no judgement: a number is either in the artifact or it is not.

Four verdicts, and the distinction between the last two is the point:

``verified``        the literal value occurs in the named file
``value_not_found`` the file exists and does not contain the value
``path_missing``    the named file does not exist here
``external``        the value's only source is outside the repository

``value_not_found`` is the dangerous one: it is what a number looks like when it
was transcribed from somewhere the shipped artifact does not reach.  ``external``
is not a failure, but it IS a disclosure obligation -- a number a reader cannot
re-derive from the published tree has to be labelled as such, which the paper
already does for its per-example files and did not do for its headline readings
until this check found them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

#: A number the manuscript cites that is NOT a claim: a citation year, an arXiv
#: identifier, a section or table reference.
CITATION_PATTERNS: Final = (
    r"\d{4}\.\d{4,5}",      # arXiv ids: 2511.15927
    r"\b(?:19|20)\d{2}\b",   # years
)

#: Below this magnitude a number is almost always structural (a layer count, a
#: percentile, a threshold) rather than a measurement, and cataloguing all of
#: them would drown the check in noise.  The limit is stated rather than implied
#: because it IS a limit: a substantive sub-100 number would slip past it.
MIN_SUBSTANTIVE: Final = 100

SCHEMA: Final = "nonlatent_paper_claims_v1"
REPORT_SCHEMA: Final = "nonlatent_paper_evidence_report_v1"

#: Verdicts, ordered by how much attention they deserve.
VERDICTS: Final = ("value_not_found", "path_missing", "external", "derived",
                   "rounding_ok", "verified")


class EvidenceCheckRefusal(ValueError):
    """The inventory itself is malformed."""


@dataclass(frozen=True, slots=True)
class Claim:
    """One cited number and where it should be checkable.

    ``near`` is a regex the value must appear WITHIN.  Its window has to be
    WIDE enough to reach the value across the artifact's own formatting -- the
    ledger puts 807 characters between an arm's name and its throughput -- and
    NARROW enough not to cross into the next arm's block, which is why the
    anchor pattern uses a negative lookahead rather than a character count alone.  It exists because a bare
    substring search produces false ``verified`` verdicts: measured on the real
    tree, ``56.60`` occurs inside ``length_matrix.json`` as part of an unrelated
    token count, and short numbers match inside hashes.  A check that reports a
    false pass is worse than no check, so a claim that cannot name the context it
    should appear in must be declared ``external`` instead of merely grepped.
    """

    claim_id: str
    value: str
    source: str
    kind: str          # measured | derived | registered | external
    appears_in: str = ""
    near: str = ""


@dataclass(frozen=True, slots=True)
class Result:
    claim_id: str
    value: str
    source: str
    kind: str
    verdict: str
    detail: str


def load_inventory(path: Path) -> list[Claim]:
    """Read the inventory, refusing one that is missing fields or duplicates ids."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise EvidenceCheckRefusal(
            f"inventory schema {document.get('schema')!r} is not {SCHEMA!r}")
    raw = document.get("claims")
    if not isinstance(raw, list) or not raw:
        raise EvidenceCheckRefusal("inventory carries no claims")
    claims: list[Claim] = []
    seen: set[str] = set()
    for entry in raw:
        for field in ("claim_id", "value", "source", "kind"):
            if not entry.get(field):
                raise EvidenceCheckRefusal(f"claim {entry} is missing {field!r}")
        if entry["claim_id"] in seen:
            raise EvidenceCheckRefusal(f"duplicate claim_id {entry['claim_id']!r}")
        seen.add(entry["claim_id"])
        claims.append(Claim(claim_id=entry["claim_id"], value=str(entry["value"]),
                            source=entry["source"], kind=entry["kind"],
                            appears_in=entry.get("appears_in", ""),
                            near=entry.get("near", "")))
    return claims


def _decimals(value: str) -> int:
    """Decimal places the claimed value is stated to."""
    return len(value.split(".", 1)[1]) if "." in value else 0


def check_claim(claim: Claim, root: Path) -> Result:
    """One claim, one verdict.

    ``external`` is taken from the inventory rather than discovered, because
    "the source is a cluster path this repo does not ship" is a fact about the
    program that only a human knows; the checker's job is to refuse to call it
    verified when nothing here carries it.
    """
    if claim.kind == "external":
        return Result(claim.claim_id, claim.value, claim.source, claim.kind, "external",
                      claim.appears_in or "no in-repository source declared")
    if claim.kind == "derived":
        # A sum, a ratio, a rounded aggregate: arithmetic over artifacted inputs
        # rather than a value any file states.  It cannot be grepped, and
        # reporting it as missing evidence would be wrong -- but it IS a
        # disclosure, because a reader cannot check it without redoing the sum.
        return Result(claim.claim_id, claim.value, claim.source, claim.kind, "derived",
                      claim.appears_in or "arithmetic over artifacted inputs")
    if not claim.near:
        return Result(claim.claim_id, claim.value, claim.source, claim.kind,
                      "value_not_found",
                      "no `near` context declared, so a bare match could not be "
                      "distinguished from a coincidental substring")
    path = Path(root) / claim.source
    if not path.is_file():
        return Result(claim.claim_id, claim.value, claim.source, claim.kind,
                      "path_missing", f"{claim.source} is not present")
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(claim.near, re.IGNORECASE)
    numbers = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
    for match in pattern.finditer(text):
        found = match.group(0)
        if claim.value in found:
            return Result(claim.claim_id, claim.value, claim.source, claim.kind,
                          "verified",
                          f"{claim.value!r} occurs inside {found[:60]!r} in {claim.source}")
        # A transcribed value is often the ROUNDED form of the artifact's, which
        # is correct and must not read as missing evidence: the ledger holds
        # 886.659... and the paper writes 886.7.  Comparing numerically at the
        # claimed precision separates "rounded correctly" from "wrong number",
        # and a check that blurred those two would train a reader to ignore it.
        for candidate in numbers.findall(found):
            try:
                if round(float(candidate), _decimals(claim.value)) == float(claim.value):
                    return Result(claim.claim_id, claim.value, claim.source, claim.kind,
                                  "rounding_ok",
                                  f"{claim.value!r} is {candidate!r} rounded to "
                                  f"{_decimals(claim.value)} dp in {claim.source}")
            except ValueError:
                continue
    return Result(claim.claim_id, claim.value, claim.source, claim.kind,
                  "value_not_found",
                  f"no occurrence of {claim.value!r} inside /{claim.near}/ in "
                  f"{claim.source}, and no number there rounds to it")


def paper_numbers(paper_text: str) -> list[str]:
    """Substantive numeric tokens in the manuscript, in order of first appearance."""
    body = "\n".join(line.split("%")[0] for line in paper_text.splitlines()
                     if not line.lstrip().startswith("%"))
    # LaTeX writes thousands separators as \\textbf{11{,}567} or 3{,}600, and a naive
    # digit run splits those into "11" and "567" -- which then read as uncatalogued
    # numbers that do not exist. Normalise the separator BEFORE extracting, or the
    # coverage check reports a page of phantom gaps and trains a reader to ignore it.
    body = body.replace("{,}", ",")
    body = re.sub(r"\\[{}%]", "", body)
    for pattern in CITATION_PATTERNS:
        body = re.sub(pattern, " ", body)
    found: list[str] = []
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", body):
        plain = token.replace(",", "")
        try:
            value = float(plain)
        except ValueError:
            continue
        if value < MIN_SUBSTANTIVE:
            continue
        if token not in found:
            found.append(token)
    return found


def declared_non_claims(inventory: dict) -> dict[str, str]:
    """Numbers the inventory declares structural rather than evidential.

    A protocol axis, a seed list, a threshold: these are the paper's own
    declarations, so the artifact that "carries" them is the code that declares
    them, and requiring a measurement source for each would be a category error.
    They are listed with a REASON rather than simply exempted, so the exemption
    is reviewable.
    """
    return {str(entry["value"]): str(entry["reason"])
            for entry in inventory.get("non_claims", [])}


def uncatalogued_numbers(paper_text: str, claims: list[Claim],
                         non_claims: dict[str, str] | None = None) -> list[str]:
    """Numbers the manuscript cites that NO claim covers.

    This exists because the completeness verdict was about the wrong thing. The
    check reported ``complete: True`` while the manuscript cited a step number
    ("stopping near step 20,600") that was in no claim at all -- so the verdict
    meant "every catalogued claim verifies", not "every number in the paper was
    checked". Those are different statements, and only the second one is worth
    making.

    What this does NOT do: decide whether an uncatalogued number is a CLAIM.
    It reports what is not covered, and the judgement that a number is
    structural rather than evidential stays with the author.
    """
    catalogued = {claim.value.replace(",", "") for claim in claims}
    catalogued |= {claim.value.replace(",", "").rstrip("0").rstrip(".")
                   for claim in claims if "." in claim.value}
    catalogued |= {value.replace(",", "") for value in (non_claims or {})}
    # A paper number that is a SUBSTRING of a catalogued one is not a gap: the
    # manuscript writes the load axis as a set, so "\{1,8,32,128\}" yields "128"
    # on its own, and "s8000" yields "800". Reporting those would bury the real
    # gaps under fragments of numbers that are already accounted for.
    return [token for token in paper_numbers(paper_text)
            if not any(token.replace(",", "") in known or known in token.replace(",", "")
                       for known in catalogued)]


def check_all(claims: list[Claim], root: Path, paper: Path | None = None,
              inventory: dict | None = None) -> dict[str, object]:
    """Every claim, grouped by verdict, worst first."""
    results = [check_claim(claim, root) for claim in claims]
    uncatalogued: list[str] = []
    if paper is not None and Path(paper).is_file():
        uncatalogued = uncatalogued_numbers(Path(paper).read_text(encoding="utf-8"),
                                            claims, declared_non_claims(inventory))
    by_verdict = {verdict: [r for r in results if r.verdict == verdict]
                  for verdict in VERDICTS}
    return {
        "schema": REPORT_SCHEMA,
        "disclosed": sorted({r.claim_id for r in
                             by_verdict["external"] + by_verdict["derived"]}),
        "checked": len(results),
        "counts": {verdict: len(items) for verdict, items in by_verdict.items()},
        "results": [asdict(r) for r in results],
        "complete": (not by_verdict["value_not_found"] and not by_verdict["path_missing"]
                     and not uncatalogued),
        "uncatalogued": uncatalogued,
        "uncatalogued_count": len(uncatalogued),
        "disclosed_count": len({r.claim_id for r in
                                by_verdict["external"] + by_verdict["derived"]}),
        "note": ("a check that cannot run is reported as such rather than as a pass; "
                 "`external` is a disclosure, not a failure; and `complete` is False "
                 "while any number the manuscript cites is in no claim, because "
                 "otherwise it would report coverage of the inventory rather than of "
                 "the paper"),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--paper", type=Path,
                        help="the manuscript, so numbers no claim covers are visible")
    args = parser.parse_args(argv)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    report = check_all(load_inventory(args.inventory), args.root, paper=args.paper,
                       inventory=inventory)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], indent=1))
    for result in report["results"]:
        if result["verdict"] in ("value_not_found", "path_missing", "external"):
            print(f"  [{result['verdict']}] {result['claim_id']}: {result['detail']}")
    if args.paper:
        print(f"uncatalogued paper numbers: {report['uncatalogued_count']}")
        for token in report["uncatalogued"]:
            print(f"  [uncatalogued] {token}")
    print(f"complete (no missing evidence): {report['complete']}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
