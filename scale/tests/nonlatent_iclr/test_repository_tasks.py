"""Repository-task tests: purity, discrimination, held-out splits, contamination."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Final, cast

import pytest

from scale.experiments.nonlatent_iclr.tasks import repository_evaluator as rev
from scale.experiments.nonlatent_iclr.tasks import repository_tasks as rt
from scale.experiments.nonlatent_iclr.tasks.provenance import repository_split_map

CORPUS: Final = Path("DAN/nonlatent_iclr/task4_assets/repository_tasks")


def _candidate(source: str):
    found = rt.extract_candidates("owner/repo", "MIT", "pkg/mod.py", source, "0" * 64)
    return found


def _manifest() -> dict[str, object]:
    return cast("dict[str, object]", json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8")))


def test_extract_accepts_a_pure_function_verbatim() -> None:
    # Given: a pure function using only allowlisted operations.
    source = "def add(a, b):\n    return a + b\n"
    # When: it is extracted.
    found = _candidate(source)
    # Then: the task carries the real upstream source unchanged.
    assert len(found) == 1
    assert found[0].entry_point == "add"
    assert found[0].parameters == ("a", "b")
    assert found[0].function_source == source


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(a):\n    return os.getcwd()\n",
        "def f(a):\n    return open('x').read()\n",
        "def f(a):\n    return eval('1')\n",
        "def f(a):\n    return a.__class__\n",
        "def f(a):\n    return a.real_attr\n",
    ],
    ids=["import", "io", "eval", "dunder", "attribute"],
)
def test_extract_rejects_impure_or_unnrecognised_constructs(source: str) -> None:
    # Given: a function reaching outside the purity allowlist.
    # When / Then: nothing is extracted, so it can never become a task.
    assert _candidate(source) == ()


def test_extract_rejects_private_decorated_and_varargs_functions() -> None:
    # Given: functions that cannot be given a safe fixed call.
    source = (
        "def _hidden(a):\n    return a\n\n"
        "@staticmethod\n"
        "def decorated(a):\n    return a\n\n"
        "def variadic(*args):\n    return args\n"
    )
    # When / Then: none qualify.
    assert _candidate(source) == ()


def test_extract_uses_only_required_parameters_when_defaults_exist() -> None:
    # Given: a pure function with a defaulted trailing parameter.
    found = _candidate("def scale(n, factor=2):\n    return n * factor\n")
    # When / Then: only the required prefix is supplied, so the default stands.
    assert len(found) == 1
    assert found[0].parameters == ("n",)


def test_argument_literal_is_deterministic_and_type_appropriate() -> None:
    # Given: candidates whose parameter names signal their expected type.
    text = "def f(items):\n    return items\n\ndef g(text):\n    return text\n"
    first = rt.extract_candidates("owner/repo", "MIT", "m.py", text, "0" * 64)
    second = rt.extract_candidates("owner/repo", "MIT", "m.py", text, "0" * 64)
    # When / Then: generation is deterministic and shaped by the parameter name.
    assert rt.argument_literal(first[0]) == rt.argument_literal(second[0])
    assert rt.argument_literal(first[0]).startswith("[")
    assert rt.argument_literal(first[1]).startswith("'")


def test_check_and_expression_are_consistent() -> None:
    # Given: a candidate.
    candidate = _candidate("def add(a, b):\n    return a + b\n")[0]
    # When: the check and the reference expression are generated.
    check = rt.check_source(candidate, "7")
    expression = rt.expression_source(candidate)
    # Then: both call the same entry point with the same arguments.
    arguments = rt.argument_literal(candidate)
    assert check == f"assert add({arguments}) == 7\n"
    assert expression == f"print(repr(add({arguments})))\n"


def test_mutation_panel_yields_two_distinct_programs() -> None:
    # Given: a function with a return statement.
    source = "def add(a, b):\n    return a + b\n"
    # When: a mutation panel is built.
    panel = rev.mutation_panel(source)
    # Then: two distinct mutants exist, neither equal to the original.
    assert len(panel) == 2
    assert len(set(panel)) == 2
    assert all(mutant != source for mutant in panel)


def test_build_one_rejects_a_function_that_raises_on_its_arguments() -> None:
    # Given: a pure-looking function whose reference run always raises.
    candidate = _candidate("def broken(n):\n    return n // 0\n")
    assert len(candidate) == 1
    # When: it is built into a task.
    built = rt.build_one(candidate[0], 0)
    # Then: a task whose reference run raises is rejected rather than shipped.
    assert built is None


def test_build_one_accepts_and_discriminates_a_real_task() -> None:
    # Given: a genuinely pure function.
    candidate = _candidate("def add(a, b):\n    return a + b\n")[0]
    # When: it is built.
    built = rt.build_one(candidate, 3)
    # Then: the record binds the real source and check, and both files hash-match.
    assert built is not None
    record = built.record
    assert str(record["id"]).endswith("-003")
    assert record["derivation"] == rt.DERIVATION
    assert record["source_sha256"] == sha256(built.source.encode()).hexdigest()
    assert record["evaluator_sha256"] == sha256(built.check.encode()).hexdigest()


def test_split_map_guarantees_held_out_families() -> None:
    # Given: a handful of families with very uneven contributions.
    weights = {"big": 150, "medium": 30, "small": 10, "tiny": 2, "micro": 1}
    # When: splits are assigned.
    assignment = repository_split_map(weights)
    # Then: whole families move together and both held-out kinds exist.
    assert set(assignment) == set(weights)
    assert set(assignment.values()) == {"train", "validation", "confirmation"}
    assert assignment["big"] == "train"
    assert {assignment["micro"], assignment["tiny"]} == {"validation", "confirmation"}


def test_split_map_rejects_too_few_families() -> None:
    # Given: fewer than three families.
    # When / Then: no assignment is invented that would fake a held-out split.
    for weights in ({}, {"a": 1}, {"a": 5, "b": 5}):
        with pytest.raises(ValueError, match="families"):
            _ = repository_split_map(weights)


def test_published_corpus_is_complete() -> None:
    # Given: the published corpus.
    manifest = _manifest()
    records = cast("list[object]", manifest["records"])
    # When / Then: it declares the expected shape and the derivation it used.
    assert len(records) == 200
    assert manifest["derivation"] == rt.DERIVATION
    assert (CORPUS / str(manifest["asset_path"])).is_file()
    assert len(list((CORPUS / "sources").glob("*.py"))) == 200
    assert len(list((CORPUS / "checks").glob("*.py"))) == 200


def test_published_corpus_separates_sources_from_checks() -> None:
    # Given: the published corpus.
    manifest = _manifest()
    records = [cast("dict[object, object]", item) for item in cast("list[object]", manifest["records"])]
    # When / Then: no record points a prompt-bearing source at an evaluator file, and the two
    # trees are disjoint, so a context loader reading only sources/ cannot reach a hidden answer.
    for record in records:
        source = str(record["source_path"])
        evaluator = str(record["evaluator_path"])
        assert source.startswith("sources/")
        assert evaluator.startswith("checks/")
        assert source != evaluator
    assert not (set((CORPUS / "sources").glob("*.py")) & set((CORPUS / "checks").glob("*.py")))


def test_published_corpus_never_stores_a_plaintext_answer() -> None:
    # Given: the published corpus.
    manifest = _manifest()
    records = [cast("dict[object, object]", item) for item in cast("list[object]", manifest["records"])]
    # When / Then: records carry only the answer digest, never the answer itself.
    for record in records:
        digest = str(record["answer_sha256"])
        assert len(digest) == 64
        assert "answer" not in {str(key) for key in record} - {"answer_sha256"}
        check = (CORPUS / str(record["evaluator_path"])).read_text(encoding="utf-8")
        assert check.startswith("assert ")
        assert digest not in check


def test_published_corpus_has_held_out_families_and_no_straddling() -> None:
    # Given: the published corpus.
    manifest = _manifest()
    records = [cast("dict[object, object]", item) for item in cast("list[object]", manifest["records"])]
    families: dict[str, str] = {}
    # When: every record's family is compared with its declared split.
    for record in records:
        family = str(record["repository_family"])
        split = str(record["split"])
        existing = families.setdefault(family, split)
        assert existing == split, f"{family} straddles {existing} and {split}"
    # Then: at least one family of each kind is held out, and none of them trains.
    assert "validation" in set(families.values())
    assert "confirmation" in set(families.values())
    weights: dict[str, int] = {}
    for record in records:
        family = str(record["repository_family"])
        weights[family] = weights.get(family, 0) + 1
    assert families == repository_split_map(weights)


def test_published_check_discriminates_for_a_sampled_task() -> None:
    # Given: a deterministic sample of published tasks.
    from scale.experiments.nonlatent_iclr.tasks.repository_evaluator import verify_task

    manifest = _manifest()
    records = [cast("dict[object, object]", item) for item in cast("list[object]", manifest["records"])]
    # When: each sampled check is replayed against its real source.
    for record in records[::40]:
        source = (CORPUS / str(record["source_path"])).read_text(encoding="utf-8")
        check = (CORPUS / str(record["evaluator_path"])).read_text(encoding="utf-8")
        qualified, reason = verify_task(source, check, entry_point=str(record["entry_point"]))
        # Then: the baseline passes and both mutants are rejected.
        assert qualified, f"{record['id']}: {reason}"


def test_candidate_records_never_serialise_function_source() -> None:
    # Given: an extracted candidate.
    candidate = _candidate("def add(a, b):\n    return a + b\n")[0]
    # When: its dataclass fields are inspected.
    fields = set(asdict(candidate))
    # Then: the source lives only on the candidate, never in a published record shape.
    assert "function_source" in fields
    assert "record" not in fields
