"""The long-context pipeline, end to end on CPU, with the REAL tokenizer.

Why this file exists
--------------------
``longcontext/runner.py`` has tests, and the token-targeted fixtures have tests,
but the runner's tests inject a ``FakeEncoder`` and a stub generator, and the
fixture tests never call the runner.  Nothing had run the real fixtures, the real
tokenizer and the real scoring path through one another, so every seam between
"a cell is declared" and "a score comes out" was unverified even though each part
was verified alone.  This file runs the whole chain:

    CellCoordinates -> generate_token_targeted_task (REAL tokenizer)
        -> the runner, in BOTH access modes -> score_cell -> macro_score

The model is a stub, and the stub is the instrument rather than the thing under
test: it answers, and it RECORDS the ids it was shown, so the prompt length, the
chunk coverage and the access-mode claim are all read off what the model actually
saw rather than inferred from a score.

The grid is TINY on purpose
---------------------------
Two cells, one instance each by default, both at the shortest registered bounded
length (16,384 tokens); two probes raise the instance count to three, one to
exercise the parseable-only accuracy denominator and one to give multi-instance
streaming its own access log per instance.  This is a PIPELINE test, not a
measurement: a token count or an accuracy here is a plumbing fact, and the macro
score is asserted to be exactly what the stub emitted, never a claim about a
model.  One further probe widens the cut to the four loads of a single
(family, length) only to record a runner-aggregation defect; it measures nothing
either.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Final, Sequence

import pytest

from scale.experiments.nonlatent_iclr.longcontext import runner as rn
from scale.experiments.nonlatent_iclr.tasks import token_targeted_matrix as tm
from scale.experiments.nonlatent_iclr.tasks.token_targeted_tasks import (
    PAD_TEXT,
    QUERY_LINE,
    TokenTargetedRequest,
    TokenTargetedTask,
    generate_token_targeted_task,
)
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import (
    VOCAB_NAME,
    VOCAB_SHA256,
    load_safe_tokenizer,
)

#: The pinned, hash-bound vocabulary.  ``load_safe_tokenizer`` verifies the sha256
#: before parsing, so a stale or swapped file refuses rather than silently
#: redefining the length axis.
VOCAB_PATH: Final = (
    Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/models/rwkv7-0.4B")
    / VOCAB_NAME
)

#: Two cells, both at ``BOUNDED_LENGTHS[0]`` = 16,384 -- the shortest registered
#: bounded length.  The bounded cut orders family slowest, so unit 0 is
#: ``associative_recall`` and unit 12 is ``overwrite_delayed_query``: two families,
#: which is the least that exercises the stubs' family-aware paths.
UNITS: Final = (0, 12)

#: One instance per cell.  Enough to run the whole chain; more is measurement.
INSTANCES: Final = 1

#: The evidence framing the token fixture copies from the character fixture.  The
#: stub recovers the evidence from the DECODED prompt -- never from a regenerated
#: task -- so it can answer only what the model was actually shown.
_EVIDENCE_OPEN: Final = "BEGIN EVIDENCE\n"
_EVIDENCE_CLOSE: Final = "\nEND EVIDENCE"

#: Exact-task evidence carries its distractor after this marker.  ``public_evidence``
#: strips it before the evaluator parses, so a solver reading the same prompt must
#: strip it too or it would parse the distractor instead of the query.
_DISTRACT_MARKER: Final = ";DISTRACT "

#: The keyword each bounded family's evidence opens with.  Used to dispatch the
#: solver off the prompt alone, and to refuse a prompt the solver cannot read.
_FAMILY_KEYWORD: Final = {
    "associative_recall": "ASSOC ",
    "overwrite_delayed_query": "OVERWRITE ",
    "code_dataflow": "FLOW ",
}

#: A well-formed, PARSEABLE answer per bounded family, none of which can ever be a
#: fixture's gold: recall/overwrite gold is ``v0``..``v9999`` (``randrange(10_000)``)
#: and dataflow gold is a small integer, so a completion built from this table is
#: wrong-but-parseable -- the case that must read as an incorrect answer rather
#: than as a missing one.
_FIXED_ANSWER: Final = {
    "associative_recall": "v99999",
    "overwrite_delayed_query": "v99999",
    "code_dataflow": "999999999",
}


class PipelineTestRefusal(ValueError):
    """The test's own instrument could not read a fixture or prompt as built.

    Raised rather than returning a guess.  A solver that fell back to, say, the
    pad character would score the padding instead of the evidence and turn a
    prompt-layout defect into a plausible-looking accuracy.
    """


class RecordingModel:
    """A ``generate()`` seam that answers AND records every prompt it was shown.

    The runner hands the model exactly the ids it is allowed to read, so the
    recorded list IS the model's whole view of the task.  A prompt shorter than
    the cell declares, or a chunked prompt that dropped a token, is visible here
    instead of being inferred from a downstream score.
    """

    def __init__(self, answer: Callable[[tuple[int, ...]], str]) -> None:
        self._answer = answer
        self.shown: list[tuple[int, ...]] = []

    def __call__(self, prompt_ids: Sequence[int], *, max_new_tokens: int) -> str:
        shown = tuple(int(token) for token in prompt_ids)
        self.shown.append(shown)
        return self._answer(shown)


def _task(tokenizer, unit: int, instance_index: int) -> TokenTargetedTask:
    """The token-exact fixture for a bounded-cut unit, built with the real tokenizer."""
    coordinates = tm.cell_coordinates(unit)
    request = TokenTargetedRequest(
        family=coordinates.family, data_seed=coordinates.data_seed,
        token_length=coordinates.token_length,
        position_fraction=tm.BOUNDED_POSITION, load=coordinates.load,
        distractor=tm.BOUNDED_DISTRACTOR, instance_index=instance_index)
    return generate_token_targeted_task(tokenizer, request)


def _gold_oracle(tokenizer, units: Sequence[int],
                 instances: int) -> dict[tuple[int, ...], str]:
    """The fixtures' own gold, keyed by the exact prompt ids the model will see.

    Keying by the shown ids is deliberate: a runner that built a different prompt
    than the declared cell would miss the key and raise here, rather than scoring
    a wrong prompt as if it were the fixture.
    """
    table: dict[tuple[int, ...], str] = {}
    for unit in units:
        for index in range(instances):
            task = _task(tokenizer, unit, index)
            table[task.prompt_ids] = task.answer
    return table


def _evidence_text(prompt_text: str) -> str:
    """The evidence block of a decoded token prompt, distractor removed.

    The token fixture hits its exact length by repeating a single pad TOKEN, so the
    decoded prompt carries a run of ``PAD_TEXT`` between the opening framing and
    the evidence.  ``lstrip`` removes exactly that run, which is safe only because
    every bounded family's evidence opens with a non-``PAD_TEXT`` keyword -- the
    dispatch in :func:`_solve` refuses any prompt whose recovered evidence does not.
    """
    if (prompt_text.count(_EVIDENCE_OPEN) != 1
            or prompt_text.count(_EVIDENCE_CLOSE) != 1):
        raise PipelineTestRefusal("the decoded prompt has no single evidence block")
    body = prompt_text.partition(_EVIDENCE_OPEN)[2].partition(_EVIDENCE_CLOSE)[0]
    return body.lstrip(PAD_TEXT).split(_DISTRACT_MARKER, 1)[0]


def _solve_associative(body: str) -> str:
    records, query = body.removeprefix(_FAMILY_KEYWORD["associative_recall"]).split(";QUERY ")
    return dict(item.split("=") for item in records.split(";"))[query]


def _solve_overwrite(body: str) -> str:
    records, query = body.removeprefix(_FAMILY_KEYWORD["overwrite_delayed_query"]).split(";QUERY ")
    values: dict[str, str] = {}
    for item in records.split(";"):
        key, value = item.split("=")
        values[key] = value  # last write wins, mirroring the task's own rule
    return values[query]


def _solve_dataflow(body: str) -> str:
    statements, query = body.removeprefix(_FAMILY_KEYWORD["code_dataflow"]).split(";QUERY ")
    values: dict[str, int] = {}
    for statement in statements.split(";"):
        name, expression = statement.split("=")
        if "+" in expression:
            parent, increment = expression.split("+")
            values[name] = values[parent] + int(increment)
        else:
            values[name] = int(expression)
    return str(values[query])


_SOLVER: Final = {
    "associative_recall": _solve_associative,
    "overwrite_delayed_query": _solve_overwrite,
    "code_dataflow": _solve_dataflow,
}


def _solve(tokenizer, prompt_ids: tuple[int, ...]) -> str:
    """Recover the gold from the shown prompt alone, via the real tokenizer.

    This is the strong stub: it proves the prompt the model was shown carries the
    answer in the layout the task intends, so a scoring bug cannot hide behind a
    stub that was handed the gold out of band.
    """
    prompt_text = tokenizer.decode_ids(prompt_ids).decode("utf-8")
    evidence = _evidence_text(prompt_text)
    for family, keyword in _FAMILY_KEYWORD.items():
        if evidence.startswith(keyword):
            return _SOLVER[family](evidence)
    raise PipelineTestRefusal(
        f"no bounded family matches the decoded evidence: {evidence[:40]!r}")


@pytest.fixture(scope="module")
def tokenizer():
    if not VOCAB_PATH.is_file():
        pytest.skip(f"pinned vocabulary absent: {VOCAB_PATH}")
    # load_safe_tokenizer re-verifies the sha256, so an unexpected file refuses.
    return load_safe_tokenizer(VOCAB_PATH, VOCAB_SHA256)


# --------------------------------------------------------------------------
# assertion 1: the model is shown exactly the declared token length
# --------------------------------------------------------------------------


@pytest.mark.parametrize("access_mode", rn.ACCESS_MODES)
@pytest.mark.parametrize("unit", UNITS)
def test_the_model_is_shown_a_prompt_of_exactly_the_declared_token_length(
        tokenizer, unit: int, access_mode: str) -> None:
    # Given: a cell, and the same fixture built directly with the real tokenizer.
    coordinates = tm.cell_coordinates(unit)
    expected = _task(tokenizer, unit, 0)
    model = RecordingModel(lambda shown: _FIXED_ANSWER[coordinates.family])

    # When: the runner drives the cell in the given access mode.
    rn.run_cell(coordinates, encode=tokenizer, generate=model,
                access_mode=access_mode, instances=INSTANCES)

    # Then: every prompt the model saw is EXACTLY the real tokenizer's fixture --
    # same ids, hence the same declared length -- not a near miss in characters.
    assert expected.n_prompt_tokens == coordinates.token_length
    assert model.shown, "the runner must call the model at least once"
    for shown in model.shown:
        assert shown == expected.prompt_ids
        assert len(shown) == coordinates.token_length
        # And it ends at the query line, which is the real tokenizer's encoding of
        # it -- the answer slot the character fixture buries under filler.
        query_ids = tokenizer.encode_text(QUERY_LINE)
        assert shown[-len(query_ids):] == query_ids


def test_the_exact_length_is_at_the_id_level_not_the_character_level(tokenizer) -> None:
    """Why "exactly the declared token length" is an ID-level claim.

    The fixture pads with one repeated TOKEN, and the pinned byte-trie merges runs
    of the pad character, so decoding the prompt to text and re-encoding it yields
    FEWER tokens.  A test that checked the length by round-tripping text would fail
    a correct fixture, so the measurement is the count of ids the model is shown.
    """
    task = _task(tokenizer, 0, 0)
    decoded = tokenizer.decode_ids(task.prompt_ids).decode("utf-8")
    assert len(task.prompt_ids) == tm.BOUNDED_LENGTHS[0]
    assert len(tokenizer.encode_text(decoded)) < tm.BOUNDED_LENGTHS[0]


class _OneIdPerCharacterEncoder:
    """An encoder that is NOT the pinned vocabulary: one id per character.

    Exists so the recorded-prompt equality above can be shown to be a real
    constraint rather than a tautology.  ``_task`` and the runner both encode with
    the pinned tokenizer, so a reader could ask whether "the shown ids equal the
    fixture ids" is true by construction; this encoder answers that it is not --
    it hits the same declared LENGTH but a different ID SEQUENCE, which is exactly
    what the equality compares.  A runner that built a cell with the wrong encoder
    would therefore fail the assertion instead of passing it.
    """

    def encode_text(self, text: str) -> list[int]:
        return [(ord(character) % 30000) + 1000 for character in text]

    def decode_ids(self, ids) -> str:
        return "".join(chr((int(token) - 1000) % 30000) for token in ids)


def test_a_mis_encoding_encoder_cannot_reproduce_the_pinned_prompt(tokenizer) -> None:
    """Why the exact-length equality is a discriminating check, not a tautology."""
    coordinates = tm.cell_coordinates(UNITS[0])
    request = TokenTargetedRequest(
        family=coordinates.family, data_seed=coordinates.data_seed,
        token_length=coordinates.token_length, position_fraction=tm.BOUNDED_POSITION,
        load=coordinates.load, distractor=tm.BOUNDED_DISTRACTOR, instance_index=0)
    real = generate_token_targeted_task(tokenizer, request)
    fake = generate_token_targeted_task(_OneIdPerCharacterEncoder(), request)
    # Same declared length, different ids: the equality compares the ids, so a
    # wrong encoder is caught, while a length-only check would not be.
    assert fake.n_prompt_tokens == real.n_prompt_tokens == coordinates.token_length
    assert fake.prompt_ids != real.prompt_ids


# --------------------------------------------------------------------------
# assertion 2: streaming is contiguous, gapless, and reconstructs the prompt
# --------------------------------------------------------------------------


def test_streaming_chunks_tile_the_prompt_with_no_gap_or_reread(tokenizer) -> None:
    coordinates = tm.cell_coordinates(UNITS[0])
    model = RecordingModel(lambda shown: _solve(tokenizer, shown))
    run = rn.run_cell(coordinates, encode=tokenizer, generate=model,
                      access_mode=rn.STREAMING, instances=INSTANCES,
                      stream_chunk=rn.DEFAULT_STREAM_CHUNK)

    # Then: the access log was checked, and the plan covers [0, n) exactly.
    assert run.reread_checked is True
    assert run.chunks, "a streaming cell must record its accesses"
    assert run.chunks[0].start == 0
    assert run.chunks[-1].end == coordinates.token_length
    for earlier, later in zip(run.chunks, run.chunks[1:], strict=False):
        assert earlier.end == later.start  # contiguous and gapless, so no reread

    # And the union of the chunk spans is the whole prompt the model was shown --
    # a plan that covered the canvas but a read that skipped a chunk would split
    # these two apart, which is the failure the access log exists to catch.
    task = _task(tokenizer, UNITS[0], 0)
    reconstructed = tuple(
        token for access in run.chunks
        for token in task.prompt_ids[access.start:access.end])
    assert reconstructed == task.prompt_ids
    assert model.shown[0] == task.prompt_ids


def test_multi_instance_streaming_gives_each_instance_its_own_gapless_log(
        tokenizer) -> None:
    """n > 1 through the real tokenizer: one access log PER INSTANCE, not per cell.

    Each instance is its own stream starting at position 0, so a log shared across
    a cell would read instance 1's first chunk as a backwards jump and the runner
    would refuse the cell -- which is why a passing multi-instance streaming run IS
    the evidence that the logs are independent, and why this cannot be read off a
    single-instance run.
    """
    coordinates = tm.cell_coordinates(UNITS[0])
    instances = 3
    model = RecordingModel(lambda shown: _solve(tokenizer, shown))
    run = rn.run_cell(coordinates, encode=tokenizer, generate=model,
                      access_mode=rn.STREAMING, instances=instances,
                      stream_chunk=rn.DEFAULT_STREAM_CHUNK)

    # Then: every instance was driven and scored, so no instance was refused by a
    # stray backwards jump.
    assert run.reread_checked is True
    assert run.instances_scored == instances
    assert len(model.shown) == instances
    # And the k-th call is instance k's fixture, in order -- not one prompt read
    # `instances` times.
    expected = [_task(tokenizer, UNITS[0], index).prompt_ids
                for index in range(instances)]
    assert model.shown == expected

    # And the chunk plan still tiles the whole prompt with no gap or reread.
    assert run.chunks[0].start == 0
    assert run.chunks[-1].end == coordinates.token_length
    for earlier, later in zip(run.chunks, run.chunks[1:], strict=False):
        assert earlier.end == later.start


def test_a_full_canvas_cell_makes_no_chunk_accesses(tokenizer) -> None:
    coordinates = tm.cell_coordinates(UNITS[0])
    model = RecordingModel(lambda shown: _solve(tokenizer, shown))
    run = rn.run_cell(coordinates, encode=tokenizer, generate=model,
                      access_mode=rn.FULL_CANVAS, instances=INSTANCES)
    assert run.chunks == []
    assert run.reread_checked is False


# --------------------------------------------------------------------------
# assertion 3: the macro score covers measured cells only
# --------------------------------------------------------------------------


def test_a_correct_stub_produces_a_nonzero_macro_score(tokenizer) -> None:
    # Given: a stub whose answers are the fixtures' own gold, keyed by shown ids.
    oracle = _gold_oracle(tokenizer, UNITS, INSTANCES)
    grid = rn.run_grid(encode=tokenizer, generate=RecordingModel(oracle.__getitem__),
                       units=list(UNITS), instances=INSTANCES)

    # Then: nothing is unmeasured, and the pipeline produced a real number.
    assert grid["cells_unmeasured"] == []
    assert grid["cells_measured"] == len(UNITS)
    assert grid["macro_score"] is not None
    assert grid["macro_score"] > 0.0


@pytest.mark.parametrize("access_mode", rn.ACCESS_MODES)
def test_a_gold_reading_stub_reaches_a_perfect_macro_score(
        tokenizer, access_mode: str) -> None:
    """The ceiling check: a stub that only reads the prompt must reach 1.0.

    If the prompt layout were wrong -- evidence missing, query line buried, framing
    misordered -- the solver would fail to parse and the score would fall, so a
    scoring bug cannot hide behind a low ceiling here.
    """
    model = RecordingModel(lambda shown: _solve(tokenizer, shown))
    grid = rn.run_grid(encode=tokenizer, generate=model, units=list(UNITS),
                       instances=INSTANCES, access_mode=access_mode)
    assert grid["cells_unmeasured"] == []
    assert grid["macro_score"] == 1.0


def test_the_macro_score_covers_only_the_cells_that_produced_a_parseable_answer(
        tokenizer) -> None:
    """Half a grid parses; the score must be the mean of the half that did.

    The unparsed cell must be NAMED beside the score rather than folded in as a
    zero, which would report a harness failure as a model failure.
    """
    def by_family(shown: tuple[int, ...]) -> str:
        # Solve the recall cells; emit an unparseable completion for the rest.
        text = tokenizer.decode_ids(shown).decode("utf-8")
        if _FAMILY_KEYWORD["associative_recall"] in text:
            return _solve(tokenizer, shown)
        return "I cannot answer that."

    grid = rn.run_grid(encode=tokenizer, generate=RecordingModel(by_family),
                       units=list(UNITS), instances=INSTANCES)

    unmeasured_coordinates = tm.cell_coordinates(UNITS[1])
    assert grid["cells_measured"] == 1
    assert grid["cells_unmeasured"] == [
        f"{unmeasured_coordinates.family}@{unmeasured_coordinates.token_length}"
        f"L{unmeasured_coordinates.load}"]
    assert grid["macro_score"] == 1.0


@pytest.mark.parametrize("access_mode", rn.ACCESS_MODES)
def test_accuracy_at_more_than_one_instance_is_over_the_parseable_completions_only(
        tokenizer, access_mode: str) -> None:
    """n > 1: one correct, one unparseable, one wrong-but-parseable in one cell.

    Accuracy is ``n_correct / (n - n_missing)``, so this cell scores 1/2, NOT the
    1/3 a denominator that folded the unparsed completion in would report.  At
    ``instances == 1`` the two denominators coincide, which is why every other
    assertion in this file could pass with the wrong one: this is the case that
    tells "half of what the model actually answered" from "half of what ran".
    """
    family = tm.cell_coordinates(UNITS[0]).family
    # The runner drives instances in index order, so call k is instance k.  A
    # single-answer stub cannot build this mixed cell; scripting per call is what
    # makes one cell carry a correct, a missing and a wrong completion at once.
    call_index = [0]

    def answer(shown: tuple[int, ...]) -> str:
        index = call_index[0]
        call_index[0] += 1
        if index == 0:
            return _solve(tokenizer, shown)  # correct
        if index == 1:
            return "I cannot answer that."  # unparseable -> n_missing
        return _FIXED_ANSWER[family]  # parseable but not the gold

    run = rn.run_cell(tm.cell_coordinates(UNITS[0]), encode=tokenizer,
                      generate=RecordingModel(answer), access_mode=access_mode,
                      instances=3)

    assert run.instances_requested == 3
    assert run.n_missing == 1
    assert run.instances_scored == 2  # n - n_missing, not n
    assert run.measured is True
    assert run.missing_fraction == pytest.approx(1.0 / 3.0)
    assert run.accuracy == pytest.approx(0.5)  # 1 correct / 2 parseable, not 1 / 3


def test_both_access_modes_show_the_model_the_same_prompts(tokenizer) -> None:
    full_model = RecordingModel(lambda shown: _solve(tokenizer, shown))
    stream_model = RecordingModel(lambda shown: _solve(tokenizer, shown))
    result = rn.paired_access_modes(encode=tokenizer, generate_a=full_model,
                                    generate_b=stream_model, units=list(UNITS),
                                    instances=INSTANCES)
    # Both modes are driven; only the access is allowed to differ.
    assert result["paired_cells"] == len(UNITS)
    assert result["delta_streaming_minus_full"] == [0.0] * len(UNITS)


# --------------------------------------------------------------------------
# assertion 4: an unparseable completion is UNMEASURED, not wrong
# --------------------------------------------------------------------------


def test_an_unparseable_completion_makes_a_cell_unmeasured_not_zero(tokenizer) -> None:
    coordinates = tm.cell_coordinates(UNITS[0])
    model = RecordingModel(lambda shown: "I cannot answer that.")
    run = rn.run_cell(coordinates, encode=tokenizer, generate=model,
                      instances=INSTANCES)

    # Then: the cell reports no accuracy at all -- not a zero that would read as a
    # model failure -- and the missing count travels beside it.
    assert run.measured is False
    assert run.instances_scored == 0
    assert run.n_missing == INSTANCES
    assert run.accuracy != 0.0
    assert math.isnan(run.accuracy)


def test_a_well_formed_but_wrong_completion_is_measured_as_wrong(tokenizer) -> None:
    """The counterpart to assertion 4: a parseable wrong answer IS a measurement.

    Without this, "unparseable is unmeasured" would be indistinguishable from
    "nothing is ever measured", and a scorer that refused every answer would look
    correct.
    """
    family = tm.cell_coordinates(UNITS[0]).family
    model = RecordingModel(lambda shown: _FIXED_ANSWER[family])
    run = rn.run_cell(tm.cell_coordinates(UNITS[0]), encode=tokenizer,
                      generate=model, instances=INSTANCES)
    assert run.measured is True
    assert run.n_missing == 0
    assert run.accuracy == 0.0


def test_a_grid_with_no_parseable_answer_has_no_macro_score(tokenizer) -> None:
    model = RecordingModel(lambda shown: "nothing to score here")
    grid = rn.run_grid(encode=tokenizer, generate=model, units=list(UNITS),
                       instances=INSTANCES)
    assert grid["cells_measured"] == 0
    assert grid["macro_score"] is None
    assert len(grid["cells_unmeasured"]) == len(UNITS)


def test_the_grid_collapses_the_load_axis_when_counting_measured_cells(
        tokenizer) -> None:
    """A DEFECT, recorded rather than silently exercised past.

    ``run_grid`` keys its ``measured`` mapping by ``(family, token_length)``, a key
    the load axis does not enter, so a cut that varies load -- which the bounded
    cut does, four loads per (family, length) -- keeps only the LAST load's
    accuracy per key.  The runner's own contract says the macro covers "every cell
    in the cut" and that ``cells_unmeasured`` names what it does not cover; here
    four cells ran and were measured, yet ``cells_measured`` is 1 and nothing is
    named as omitted, so 27 of the bounded cut's 36 cells would vanish from the
    score at ``units=None``.  The silent last-wins overwrite, not the collapse
    itself, is the defect: ``cells_measured + len(cells_unmeasured)`` is supposed
    to equal ``len(cells)`` and does not.

    The two assertions below pin both halves of the contradiction, so a fix cannot
    land without replacing this test.  The fix is NOT to make ``cell_key`` carry
    the load, which is the repair the first version of this docstring prescribed
    and which does not work: ``macro_score`` takes a ``dict[(str, int), float]``
    and refuses any other key shape -- ``('associative_recall', 16384, 1)`` raises
    ``StatisticsRefusal`` "a macro cell is keyed by (family, length)" -- and
    ``run_grid`` never calls ``cell_key`` anyway, it inlines ``(c.family,
    c.token_length)``.  The contract has to be chosen explicitly and this test
    replaced with one asserting the chosen one: either average the loads within
    each ``(family, token_length)`` so the macro stays the protocol's nine
    equal-weight cells, or widen the macro to cover loads and change
    ``statistics.macro_score`` and ``test_statistics.py`` with it.
    """
    units = (0, 1, 2, 3)  # one (family, token_length), four different loads
    grid = rn.run_grid(encode=tokenizer,
                       generate=RecordingModel(lambda shown: _solve(tokenizer, shown)),
                       units=list(units), instances=INSTANCES)

    loads = {tm.cell_coordinates(unit).load for unit in units}
    assert len(loads) == len(units), "the probe needs one load per cell"
    assert len(grid["cells"]) == len(units)
    assert grid["cells_measured"] == 1
    assert grid["cells_measured"] + len(grid["cells_unmeasured"]) != len(grid["cells"])
