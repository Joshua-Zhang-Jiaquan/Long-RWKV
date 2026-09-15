from __future__ import annotations

import json
from dataclasses import replace

import pytest
from scale.experiments.nonlatent_iclr.tasks.exact_tasks import ExactTaskRequest
from scale.experiments.nonlatent_iclr.tasks.exact_tasks import InfeasibleTaskError
from scale.experiments.nonlatent_iclr.tasks.exact_tasks import generate_exact_task
from scale.experiments.nonlatent_iclr.tasks.exact_tasks import public_evidence
from scale.experiments.nonlatent_iclr.tasks.exact_tasks import validate_exact_gold


def test_exact_task_is_deterministic_when_request_is_identical() -> None:
    # Given: a fixed controlled-task request
    request = ExactTaskRequest("associative_recall", 101, 4096, 10, 1, "none", 0)

    # When: the lazy generator is called twice
    first = generate_exact_task(request)
    second = generate_exact_task(request)

    # Then: the canonical public condition and private answer agree exactly
    assert first == second
    assert first.condition.source_bytes.decode("utf-8") in first.condition.public_prompt


def test_exact_gold_is_not_serialized_into_public_condition() -> None:
    # Given: every required exact family at a single declared cell
    requests = tuple(
        ExactTaskRequest(family, 103, 8192, 50, 8, "similar", 4)
        for family in ("associative_recall", "overwrite_delayed_query", "finite_hmm", "code_dataflow")
    )

    # When: public conditions are serialized
    public_documents = tuple(json.dumps(generate_exact_task(request).condition.public_dict()) for request in requests)

    # Then: hidden answer storage is absent, while an answer may naturally occur in evidence
    assert all("hidden_answer" not in document for document in public_documents)
    assert any(task.correct_answer in task.condition.public_prompt for task in map(generate_exact_task, requests))


def test_hmm_answer_uses_exact_fraction_when_evidence_is_known() -> None:
    # Given: a fixed finite-HMM task whose prompt contains observed emissions
    task = generate_exact_task(ExactTaskRequest("finite_hmm", 105, 4096, 90, 32, "random", 7))

    # When: its independent gold answer is exposed only to the evaluator object
    answer = task.correct_answer

    # Then: it is an exact reduced fraction rather than a floating-point approximation
    numerator, denominator = answer.split("/")
    assert int(numerator) > 0
    assert int(denominator) > int(numerator)


def test_distractor_variants_change_the_public_condition_without_changing_gold() -> None:
    # Given: otherwise identical associative-recall cells with declared distractors
    base = ("associative_recall", 101, 4096, 50, 8)
    random_task = generate_exact_task(ExactTaskRequest(*base, "random", 3))
    similar_task = generate_exact_task(ExactTaskRequest(*base, "similar", 3))

    # When: the lazy variants are generated
    prompts = (random_task.condition.public_prompt, similar_task.condition.public_prompt)

    # Then: each view records its distractor content while preserving the analytic target
    assert all("DISTRACT" in prompt for prompt in prompts)
    assert random_task.correct_answer == similar_task.correct_answer


def test_gold_validation_rejects_redacted_or_contradicted_public_evidence() -> None:
    # Given: a generated public prompt and private evaluator gold
    task = generate_exact_task(ExactTaskRequest("overwrite_delayed_query", 101, 8192, 50, 8, "none", 0))

    # When: the public evidence is redacted or contradicted without touching private source bytes
    redacted = replace(task, condition=replace(task.condition, public_prompt="CONTEXT\nBEGIN EVIDENCE\nEND EVIDENCE\n"))
    contradicted_prompt = task.condition.public_prompt.replace(public_evidence(task), "OVERWRITE k0=v1;QUERY k0")
    contradicted = replace(task, condition=replace(task.condition, public_prompt=contradicted_prompt))

    # Then: evaluator validation rejects both public-evidence mutations
    assert validate_exact_gold(redacted) is False
    assert validate_exact_gold(contradicted) is False


def test_request_bounds_and_exact_loads_are_enforced() -> None:
    # Given: out-of-contract requests and declared HMM/dataflow endpoints
    invalid = (
        ("associative_recall", 999, 4096, 10, 1, "none", 0),
        ("associative_recall", 101, 4096, 10, 1, "none", 200),
        ("associative_recall", 101, 4096, 10, 2, "none", 0),
    )

    # When: requests are parsed and endpoint prompts are generated
    for values in invalid:
        with pytest.raises(ValueError):
            ExactTaskRequest(*values)
    hmm_one = public_evidence(generate_exact_task(ExactTaskRequest("finite_hmm", 101, 8192, 50, 1, "none", 0)))
    hmm_many = public_evidence(generate_exact_task(ExactTaskRequest("finite_hmm", 101, 8192, 50, 128, "none", 0)))
    flow_one = public_evidence(generate_exact_task(ExactTaskRequest("code_dataflow", 101, 8192, 50, 1, "none", 0)))
    flow_many = public_evidence(generate_exact_task(ExactTaskRequest("code_dataflow", 101, 8192, 50, 128, "none", 0)))

    # Then: no endpoint is silently capped
    assert len(hmm_one.split("observations=")[1].split(";")[0]) == 1
    assert len(hmm_many.split("observations=")[1].split(";")[0]) == 128
    assert flow_one.count("=") == 2
    assert flow_many.count("=") == 129


def test_tiny_or_unplaceable_character_fixtures_are_explicitly_infeasible() -> None:
    # Given: declared lengths incapable of containing an exact task at its requested position
    with pytest.raises(ValueError):
        ExactTaskRequest("associative_recall", 101, 0, 10, 1, "none", 0)
    request = ExactTaskRequest("finite_hmm", 101, 4096, 90, 128, "similar", 0)

    # When: task construction is attempted
    # Then: the generator fails rather than clamping padding or positions
    with pytest.raises(InfeasibleTaskError):
        generate_exact_task(request)


def test_public_evidence_starts_at_the_declared_character_position() -> None:
    # Given: a feasible declared location fixture
    request = ExactTaskRequest("code_dataflow", 101, 8192, 50, 32, "none", 0)
    task = generate_exact_task(request)

    # When: the public evidence projection is located in the prompt
    offset = task.condition.public_prompt.index(public_evidence(task))

    # Then: target evidence is at the requested character fraction exactly
    assert offset == request.declared_length * request.position_fraction // 100


def test_hmm_query_and_observations_are_strict_public_evidence() -> None:
    # Given: a valid HMM public prompt and evaluator-private state-A gold
    task = generate_exact_task(ExactTaskRequest("finite_hmm", 101, 8192, 50, 8, "none", 0))

    # When: its query or binary observations are contradicted at the same prompt length
    state_b = replace(task, condition=replace(task.condition, public_prompt=task.condition.public_prompt.replace("state_A", "state_B")))
    malformed = replace(task, condition=replace(task.condition, public_prompt=task.condition.public_prompt.replace("observations=", "observations=x", 1).replace("x" + public_evidence(task).split("observations=")[1][0], "x", 1)))

    # Then: neither altered public query nor nonbinary observation can retain old gold
    assert validate_exact_gold(state_b) is False
    assert validate_exact_gold(malformed) is False


def test_public_prompt_rejects_multiple_evidence_blocks_at_same_length() -> None:
    # Given: a valid public condition with padding available for a second block
    task = generate_exact_task(ExactTaskRequest("associative_recall", 101, 8192, 10, 1, "none", 0))
    block = "\nBEGIN EVIDENCE\nASSOC k0=v0;QUERY k0\nEND EVIDENCE"

    # When: equal-length padding is replaced by contradictory evidence markup
    altered_prompt = task.condition.public_prompt[:-len(block)] + block
    altered = replace(task, condition=replace(task.condition, public_prompt=altered_prompt))

    # Then: ambiguity fails parsing rather than silently selecting the first block
    assert len(altered.condition.public_prompt) == task.condition.declared_length
    assert validate_exact_gold(altered) is False
