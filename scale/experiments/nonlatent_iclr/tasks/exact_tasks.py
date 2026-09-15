"""Lazy deterministic exact-task families with independent evaluator checks."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from random import Random

from .models import ExactTask
from .models import ExactTaskRequest
from .models import PublicCondition


def generate_exact_task(request: ExactTaskRequest) -> ExactTask:
    """Generate one public condition and evaluator-private exact answer lazily."""
    rng = Random(_seed(request))
    match request.family:
        case "associative_recall":
            evidence, answer = _associative(rng, request.load)
        case "overwrite_delayed_query":
            evidence, answer = _overwrite(rng, request.load)
        case "finite_hmm":
            evidence, answer = _hmm(rng, request.load)
        case "code_dataflow":
            evidence, answer = _dataflow(rng, request.load)
        case other:
            raise ExactFamilyError(other)
    source = evidence.encode("utf-8")
    family = f"{request.family}:seed-{request.data_seed}:instance-{request.instance_index}"
    prompt_evidence = evidence + _distractor_text(request)
    prompt = _positioned_prompt(prompt_evidence, request)
    condition = PublicCondition(prompt, source, family, request.declared_length, "logical_character_fixture")
    return ExactTask(condition, answer)


def validate_exact_gold(task: ExactTask) -> bool:
    """Independently recompute gold from the public prompt evidence."""
    if len(task.condition.public_prompt) != task.condition.declared_length:
        return False
    try:
        evidence = public_evidence(task)
        match evidence.split(" ", maxsplit=1)[0]:
            case "HMM":
                answer = _analytic_hmm(evidence)
            case "FLOW":
                answer = _evaluate_flow(evidence)
            case "OVERWRITE":
                answer = _overwritten_value(evidence)
            case "ASSOC":
                answer = _lookup_association(evidence)
            case _:
                return False
    except (EvidenceFormatError, IndexError, KeyError, ValueError):
        return False
    return answer == task.correct_answer


def public_evidence(task: ExactTask) -> str:
    """Parse the evidence projection from the public prompt, never private bytes."""
    opening = "BEGIN EVIDENCE\n"
    closing = "\nEND EVIDENCE"
    if task.condition.public_prompt.count("BEGIN EVIDENCE") != 1 or task.condition.public_prompt.count("END EVIDENCE") != 1:
        raise EvidenceFormatError()
    _, found_opening, remainder = task.condition.public_prompt.partition(opening)
    evidence, found_closing, _ = remainder.partition(closing)
    if not found_opening or not found_closing or not evidence:
        raise EvidenceFormatError()
    return evidence.split(";DISTRACT ", maxsplit=1)[0]


def _associative(rng: Random, load: int) -> tuple[str, str]:
    pairs = tuple((f"k{i}", f"v{rng.randrange(10_000)}") for i in range(load))
    key, answer = pairs[rng.randrange(len(pairs))]
    records = ";".join(f"{left}={right}" for left, right in pairs)
    return f"ASSOC {records};QUERY {key}", answer


def _overwrite(rng: Random, load: int) -> tuple[str, str]:
    key = f"k{rng.randrange(max(load, 1))}"
    initial = f"v{rng.randrange(10_000)}"
    final = f"v{rng.randrange(10_000)}"
    distractors = ";".join(f"k{i}=v{rng.randrange(10_000)}" for i in range(load))
    return f"OVERWRITE {key}={initial};{distractors};{key}={final};QUERY {key}", final


def _hmm(rng: Random, load: int) -> tuple[str, str]:
    observations = "".join(str(rng.randrange(2)) for _ in range(load))
    evidence = f"HMM observations={observations};query=state_A"
    return evidence, _analytic_hmm(evidence)


def _dataflow(rng: Random, load: int) -> tuple[str, str]:
    first = rng.randrange(2, 20)
    statements = [f"a={first}"]
    previous = "a"
    for index in range(1, load + 1):
        name = f"n{index}"
        statements.append(f"{name}={previous}+{rng.randrange(1, 9)}")
        previous = name
    return f"FLOW {';'.join(statements)};QUERY {previous}", _evaluate_flow(f"FLOW {';'.join(statements)};QUERY {previous}")


def _positioned_prompt(evidence: str, request: ExactTaskRequest) -> str:
    evidence_start = request.declared_length * request.position_fraction // 100
    prefix_framing = "CONTEXT \nBEGIN EVIDENCE\n"
    suffix_framing = "\nEND EVIDENCE\nANSWER THE QUERY\n"
    prefix_size = evidence_start - len(prefix_framing)
    remaining = request.declared_length - evidence_start - len(evidence) - len(suffix_framing)
    if prefix_size < 0 or remaining < 0:
        raise InfeasibleTaskError(request, len(evidence), evidence_start)
    return f"CONTEXT {'x' * prefix_size}\nBEGIN EVIDENCE\n{evidence}{suffix_framing}{'y' * remaining}"


def _distractor_text(request: ExactTaskRequest) -> str:
    rng = Random(_seed(request) ^ 0x5A17)
    match request.distractor:
        case "none":
            return ""
        case "random":
            return ";DISTRACT random=" + "".join(chr(97 + rng.randrange(26)) for _ in range(request.load))
        case "similar":
            return ";DISTRACT similar=" + ";".join(f"k{i}=v{rng.randrange(10_000)}" for i in range(request.load))
        case other:
            raise DistractorError(other)


def _lookup_association(evidence: str) -> str:
    records, query = evidence.removeprefix("ASSOC ").split(";QUERY ")
    values = dict(item.split("=") for item in records.split(";"))
    return values[query]


def _overwritten_value(evidence: str) -> str:
    records, query = evidence.removeprefix("OVERWRITE ").split(";QUERY ")
    values: dict[str, str] = {}
    for item in records.split(";"):
        key, value = item.split("=")
        values[key] = value
    return values[query]


def _analytic_hmm(evidence: str) -> str:
    observations, query = _parse_hmm(evidence)
    state_a = Fraction(1, 2)
    state_b = Fraction(1, 2)
    for observation in observations:
        next_a = state_a * Fraction(3, 4) + state_b * Fraction(1, 3)
        next_b = state_a * Fraction(1, 4) + state_b * Fraction(2, 3)
        match observation:
            case "0":
                likelihood_a, likelihood_b = Fraction(2, 3), Fraction(1, 3)
            case "1":
                likelihood_a, likelihood_b = Fraction(1, 3), Fraction(2, 3)
            case _:
                raise EvidenceFormatError()
        normalizer = next_a * likelihood_a + next_b * likelihood_b
        state_a, state_b = next_a * likelihood_a / normalizer, next_b * likelihood_b / normalizer
    match query:
        case "state_A":
            answer = state_a
        case "state_B":
            answer = state_b
        case _:
            raise EvidenceFormatError()
    return f"{answer.numerator}/{answer.denominator}"


def _parse_hmm(evidence: str) -> tuple[str, str]:
    prefix = "HMM observations="
    if not evidence.startswith(prefix):
        raise EvidenceFormatError()
    observations, separator, query = evidence.removeprefix(prefix).partition(";query=")
    if not separator or not observations or ";" in query:
        raise EvidenceFormatError()
    return observations, query


def _evaluate_flow(evidence: str) -> str:
    statements, query = evidence.removeprefix("FLOW ").split(";QUERY ")
    values: dict[str, int] = {}
    for statement in statements.split(";"):
        name, expression = statement.split("=")
        if "+" in expression:
            parent, increment = expression.split("+")
            values[name] = values[parent] + int(increment)
        else:
            values[name] = int(expression)
    return str(values[query])


def _seed(request: ExactTaskRequest) -> int:
    framed = ":".join((request.family, str(request.data_seed), str(request.instance_index)))
    return int(hashlib.sha256(framed.encode("utf-8")).hexdigest()[:16], 16)


class ExactFamilyError(ValueError):
    """Raised when a registry refers to an undeclared exact family."""

    def __init__(self, family: str) -> None:
        super().__init__(f"undeclared exact task family: {family}")


class DistractorError(ValueError):
    def __init__(self, distractor: str) -> None:
        super().__init__(f"undeclared distractor class: {distractor}")


class EvidenceFormatError(ValueError):
    """Raised when a public prompt lacks a single usable evidence projection."""


class InfeasibleTaskError(ValueError):
    """Raised when declared length and position cannot contain the public evidence."""

    def __init__(self, request: ExactTaskRequest, evidence_length: int, evidence_start: int) -> None:
        super().__init__(f"INFEASIBLE length={request.declared_length} position={evidence_start} evidence={evidence_length}")
