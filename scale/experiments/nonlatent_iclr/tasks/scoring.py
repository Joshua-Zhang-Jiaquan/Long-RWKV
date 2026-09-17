"""Answer scoring for the long-context exact tasks.

``exact_tasks.validate_exact_gold`` recomputes the evaluator-private gold from
the public evidence; it decides whether a *task* is well-formed.  This module
answers a different question -- whether a model *completion* matched that gold --
and the two must not be confused for one another.

The distinction matters because scoring is where a harness failure can
masquerade as a model failure:

* A completion the extractor cannot parse is recorded as MISSING, never as a
  wrong answer.  A scorer that substituted the empty string for "could not
  parse" and compared it to gold would silently turn "the decoder emitted text
  this grammar does not describe" into "the model answered incorrectly", so a
  cell would report the health of the harness as if it were the accuracy of the
  model.
* ``score_cell`` reports ``missing_fraction`` next to ``accuracy`` and computes
  accuracy over the *parseable* completions only.  A cell where half the
  completions were unparseable is a 50%-missing cell, not a 50%-accuracy cell;
  folding the unparsed half into the denominator caps a perfect model at 50% and
  is exactly the confusion this module exists to prevent.

Extraction is family-aware because the four registered families write their
answer with different grammars -- a ``v``-prefixed value for the recall and
overwrite families, a decimal integer for the dataflow family, and an exact
reduced fraction for the finite-HMM family -- and one pattern for all four would
either miss valid answers or bind to unrelated text.
"""

from __future__ import annotations

import math
import re
from typing import Final

from .models import FAMILIES

#: The keys ``score_cell`` reads from each record.  Named here, and read from
#: here, so a producer and this scorer cannot drift onto different spellings and
#: silently score an empty column as a column of missing answers.
DECODED_KEY: Final = "decoded"
GOLD_KEY: Final = "gold"

#: The answer grammar of each registered family, as the generator writes it.
#: ``associative_recall`` and ``overwrite_delayed_query`` answer with a value
#: token (``v<digits>``); ``code_dataflow`` with a decimal integer;
#: ``finite_hmm`` with an exact reduced fraction (``<num>/<den>``).  Case is
#: ignored so a completion that capitalises the value prefix still parses; a
#: family absent from this map has no grammar and is refused, never guessed at.
FAMILY_ANSWER_PATTERN: Final = {
    "associative_recall": re.compile(r"v\d+", re.IGNORECASE),
    "overwrite_delayed_query": re.compile(r"v\d+", re.IGNORECASE),
    "code_dataflow": re.compile(r"\d+"),
    "finite_hmm": re.compile(r"\d+/\d+"),
}


class ScoringRefusal(ValueError):
    """A cell cannot be scored as declared.

    Raised rather than returning a number.  An empty record list has no
    accuracy, and an unknown family has no answer grammar; both would otherwise
    surface as a zero that reads like a measured result.  A completion the
    extractor cannot parse is deliberately *not* this: that is a MISSING record,
    which is data about the run, not a refusal to score it.
    """


def extract_answer(decoded: str, family: str) -> str | None:
    """The model's answer as that family writes it, or ``None`` if absent.

    ``None`` -- never the empty string -- when no answer-grammar token is
    present, so a caller is compelled to record the completion as missing rather
    than compare an empty prediction and charge it to the model as wrong.

    The LAST match wins.  An overwrite task's answer is by construction its last
    write, so reading the final value token mirrors the task's own rule, and it
    also keeps a trailing clause or a sentence-final period from shadowing the
    answer the model committed to.
    """
    pattern = FAMILY_ANSWER_PATTERN.get(family)
    if pattern is None:
        raise ScoringRefusal(f"no answer grammar for family: {family}")
    matches = pattern.findall(decoded)
    return matches[-1] if matches else None


def normalize(text: str) -> str:
    """Case, whitespace, and a trailing period are not answer differences.

    The generator writes its gold in canonical form; a completion that agrees
    but ends a sentence (``42.``) or capitalises a value prefix (``V1234``)
    should still match.  Collapsing internal whitespace before removing the
    period means a period separated from the answer by a space is removed too.
    """
    normalized = " ".join(text.lower().split())
    return normalized[:-1].strip() if normalized.endswith(".") else normalized


def exact_match(pred: str | None, gold: str) -> bool:
    """Whether an extracted answer equals the gold, modulo normalization.

    ``None`` is never a match.  An unparsed completion is missing, and answering
    True or False for it would be a claim about the model rather than about the
    scorer; callers that need missing counted separately use ``score_cell``.
    """
    if pred is None:
        return False
    return normalize(pred) == normalize(gold)


def score_cell(records: list[dict], family: str) -> dict:
    """Score one cell, keeping unparseable completions apart from accuracy.

    ``accuracy`` is ``n_correct / (n - n_missing)`` -- the fraction of the
    *parseable* completions that were right.  A cell where half the completions
    were unparseable is therefore a 50%-missing cell, not a 50%-accuracy one:
    folding the unparsed half into the denominator would report a harness or
    decoding failure as a model failure, which is the defect this module exists
    to prevent.  When nothing parsed, accuracy is NaN -- undefined, not zero --
    and ``missing_fraction`` carries the report.

    An empty record list or an unknown family is refused, because either would
    otherwise yield a number that reads like a measurement.
    """
    if family not in FAMILY_ANSWER_PATTERN:
        raise ScoringRefusal(f"no answer grammar for family: {family}")
    if not records:
        raise ScoringRefusal("cannot score an empty cell: it has no accuracy")
    n_correct = 0
    n_missing = 0
    for record in records:
        decoded = record.get(DECODED_KEY)
        gold = record.get(GOLD_KEY)
        if not isinstance(decoded, str) or not isinstance(gold, str):
            msg = (f"record lacks a string {DECODED_KEY!r} and {GOLD_KEY!r}: "
                   f"found keys {sorted(record)}")
            raise ScoringRefusal(msg)
        answer = extract_answer(decoded, family)
        if answer is None:
            n_missing += 1
        elif exact_match(answer, gold):
            n_correct += 1
    n = len(records)
    answered = n - n_missing
    accuracy = n_correct / answered if answered else math.nan
    return {
        "n": n,
        "n_missing": n_missing,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "missing_fraction": n_missing / n,
    }


if set(FAMILY_ANSWER_PATTERN) != set(FAMILIES):
    # A registered family with no grammar would be silently unscorable, which is
    # the same class of defect as an unparsed completion scored as wrong: a
    # missing instrument read as a measured result.  Fail at import so a new
    # registry family cannot be added without an answer grammar.
    raise ScoringRefusal(
        "the answer grammars and the registered families have diverged: "
        f"grammars={sorted(FAMILY_ANSWER_PATTERN)} families={sorted(FAMILIES)}")
