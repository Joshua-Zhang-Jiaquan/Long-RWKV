"""The experimental arms, as data -- importable without a GPU.

This table lived in :mod:`longrwkv.model`, which imports ``fla`` at module scope
and therefore cannot import on a host without a Triton driver.  The consequence
was not a missing test but an *unrunnable* one: ``test_eval_interventions``
reaches for ``ARM_SPECS`` inside a ``try`` and skips when the import raises, so
on this CPU host every assertion about what an arm is was silently skipped.

That is how A0 and M6 came to be field-for-field identical.  A0 is the
manuscript's "autoregressive adaptation of the same base" and its "ordinary
recurrent reference"; M6 is "a forward-only denoiser".  They differ in
*objective* alone, and with the objective absent from the table the two arms
collapsed onto one definition -- ``train_body("A0")`` and ``train_body("M6")``
emitted byte-identical ``EXTRA_ARGS``, and the baseline that every diffusion arm
is measured against trained the masked objective with a 0.1-weighted causal
auxiliary.  Nothing downstream could tell: ``run_record.arm`` read ``"A0"``.

So the table is here, in a module with no import that can fail, and the arm
identity tests run on every CPU suite invocation rather than skipping.
``longrwkv.model`` re-exports these names, so every existing importer is
unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass

from .refusal import Refusal

#: The two training objectives an arm can carry.  ``masked_denoising`` is the
#: paper's selected-token objective (Equation (loss)) with the causal term as a
#: small auxiliary; ``autoregressive`` is next-token prediction on *clean* input
#: with no masked term at all.
MASKED_DENOISING: str = "masked_denoising"
AUTOREGRESSIVE: str = "autoregressive"
OBJECTIVES: tuple[str, ...] = (MASKED_DENOISING, AUTOREGRESSIVE)


@dataclass(frozen=True)
class ArmSpec:
    """What each experimental arm turns on.  One place, so no arm is mislabelled.

    ``objective`` is a field here, and not a training flag, because it is the
    axis that separates A0 from M6.  A flag a caller may forget cannot carry a
    distinction the paper's causal claims rest on; a field of the arm's own
    specification can, and ``spec_digest`` below makes two arms that claim to
    differ prove it.
    """

    name: str
    bidirectional: bool
    hidden_loop: bool
    gate_mod: bool
    carried_state: bool = False
    soft_feedback: bool = False
    objective: str = MASKED_DENOISING

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise Refusal(
                f"arm {self.name} declares objective {self.objective!r}; the "
                f"declared objectives are {list(OBJECTIVES)}")
        if self.objective == AUTOREGRESSIVE and self.bidirectional:
            raise Refusal(
                f"arm {self.name} declares the autoregressive objective with "
                f"bidirectional=True: next-token prediction over a mixer that "
                f"reads the suffix is not a causal language model, and the row "
                f"it produced would be labelled as one")

    @property
    def causal_only(self) -> bool:
        """Whether the backward direction is never called."""
        return not self.bidirectional

    @property
    def autoregressive(self) -> bool:
        """Whether this arm both trains and decodes left-to-right.

        Read by the trainer to pick the objective and by the eval lane to pick
        the operating point, so the two cannot disagree about what an arm is.
        """
        return self.objective == AUTOREGRESSIVE

    def spec_digest(self) -> tuple:
        """Everything that defines the arm except its name.

        Two arms with equal digests are the same experiment under two labels.
        :func:`require_distinct_arms` reads this, so the A0/M6 collapse is a
        test failure on this host rather than a pair of duplicate 32-GPU runs.
        """
        return (self.bidirectional, self.hidden_loop, self.gate_mod,
                self.carried_state, self.soft_feedback, self.objective)


ARM_SPECS: dict[str, ArmSpec] = {
    # A0 is the manuscript's "autoregressive adaptation of the same base" and its
    # "ordinary recurrent reference" -- the baseline every diffusion arm is
    # measured against.  It trains next-token CE on clean input and decodes one
    # position per call; M6 below is the *denoiser* sharing its forward-only
    # direction, and the pair isolates direction-versus-objective only while
    # these two lines differ.
    "A0": ArmSpec("A0", bidirectional=False, hidden_loop=False, gate_mod=False,
                  objective=AUTOREGRESSIVE),
    "A1": ArmSpec("A1", bidirectional=True, hidden_loop=False, gate_mod=False),
    "A2": ArmSpec("A2", bidirectional=True, hidden_loop=True, gate_mod=False),
    "A3": ArmSpec("A3", bidirectional=True, hidden_loop=True, gate_mod=True),
    "M6": ArmSpec("M6", bidirectional=False, hidden_loop=False, gate_mod=False),
    "R0": ArmSpec("R0", bidirectional=True, hidden_loop=False, gate_mod=False,
                  carried_state=True),
    "M4": ArmSpec("M4", bidirectional=True, hidden_loop=False, gate_mod=True,
                  soft_feedback=True),
    "M5": ArmSpec("M5", bidirectional=True, hidden_loop=True, gate_mod=True,
                  carried_state=True),
}


def require_distinct_arms(specs: dict[str, ArmSpec] | None = None) -> None:
    """Refuse a table in which two arms are the same experiment.

    The campaign spends a 32-GPU job per arm and the paper reads each as a
    separate row, so two labels over one specification is both a wasted run and
    a control that controls nothing.
    """
    table = ARM_SPECS if specs is None else specs
    seen: dict[tuple, str] = {}
    for name, spec in table.items():
        digest = spec.spec_digest()
        if digest in seen:
            raise Refusal(
                f"arms {seen[digest]} and {name} have identical specifications "
                f"{digest}: two labels over one experiment. Every arm the paper "
                f"reports as a separate row must differ in at least one field.")
        seen[digest] = name
