"""Measured model semantics; pass predicates are derived, never supplied."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from .lifecycle_evidence import StrictEvidence


class ParameterGroup(StrictEvidence):
    names: tuple[str, ...]
    tensors: int = Field(ge=0)
    numel: int = Field(ge=0)


class MembershipEvidence(StrictEvidence):
    freeze_backbone: bool
    latent_on: Literal[False] = False
    total_tensors: int = Field(gt=0)
    total_numel: int = Field(gt=0)
    trainable_tensors: int = Field(ge=0)
    trainable_numel: int = Field(ge=0)
    groups: tuple[ParameterGroup, ...]
    each_trainable_once: bool
    frozen_absent: bool
    cross_group_disjoint: bool
    freeze_rule_exact: bool
    optimizer_state_entries: Literal[0] = 0

    @property
    def passed(self) -> bool:
        names = tuple(name for group in self.groups for name in group.names)
        return (self.each_trainable_once and self.frozen_absent and self.cross_group_disjoint
                and self.freeze_rule_exact and len(self.groups) == 2
                and all(group.tensors == len(group.names) for group in self.groups)
                and len(names) == len(set(names)) == self.trainable_tensors
                and sum(group.numel for group in self.groups) == self.trainable_numel)


GradientState = Literal["none", "zero", "nonzero", "nonfinite"]


class ParameterGradient(StrictEvidence):
    name: str
    name_class: Literal["backward_attention", "forward_attention", "fusion", "loop", "shared"]
    numel: int = Field(gt=0)
    requires_grad: bool
    gated: bool
    before: GradientState
    after: GradientState
    unchanged_when_ungated: bool

    @property
    def passed(self) -> bool:
        if self.before == "nonfinite" or self.after == "nonfinite":
            return False
        if not self.requires_grad and self.before != "none":
            return False
        expected = "zero" if self.gated and self.before != "none" else self.before
        return self.after == expected and (self.gated or self.unchanged_when_ungated)


class GradientEvidence(StrictEvidence):
    step: Literal[0] = 0
    steps: Literal[100] = 100
    stage_a_frac: float = Field(default=0.2, ge=0.2, le=0.2)
    stage_b_frac: float = Field(default=0.6, ge=0.6, le=0.6)
    n_layers: int = Field(gt=0)
    parameters: tuple[ParameterGradient, ...] = Field(min_length=1)

    @property
    def passed(self) -> bool:
        return (all(row.passed for row in self.parameters)
                and len({row.name for row in self.parameters}) == len(self.parameters)
                and any(row.gated and row.before == "nonzero" for row in self.parameters)
                and any(not row.gated and row.after == "nonzero" for row in self.parameters))


class MaskEvidence(StrictEvidence):
    device: str
    shape: tuple[Literal[1], Literal[48], Literal[65536]]
    reduction: Literal["historical_rank_local"] = "historical_rank_local"
    selected_tokens: int = Field(gt=0, lt=48)
    eligible_tokens: Literal[46] = 46
    eligibility_exact: bool
    corruption_exact: bool
    logits_finite: bool
    loss: float
    selected_mean: float
    unselected_loss: float
    empty_loss: float
    unselected_logit_grad_zero: bool

    @property
    def passed(self) -> bool:
        return (self.eligibility_exact and self.corruption_exact and self.logits_finite
                and self.selected_tokens <= self.eligible_tokens
                and all(math.isfinite(value) for value in
                        (self.loss, self.selected_mean, self.unselected_loss, self.empty_loss))
                and math.isclose(self.loss, self.selected_mean, rel_tol=1e-6, abs_tol=2e-6)
                and math.isclose(self.loss, self.unselected_loss, rel_tol=1e-6, abs_tol=2e-6)
                and self.empty_loss == 0.0 and self.unselected_logit_grad_zero)
