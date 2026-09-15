"""CPU synthetic seams only; no GPU qualification evidence."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from scale.experiments.nonlatent_iclr.qualification import semantics_optimizer, semantics_probe


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_bwd = nn.Linear(1, 1)
        self.backbone = nn.Linear(1, 1)
        self.latent_cond = nn.Linear(1, 1)
        self.unused = nn.Parameter(torch.ones(1))
        self.eval()

    def forward(self, input_ids: torch.Tensor, **kwargs: None | bool) -> torch.Tensor:
        hidden = input_ids.float().unsqueeze(-1) / 65536
        hidden = self.attn_bwd(hidden) + self.backbone(hidden)
        return hidden * torch.linspace(-1, 1, 65536).view(1, 1, -1)


def test_membership_when_loaded_parameters_are_grouped() -> None:
    # Given: a tiny real CPU module including dormant conditioner parameters.
    model = TinyModel()
    module = semantics_optimizer
    # When: exact historical grouping and both freeze policies are applied.
    observations = module.probe_membership(model)
    # Then: identity coverage, counts and original flags are preserved.
    assert len(observations) == 2 and all(item.passed for item in observations)
    assert observations[0].total_tensors == 7
    assert observations[0].trainable_tensors == 7
    assert observations[1].trainable_tensors == 2
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_membership_rejects_when_group_duplicates_an_object() -> None:
    # Given: duplicate references across decay groups.
    model = TinyModel()
    module = semantics_optimizer
    parameters = tuple(model.parameters())
    # When: the identity auditor sees the same loaded object twice.
    result = module.audit_membership(model, (parameters, parameters), False)
    # Then: a passing count cannot conceal duplicate optimizer membership.
    assert not result.passed and not result.each_trainable_once
    assert not result.cross_group_disjoint


def test_mask_and_gradients_when_actual_cpu_logits_are_used() -> None:
    # Given: an actual differentiable CPU model; unused parameters stay None.
    model = TinyModel()
    module = semantics_probe
    # When: one bounded forward/backward is followed by the historical stage-A gate.
    mask, gradients = module.probe_loss_and_gradients(model, torch.device("cpu"), 4)
    # Then: selected-token CE and zero-vs-None policy hold on actual tensors.
    assert mask.passed and gradients.passed
    rows = {row.name: row for row in gradients.parameters}
    assert rows["backbone.weight"].before == "nonzero"
    assert rows["backbone.weight"].after == "zero"
    assert rows["attn_bwd.weight"].after == "nonzero"
    assert rows["unused"].before == rows["unused"].after == "none"
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("field,value", [
    ("selected_mean", 100.0), ("unselected_loss", 100.0),
    ("empty_loss", 1.0), ("eligibility_exact", False),
])
def test_mask_rejects_when_observations_disagree(field: str, value: float | bool) -> None:
    # Given: valid observations from the synthetic CPU model.
    module = semantics_probe
    result, _ = module.probe_loss_and_gradients(TinyModel(), torch.device("cpu"), 4)
    # When: a single measured mask/loss invariant is broken.
    altered = result.model_copy(update={field: value})
    # Then: the derived predicate fails, regardless of other successful checks.
    assert not altered.passed
