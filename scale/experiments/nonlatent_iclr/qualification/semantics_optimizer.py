"""Execute pinned historical optimizer policy on loaded objects; never step."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from types import ModuleType
from typing import Protocol

import torch

from ..trainer_semantics import SourceSnapshot, execute_segment
from ..trainer_semantics_contract import OptimizerConfig, SourceContractError
from .semantics_evidence import MembershipEvidence, ParameterGroup


class NamedModel(Protocol):
    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]: ...


def audit_membership(
    model: NamedModel, groups: tuple[tuple[torch.Tensor, ...], ...], freeze_backbone: bool,
) -> MembershipEvidence:
    named = tuple(model.named_parameters())
    names = {id(parameter): name for name, parameter in named}
    counts = Counter(id(parameter) for group in groups for parameter in group)
    trainable = {id(parameter) for _, parameter in named if parameter.requires_grad}
    frozen = {id(parameter) for _, parameter in named if not parameter.requires_grad}
    return MembershipEvidence(
        freeze_backbone=freeze_backbone, total_tensors=len(named),
        total_numel=sum(parameter.numel() for _, parameter in named),
        trainable_tensors=len(trainable),
        trainable_numel=sum(parameter.numel() for _, parameter in named if parameter.requires_grad),
        groups=tuple(ParameterGroup(names=tuple(names.get(id(p), "<foreign>") for p in group),
                                    tensors=len(group), numel=sum(p.numel() for p in group)) for group in groups),
        each_trainable_once=set(counts) == trainable and all(count == 1 for count in counts.values()),
        frozen_absent=not frozen.intersection(counts),
        cross_group_disjoint=sum(len({id(p) for p in group}) for group in groups) == len(counts),
        freeze_rule_exact=all(not p.requires_grad for name, p in named if "latent_cond" not in name)
        if freeze_backbone else True,
    )


def probe_membership(model: NamedModel) -> tuple[MembershipEvidence, ...]:
    """Temporarily apply both historical freeze settings, restoring original flags."""
    snapshot = SourceSnapshot.read()
    original = tuple((parameter, parameter.requires_grad) for _, parameter in model.named_parameters())
    observations: list[MembershipEvidence] = []
    try:
        for freeze in (False, True):
            namespace = ModuleType("loaded_historical_optimizer")
            messages: list[str] = []
            namespace.__dict__.update(torch=torch, model=model, args=OptimizerConfig(freeze_backbone=freeze),
                                      latent_on=False, log=messages.append)
            execute_segment(snapshot, "constants", namespace)
            execute_segment(snapshot, "freeze", namespace)
            execute_segment(snapshot, "optimizer", namespace)
            optimizer = namespace.__dict__.get("optimizer")
            if not isinstance(optimizer, torch.optim.AdamW):
                raise SourceContractError("historical optimizer interface mismatch")
            groups: list[tuple[torch.Tensor, ...]] = []
            for group in optimizer.param_groups:
                parameters = group["params"]
                if not isinstance(parameters, list) or not all(isinstance(p, torch.Tensor) for p in parameters):
                    raise SourceContractError("historical optimizer parameter interface mismatch")
                groups.append(tuple(parameters))
            if optimizer.state:
                raise SourceContractError("historical optimizer unexpectedly allocated state")
            observations.append(audit_membership(model, tuple(groups), freeze))
    finally:
        for parameter, requires_grad in original:
            parameter.requires_grad_(requires_grad)
    return tuple(observations)
