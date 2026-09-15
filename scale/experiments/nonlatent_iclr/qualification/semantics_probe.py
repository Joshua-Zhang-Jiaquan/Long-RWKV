"""One real full-canvas backward; historical rank-local loss and stage-A gate."""
from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Protocol

import torch

from ..full_canvas import FullCanvasModel
from ..trainer_semantics import SourceSnapshot, execute_segment, load_functions
from ..trainer_semantics_contract import GateConfig
from .semantics_evidence import GradientEvidence, GradientState, MaskEvidence, ParameterGradient
from .semantics_optimizer import NamedModel


class SemanticsModel(NamedModel, FullCanvasModel[torch.Tensor], Protocol):
    @property
    def training(self) -> bool: ...

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...


def gradient_state(gradient: torch.Tensor | None) -> GradientState:
    if gradient is None:
        return "none"
    if not bool(torch.isfinite(gradient).all().item()):
        return "nonfinite"
    return "nonzero" if bool(torch.count_nonzero(gradient).item()) else "zero"


def name_class(name: str) -> str:
    if "attn_bwd" in name:
        return "backward_attention"
    if "attn_fwd" in name:
        return "forward_attention"
    if "fuse_" in name:
        return "fusion"
    return "loop" if name.startswith("loop.") else "shared"


def probe_loss_and_gradients(
    model: SemanticsModel, device: torch.device, n_layers: int,
) -> tuple[MaskEvidence, GradientEvidence]:
    """Restore pre-existing gradient objects even if forward/backward fails."""
    snapshot = SourceSnapshot.read()
    functions = load_functions(snapshot)
    named = tuple(model.named_parameters())
    original = tuple(parameter.grad for _, parameter in named)
    try:
        for _, parameter in named:
            parameter.grad = None
        with torch.enable_grad():
            ids = torch.arange(1, 49, device=device).view(1, 48)
            ids[0, 0] = 0
            attention = torch.ones_like(ids, dtype=torch.bool)
            attention[0, 1] = False
            generator = torch.Generator(device=device).manual_seed(20260914)
            # Doc lane guarantees a nonempty selection while testing both eligibility exclusions.
            spans = tuple(torch.tensor([[value]], device=device) for value in (0, 0, 24))
            corrupted, mask, bucket, _, _ = functions.sample_corruption(
                ids, attention, 48, 0.0, generator, docgen_prob=1.0,
                doc_spans=(spans[0], spans[1], spans[2]),
            )
            eligible = attention & ids.ne(0)
            expected_mask = eligible & (torch.arange(48, device=device).view(1, 48) < 24)
            logits = model(corrupted, force_forward=False, z_slots=None, state_cache=None, use_cache=False)
            logits.retain_grad()
            loss, _ = functions.masked_diffusion_loss(logits, ids, mask, bucket, 48)
            with torch.no_grad():
                selected = logits.float()[mask]
                selected_mean = (selected.logsumexp(-1) - selected.gather(-1, ids[mask].unsqueeze(-1)).squeeze(-1)).mean()
                # Change only loss inputs outside selection, not visible context in the model forward.
                changed = logits.detach().clone()
                changed[~mask] = 0
                changed_targets = ids.clone()
                changed_targets[~mask] = 123
                unselected_loss, _ = functions.masked_diffusion_loss(changed, changed_targets, mask, bucket, 48)
                empty_loss, _ = functions.masked_diffusion_loss(logits.detach(), ids, torch.zeros_like(mask), bucket, 48)
            loss.backward()
            logit_gradient = logits.grad
            mask_evidence = MaskEvidence.model_validate({
                "device": str(device), "shape": tuple(logits.shape), "selected_tokens": int(mask.sum().item()),
                "eligibility_exact": torch.equal(mask, expected_mask),
                "corruption_exact": torch.equal(corrupted, torch.where(mask, 65535, ids)),
                "logits_finite": bool(torch.isfinite(logits).all().item()), "eligible_tokens": int(eligible.sum().item()),
                "loss": float(loss.item()), "selected_mean": float(selected_mean.item()),
                "unselected_loss": float(unselected_loss.item()), "empty_loss": float(empty_loss.item()),
                "unselected_logit_grad_zero": logit_gradient is not None
                and bool((logit_gradient[~mask] == 0).all().item()),
            })
            before = tuple(gradient_state(parameter.grad) for _, parameter in named)
            config = GateConfig(n_layers=n_layers)
            gated = tuple(functions.grad_frozen(name, 0.0, n_layers, config.stage_a_frac, config.stage_b_frac)
                          for name, _ in named)
            # Version counters detect unintended writes without cloning full-model gradients.
            versions = tuple(None if p.grad is None else p.grad._version for _, p in named)
            namespace = ModuleType("loaded_historical_gate")
            namespace.__dict__.update(model=model, args=config, step=config.step, n_layers=n_layers,
                                      grad_frozen=functions.grad_frozen)
            execute_segment(snapshot, "gate", namespace)
            rows = tuple(ParameterGradient.model_validate({
                "name": name, "name_class": name_class(name), "numel": p.numel(), "requires_grad": p.requires_grad,
                "gated": gate, "before": state, "after": gradient_state(p.grad),
                "unchanged_when_ungated": version == (None if p.grad is None else p.grad._version),
            }) for (name, p), state, gate, version in zip(named, before, gated, versions, strict=True))
            return mask_evidence, GradientEvidence(n_layers=n_layers, parameters=rows)
    finally:
        for (_, parameter), gradient in zip(named, original, strict=True):
            parameter.grad = gradient
