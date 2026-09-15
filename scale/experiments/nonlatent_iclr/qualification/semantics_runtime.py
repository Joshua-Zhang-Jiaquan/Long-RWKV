"""Loaded-model semantics boundary: publish partial evidence before rejection."""
from __future__ import annotations

from pydantic import ValidationError

from .contracts import QualificationRuntimeError
from .lifecycle_binding import LifecycleInvocation
from .lifecycle_settings import observe_settings
from .lifecycle_sidecar import LifecycleIssue
from .semantics_optimizer import probe_membership
from .semantics_probe import SemanticsModel, probe_loss_and_gradients
from .semantics_sidecar import SemanticsSidecar


def qualify_semantics(model: SemanticsModel, invocation: LifecycleInvocation) -> SemanticsSidecar:
    before = after = None
    membership = ()
    mask = gradients = issue = None
    try:
        before = observe_settings(model)
        if (before.training or before.inference_mode or not before.grad_enabled
                or before.parameter_devices != (invocation.binding.device,)
                or str(invocation.device) != invocation.binding.device):
            issue = LifecycleIssue(code="precondition_failed", exception_type="SemanticsPrecondition",
                                   detail="semantics_requires_eval_grad_enabled_model_on_bound_device")
        else:
            membership = probe_membership(model)
            if all(item.passed for item in membership):
                mask, gradients = probe_loss_and_gradients(model, invocation.device,
                                                          invocation.binding.geometry.num_hidden_layers)
        after = observe_settings(model)
    except ValidationError as error:
        issue = LifecycleIssue(code="invalid_observation", exception_type=type(error).__name__, detail=str(error))
    except (AssertionError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        issue = LifecycleIssue(code="execution_error", exception_type=type(error).__name__, detail=str(error))
    sidecar = SemanticsSidecar.model_validate_json(SemanticsSidecar(
        binding=invocation.binding, before=before, after=after, membership=membership,
        mask=mask, gradients=gradients, issue=issue,
    ).model_dump_json())
    sidecar.write_once(invocation.output_dir)
    if not sidecar.passed:
        raise QualificationRuntimeError("model_semantics_qualification_failed")
    return sidecar
