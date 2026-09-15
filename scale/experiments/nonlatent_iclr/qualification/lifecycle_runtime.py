"""Loaded-model lifecycle qualification boundary."""

from __future__ import annotations

from pydantic import ValidationError
import torch

from .contracts import QualificationRuntimeError
from .lifecycle import probe_lifecycle
from .lifecycle_binding import LifecycleInvocation
from .lifecycle_evidence import LifecycleConfig, LifecycleEvidence
from .lifecycle_settings import LoadedLifecycleModel, observe_settings
from .lifecycle_sidecar import LifecycleIssue, LifecycleSidecar


def qualify_lifecycle(model: LoadedLifecycleModel, invocation: LifecycleInvocation) -> LifecycleSidecar:
    """Publish observations before rejecting; never load, retune or repair a model."""
    before = after = None
    observations = None
    raw = None
    issue = None
    with torch.inference_mode():
        try:
            before = observe_settings(model)
            if (before.training or before.parameter_devices != (invocation.binding.device,)
                    or str(invocation.device) != invocation.binding.device):
                issue = LifecycleIssue(code="precondition_failed", exception_type="LifecyclePrecondition",
                                       detail="lifecycle_requires_eval_model_on_bound_device")
            else:
                returned = probe_lifecycle(
                    model, LifecycleConfig(vocab_size=invocation.binding.geometry.vocab_size,
                                           device=str(invocation.device)),
                    lambda: torch.cuda.synchronize(invocation.device),
                )
                raw = returned.model_dump_json()
                observations = LifecycleEvidence.model_validate_json(raw)
                raw = None
            after = observe_settings(model)
        except ValidationError as error:
            issue = LifecycleIssue(code="invalid_observation", exception_type=type(error).__name__, detail=str(error))
        except (AssertionError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            issue = LifecycleIssue(code="execution_error", exception_type=type(error).__name__, detail=str(error))
    sidecar = LifecycleSidecar.model_validate_json(LifecycleSidecar(
        binding=invocation.binding, before=before, after=after, observations=observations,
        rejected_observation_json=raw, issue=issue,
    ).model_dump_json())
    sidecar.write_once(invocation.output_dir)
    if not sidecar.passed:
        raise QualificationRuntimeError("lifecycle_qualification_failed")
    return sidecar
