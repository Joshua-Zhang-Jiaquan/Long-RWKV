from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Literal

from pydantic import BaseModel, ConfigDict
import pytest


V6_ROOT = Path(
    os.environ.get(
        "QUALIFICATION_RELEASE_UNDER_TEST",
        "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
        "nonlatent_iclr_qualification/payloads/"
        "qualification-20260913-06-global-ffaa464d-adapter-v6",
    )
)
PYTHON = Path("/usr/bin/python")

CALLER_SCRIPT = r"""
import json
import os
import torch

from scale.experiments.nonlatent_iclr.full_canvas import CanvasBoundaryError, FullCanvasAdapter
from scale.experiments.nonlatent_iclr.qualification.contracts import QualificationRuntimeError
from scale.experiments.nonlatent_iclr.qualification.model_checks import masked_forward

scenario = os.environ["SCENARIO"]
failure = RuntimeError("model_failure")
calls = []
opened = []
real_open = FullCanvasAdapter.open

def record_open(adapter, input_ids):
    request = real_open(adapter, input_ids)
    opened.append((adapter, request))
    return request

FullCanvasAdapter.open = record_open

class Model:
    def __call__(self, input_ids, *, force_forward=None, z_slots="omitted", state_cache="omitted", use_cache=None):
        calls.append({
            "input_ids": input_ids,
            "force_forward": force_forward,
            "z_slots": z_slots,
            "state_cache": state_cache,
            "use_cache": use_cache,
            "inference": torch.is_inference_mode_enabled(),
        })
        match scenario:
            case "valid":
                return torch.zeros(()).expand(1, 48, 65_536)
            case "shape":
                return torch.zeros(()).expand(1, 47, 65_536)
            case "nonfinite":
                return torch.tensor(float("nan")).expand(1, 48, 65_536)
            case "error":
                raise failure
            case _:
                raise AssertionError(scenario)

model = Model()
outcome = ""
try:
    result = masked_forward(model, torch, torch.device("cpu"))
    assert scenario == "valid"
    assert result == ((1, 48, 65_536), True)
    outcome = "valid"
except QualificationRuntimeError as error:
    assert scenario in {"shape", "nonfinite"}
    outcome = str(error)
except RuntimeError as error:
    assert scenario == "error"
    assert error is failure
    outcome = "model_failure_identity_preserved"

assert len(calls) == 1
assert len(opened) == 1
call = calls[0]
ids = call["input_ids"]
assert ids.shape == (1, 48)
assert ids.dtype == torch.int64
assert ids.device.type == "cpu"
assert bool((ids[:, 8:12] == 65_535).all())
assert bool(((ids >= 0) & (ids <= 65_535)).all())
assert (call["force_forward"], call["z_slots"], call["state_cache"], call["use_cache"]) == (False, None, None, False)
assert call["inference"]
adapter, request = opened[0]
try:
    adapter.forward(request)
except CanvasBoundaryError as error:
    assert error.code == "session_closed"
else:
    raise AssertionError("adapter session remained active")

print(json.dumps({"outcome": outcome, "retired": True, "explicit_flags": True, "inference": True}, sort_keys=True))
"""

PARAMETER_SCRIPT = r"""
import json
from dataclasses import dataclass

from pydantic import ValidationError
from scale.experiments.nonlatent_iclr.qualification.model_checks import parameter_evidence

@dataclass(frozen=True, slots=True)
class Parameter:
    elements: int
    def numel(self):
        return self.elements

@dataclass(frozen=True, slots=True)
class Model:
    parameters: tuple
    def named_parameters(self):
        return iter(self.parameters)

groups = (
    ("layers.attn_fwd", 829, 934_049_280),
    ("layers.attn_bwd", 829, 934_049_280),
    ("layers.fuse_proj", 63, 209_797_119),
    ("shared", 230, 2_013_685_760),
    ("loop", 1, 1),
)
parameters = tuple(
    (f"{prefix}.{index}", Parameter(total - count + 1 if index == 0 else 1))
    for prefix, count, total in groups
    for index in range(count)
) + (("layers.fuse_bias", Parameter(1)),)
evidence = parameter_evidence(Model(parameters))
assert evidence.total_tensors == 1_953
assert evidence.total_numel == 4_091_581_441
try:
    parameter_evidence(Model(parameters[1:]))
except ValidationError as error:
    locations = sorted(item["loc"][0] for item in error.errors())
    kinds = sorted(set(item["type"] for item in error.errors()))
else:
    raise AssertionError("invalid observed counts were accepted")
print(json.dumps({"total_tensors": evidence.total_tensors, "total_numel": evidence.total_numel, "locations": locations, "kinds": kinds}, sort_keys=True))
"""


class CallerObservation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    outcome: str
    retired: bool
    explicit_flags: bool
    inference: bool


class ParameterObservation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    total_tensors: int
    total_numel: int
    locations: tuple[str, ...]
    kinds: tuple[str, ...]


def run_candidate(
    tmp_path: Path, script: str, scenario: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {
        "LC_ALL": "C",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(V6_ROOT),
        "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
    }
    if scenario is not None:
        environment["SCENARIO"] = scenario
    return subprocess.run(
        (str(PYTHON), "-P", "-c", script),
        cwd=V6_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("scenario", "expected"),
    (
        ("valid", "valid"),
        ("shape", "masked_logits_shape_mismatch:(1, 47, 65536)"),
        ("nonfinite", "masked_logits_non_finite"),
        ("error", "model_failure_identity_preserved"),
    ),
)
def test_candidate_masked_forward_uses_and_retires_real_adapter(
    tmp_path: Path,
    scenario: Literal["valid", "shape", "nonfinite", "error"],
    expected: str,
) -> None:
    # Given: the staged caller and real adapter with a CPU recording callable.
    # When: success or each existing output/model failure path executes.
    result = run_candidate(tmp_path, CALLER_SCRIPT, scenario)

    # Then: explicit full-canvas forwarding occurs once and finally retires the session.
    assert result.returncode == 0, result.stderr
    observation = CallerObservation.model_validate_json(result.stdout)
    assert observation.outcome == expected
    assert observation.retired
    assert observation.explicit_flags
    assert observation.inference


def test_candidate_parameter_evidence_parses_computed_fields_at_boundary(
    tmp_path: Path,
) -> None:
    # Given: lightweight computed observations matching the exact frozen schema.
    # When: valid and changed observations cross the candidate parse boundary.
    result = run_candidate(tmp_path, PARAMETER_SCRIPT)

    # Then: measured values survive and changed counts retain literal rejection.
    assert result.returncode == 0, result.stderr
    observation = ParameterObservation.model_validate_json(result.stdout)
    assert observation.total_tensors == 1_953
    assert observation.total_numel == 4_091_581_441
    assert observation.locations == (
        "forward_attention_numel",
        "forward_attention_tensors",
        "total_numel",
        "total_tensors",
    )
    assert observation.kinds == ("literal_error",)
