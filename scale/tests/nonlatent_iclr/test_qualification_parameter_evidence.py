from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

from pydantic import ValidationError
import pytest

from scale.experiments.nonlatent_iclr.qualification.model_checks import parameter_evidence


@dataclass(frozen=True, slots=True)
class ObservedParameter:
    elements: int

    def numel(self) -> int:
        return self.elements


@dataclass(frozen=True, slots=True)
class ObservedModel:
    parameters: tuple[tuple[str, ObservedParameter], ...]

    def named_parameters(self) -> Iterator[tuple[str, ObservedParameter]]:
        return iter(self.parameters)


@pytest.fixture
def observed_model() -> ObservedModel:
    # Scalar metadata exercises the real counting loop without allocating weights.
    groups = (
        ("layers.attn_fwd", 829, 934_049_280),
        ("layers.attn_bwd", 829, 934_049_280),
        ("layers.fuse_proj", 63, 209_797_119),
        ("shared", 230, 2_013_685_760),
        ("loop", 1, 1),
    )
    parameters = tuple(
        (f"{prefix}.{index}", ObservedParameter(total - count + 1 if index == 0 else 1))
        for prefix, count, total in groups
        for index in range(count)
    )
    return ObservedModel((*parameters, ("layers.fuse_bias", ObservedParameter(1))))


def test_parameter_evidence_preserves_all_counts_when_observations_valid(
    observed_model: ObservedModel,
) -> None:
    # Given: dynamic observations matching every independently fixed schema count.
    # When: the actual function classifies and totals named parameters.
    evidence = parameter_evidence(observed_model)
    # Then: all measured fields, including both fusion naming branches, survive.
    assert evidence.model_dump() == {
        "total_tensors": 1_953,
        "total_numel": 4_091_581_441,
        "forward_attention_tensors": 829,
        "forward_attention_numel": 934_049_280,
        "backward_attention_tensors": 829,
        "backward_attention_numel": 934_049_280,
        "fusion_tensors": 64,
        "fusion_numel": 209_797_120,
        "shared_tensors": 230,
        "shared_numel": 2_013_685_760,
        "loop_tensors": 1,
        "loop_numel": 1,
    }


@pytest.mark.parametrize("index,category", [
    (0, "forward_attention"), (829, "backward_attention"),
    (1658, "fusion"), (1721, "shared"), (1951, "loop"),
    (1952, "fusion"),
])
def test_parameter_evidence_rejects_when_observed_tensor_count_wrong(
    observed_model: ObservedModel, index: int, category: str,
) -> None:
    # Given: one observed tensor removed from each category or fusion spelling.
    parameters = observed_model.parameters
    invalid = ObservedModel(parameters[:index] + parameters[index + 1:])
    # When: the actual measured inventory is parsed.
    with pytest.raises(ValidationError) as error:
        parameter_evidence(invalid)
    # Then: exact category and overall counts reject rather than being replaced.
    locations = {entry["loc"] for entry in error.value.errors()}
    assert locations == {
        ("total_tensors",), ("total_numel",),
        (f"{category}_tensors",), (f"{category}_numel",),
    }
    assert {entry["type"] for entry in error.value.errors()} == {"literal_error"}


@pytest.mark.parametrize("index,category", [
    (0, "forward_attention"), (829, "backward_attention"),
    (1658, "fusion"), (1721, "shared"), (1951, "loop"),
    (1952, "fusion"),
])
def test_parameter_evidence_rejects_when_observed_numel_wrong(
    observed_model: ObservedModel, index: int, category: str,
) -> None:
    # Given: tensor count unchanged but an actual numel observation differs by one.
    parameters = observed_model.parameters
    name, parameter = parameters[index]
    changed = (name, replace(parameter, elements=parameter.elements + 1))
    invalid = ObservedModel(parameters[:index] + (changed,) + parameters[index + 1:])
    # When: the actual function aggregates the changed observations.
    with pytest.raises(ValidationError) as error:
        parameter_evidence(invalid)
    # Then: both overall and category element constraints remain exact.
    locations = {entry["loc"] for entry in error.value.errors()}
    assert locations == {("total_numel",), (f"{category}_numel",)}
    assert {entry["type"] for entry in error.value.errors()} == {"literal_error"}
