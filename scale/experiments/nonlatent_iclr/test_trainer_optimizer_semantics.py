"""Tiny parameter identity fixtures; no actual-model optimizer qualification."""
import ast
from math import isclose
from types import ModuleType

import pytest
import torch
from torch import nn

from scale.experiments.nonlatent_iclr import trainer_semantics as seam


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix: nn.Parameter = nn.Parameter(torch.ones(2, 2))
        self.bias: nn.Parameter = nn.Parameter(torch.ones(2))
        self.LayerNorm: nn.Parameter = nn.Parameter(torch.ones(2, 2))
        self.frozen: nn.Parameter = nn.Parameter(torch.ones(2), requires_grad=False)
        self.latent_cond: nn.Linear = nn.Linear(2, 2)
        self.alias: nn.Parameter = self.matrix


def tiny_model() -> TinyModel:
    return TinyModel()


@pytest.mark.parametrize("latent", [False, True])
def test_identity_partition_when_parameters_are_tiny(latent: bool) -> None:
    # Given: nonlatent default; True only characterizes the historical branch.
    model = tiny_model()
    config = seam.OptimizerConfig(latent_on=latent, latent_lr_mult=7.0)
    # When
    optimizer = seam.build_optimizer(model, config)
    # Then: identity, not Tensor equality; tied alias appears only once.
    groups = optimizer.param_groups
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert id(model.frozen) not in ids
    expected_decay = {id(model.matrix)}
    expected_clean = {id(model.bias), id(model.LayerNorm)}
    if not latent:
        expected_decay.add(id(model.latent_cond.weight))
        expected_clean.add(id(model.latent_cond.bias))
    assert {id(p) for p in groups[0]["params"]} == expected_decay
    assert {id(p) for p in groups[1]["params"]} == expected_clean
    assert [g["weight_decay"] for g in groups] == ([0.1, 0.0, 0.0] if latent else [0.1, 0.0])
    assert [g["lr_mult"] for g in groups] == ([1.0, 1.0, 7.0] if latent else [1.0, 1.0])
    if latent:
        assert {id(p) for p in groups[2]["params"]} == {id(p) for p in model.latent_cond.parameters()}
    assert all(g["betas"] == (0.9, 0.95) and g["eps"] == 1e-8 for g in groups)


def test_lr_multiplier_when_original_update_runs() -> None:
    # Given
    optimizer = seam.build_optimizer(tiny_model(), seam.OptimizerConfig(latent_on=True, latent_lr_mult=7.0))
    assert [g["lr"] for g in optimizer.param_groups] == [0.01] * 3
    # When
    seam.update_learning_rate(optimizer, 0.002)
    # Then
    assert [g["lr"] for g in optimizer.param_groups] == [0.002, 0.002, 0.014]


def test_empty_groups_when_nonlatent_backbone_is_frozen() -> None:
    # Given: there is no latent_cond in a nonlatent model.
    model = nn.Linear(2, 2)
    # When
    optimizer = seam.build_optimizer(model, seam.OptimizerConfig(freeze_backbone=True))
    # Then: historical constructor accepts two empty groups, not useful training.
    assert all(not p.requires_grad for p in model.parameters())
    assert [len(g["params"]) for g in optimizer.param_groups] == [0, 0]


@pytest.mark.parametrize("frac,expected", [(0.0, [True, True, False, False, True]), (0.2, [True, False, False, False, True]), (0.6, [False] * 5)])
def test_stage_boundaries_when_original_predicate_runs(frac: float, expected: list[bool]) -> None:
    # Given
    functions = seam.load_functions()
    names = ["layers.1.attn.weight", "layers.2.attn.weight", "layers.0.attn_bwd.weight", "layers.0.fuse_gate", "head.weight"]
    # When
    actual = [functions.grad_frozen(name, frac, 4, 0.2, 0.6) for name in names]
    # Then
    assert actual == expected
    assert not functions.grad_frozen("head.weight", frac, 4, 0.2, 0.0)


def test_zero_grad_decay_when_original_gate_precedes_adamw() -> None:
    # Given: one existing gradient and one absent gradient.
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(1.0)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer = seam.build_optimizer(model, seam.OptimizerConfig())
    # When: original application zeros the tensor, then original AdamW steps.
    seam.apply_gradient_gate(model, seam.GateConfig(step=0))
    optimizer.step()
    # Then: zero is not None; decay still moves gated matrix parameters.
    assert model.weight.grad is not None
    assert torch.count_nonzero(model.weight.grad).item() == 0
    assert model.bias.grad is None
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.999))
    assert model.weight in optimizer.state
    assert model.bias not in optimizer.state
    assert isclose(float(optimizer.state[model.weight]["step"]), 1.0)


def test_source_rejected_when_bytes_are_tampered() -> None:
    # Given
    source = seam.SourceSnapshot.read()
    # When / Then: including comment-only changes.
    with pytest.raises(seam.SourceContractError):
        seam.SourceSnapshot(source.content + b"\n# changed\n")


def test_shape_rejected_when_selected_ast_is_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: source hash is intact, parser output is corrupted independently.
    source = seam.SourceSnapshot.read()
    tree = ast.parse(source.content)
    selected = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "grad_frozen")
    selected.body = [ast.Return(value=ast.Constant(value=False))]
    ast.fix_missing_locations(tree)
    # When / Then
    with monkeypatch.context() as scoped:
        scoped.setattr(seam.ast, "parse", lambda *args, **kwargs: tree)
        with pytest.raises(seam.SourceContractError):
            seam.load_functions(source)


def test_binding_rejected_when_required_torch_is_absent() -> None:
    # Given: no ambient builtins or trainer globals may rescue missing bindings.
    namespace = ModuleType("empty")
    # When / Then
    with pytest.raises(seam.SourceContractError):
        seam.execute_segment(seam.SourceSnapshot.read(), "corruption", namespace)


def test_binding_rejected_when_noise_buckets_are_changed() -> None:
    # Given
    source = seam.SourceSnapshot.read()
    namespace = seam._namespace(source)
    namespace.__dict__["NOISE_BUCKETS"] = ((1.0, 1.0, 1.0),)
    # When / Then
    with pytest.raises(seam.SourceContractError):
        seam.execute_segment(source, "corruption", namespace)


@pytest.mark.parametrize("device", ["meta", "cpu"])
def test_fixture_rejected_when_model_exceeds_admission(device: str) -> None:
    # Given
    model = nn.Linear(400, 400, device=device)
    # When / Then: neither fixture needs a GPU.
    with pytest.raises(seam.SourceContractError):
        seam.build_optimizer(model, seam.OptimizerConfig())
