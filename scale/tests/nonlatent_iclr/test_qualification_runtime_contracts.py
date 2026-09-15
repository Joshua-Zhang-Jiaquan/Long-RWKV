from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from scale.experiments.nonlatent_iclr.qualification.controls import ProbeConfig
from scale.experiments.nonlatent_iclr.qualification.model_checks import load_checkpoint
from scale.experiments.nonlatent_iclr.qualification.runtime_identity import (
    is_authorized_h100,
)


class _FakeTensor:
    pass


@dataclass(frozen=True, slots=True)
class _IncompatibleKeys:
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


class _FakeTorch:
    Tensor = _FakeTensor

    def __init__(self, state: OrderedDict[str, _FakeTensor]) -> None:
        self.state = state
        self.load_options: tuple[bool, str, bool] | None = None

    def load(
        self,
        _path: Path,
        *,
        weights_only: bool,
        map_location: str,
        mmap: bool,
    ) -> OrderedDict[str, _FakeTensor]:
        self.load_options = (weights_only, map_location, mmap)
        return self.state


class _FakeModel:
    def __init__(self) -> None:
        self.loaded_state: Mapping[str, _FakeTensor] | None = None

    def load_state_dict(
        self, state: Mapping[str, _FakeTensor], *, strict: bool
    ) -> _IncompatibleKeys:
        if not strict:
            raise AssertionError("checkpoint load must remain strict")
        self.loaded_state = state
        return _IncompatibleKeys()


def _probe_config(tmp_path: Path) -> ProbeConfig:
    return ProbeConfig(
        output_dir=tmp_path / "output",
        nonce="a" * 32,
        expected_world_size=8,
        timeout_seconds=1_200,
        memory_fraction=0.90,
        staged_root=tmp_path / "staged",
        checkpoint_dir=tmp_path / "checkpoint",
        hf_model_root=tmp_path / "model",
        manifest_path=tmp_path / "runtime-manifest.json",
        manifest_sha256="b" * 64,
        preflight_receipt=tmp_path / "preflight.json",
        controller_receipt=tmp_path / "controller-receipt.json",
        run_id="qualification-20260912-01-deadbeef",
        controller_receipt_timeout_seconds=60.0,
        nccl_timeout_seconds=120,
    )


def test_checkpoint_loader_accepts_trusted_ordered_tensor_mapping(
    tmp_path: Path,
) -> None:
    # Given: the weights-only representation statically proven for the bound checkpoint.
    state = OrderedDict((f"layer.{index}", _FakeTensor()) for index in range(2))
    torch_module = _FakeTorch(state)
    model = _FakeModel()

    # When: the strict checkpoint boundary consumes that trusted mapping.
    tensor_count = load_checkpoint(model, _probe_config(tmp_path), torch_module)

    # Then: no unsafe fallback or copy is needed and strict state loading is preserved.
    assert tensor_count == 2
    assert torch_module.load_options == (True, "cpu", True)
    assert model.loaded_state is state


def test_hardware_accepts_only_reviewed_sxm80_aliases() -> None:
    # Given: the two exact SXM80 product aliases reviewed for this qualification.
    aliases = ("NVIDIA H100 80GB HBM3", "NVIDIA H100-SXM5-80GB")

    # When: each alias reports the expected physical 80 GB capacity on eight GPUs.
    decisions = tuple(is_authorized_h100(alias, 81_559) for alias in aliases)

    # Then: both exact aliases are accepted without broad H100 substring matching.
    assert decisions == (True, True)


def test_hardware_rejects_h100_nvl_and_pcie_variants() -> None:
    # Given: H100 products outside the authorized SXM80 hardware class.
    forbidden = (
        ("NVIDIA H100 NVL", 95_830),
        ("NVIDIA H100 PCIe", 81_559),
    )

    # When/Then: product-name similarity cannot admit either unauthorized variant.
    for name, memory_mib in forbidden:
        assert not is_authorized_h100(name, memory_mib)


def test_hardware_rejects_sxm_alias_outside_80gb_capacity_range() -> None:
    # Given: an approved product alias paired with a capacity outside its reviewed range.
    name = "NVIDIA H100 80GB HBM3"

    # When/Then: exact naming cannot override the independent capacity boundary.
    assert not is_authorized_h100(name, 95_830)
