"""Fail-closed extraction and process-isolation controls."""
import ast
import subprocess
import sys
from types import ModuleType

import pytest
import torch

from . import initialization_extract as extract
from .initialization_contract import SourceContractError


@pytest.mark.parametrize("name", ["residual_streams.py", "birwkv7_diffusion.py"])
def test_source_rejected_when_bytes_changed(name: str) -> None:
    # Given: original bytes with even a comment-only modification.
    snapshot = extract.SourceSnapshot.read(name)
    # When / Then: hashing rejects before execution.
    with pytest.raises(SourceContractError, match="SHA-256"):
        extract.SourceSnapshot(name, snapshot.content + b"\n# changed\n")


@pytest.mark.parametrize("region", ["controls", "methods", "loop", "fusion_init", "fusion_mix", "fusion_alpha"])
def test_ast_rejected_when_selected_shape_changed(monkeypatch: pytest.MonkeyPatch, region: str) -> None:
    # Given: valid source bytes but a tampered extraction result.
    snapshot = extract.SourceSnapshot.read(extract.REGIONS[region][0])
    changed = ast.parse("executed = True")
    namespace = ModuleType("negative")
    with monkeypatch.context() as patch:
        patch.setattr(extract, "select_region", lambda snapshot, region: changed)
        # When / Then: independent shape pin rejects execution.
        with pytest.raises(SourceContractError, match="AST shape"):
            extract.execute_region(snapshot, region, namespace)
    assert "executed" not in namespace.__dict__


def test_loop_rejected_before_execution_when_hidden_input_missing() -> None:
    # Given: all explicit inputs except h; h is both read and assigned in this region.
    namespace = extract.control_namespace()
    namespace.__dict__.update(self=None, force_forward=False, film_params=None,
                              xattn_kv=None, state_cache=None, use_cache=False,
                              loop_reps_override=None)
    # When / Then: live-in checking must not confuse an assignment with an input binding.
    with pytest.raises(SourceContractError, match="missing dependencies"):
        extract.execute_region(extract.SourceSnapshot.read("birwkv7_diffusion.py"), "loop", namespace)


def test_dependencies_rejected_when_missing() -> None:
    # Given: no implicit torch or model globals.
    namespace = ModuleType("negative")
    # When / Then: dependency checking fails before creating classes.
    with pytest.raises(SourceContractError, match="missing dependencies"):
        extract.execute_region(extract.SourceSnapshot.read("residual_streams.py"), "controls", namespace)
    assert "BackboneLoopControl" not in namespace.__dict__


def test_binding_rejected_when_torch_replaced() -> None:
    # Given: all names present but torch is a different module.
    namespace = ModuleType("negative")
    namespace.__dict__.update(torch=ModuleType("fake"), nn=torch.nn, TypedTorchModule=torch.nn.Module)
    # When / Then: presence alone cannot satisfy the fixed binding.
    with pytest.raises(SourceContractError, match="changed binding"):
        extract.execute_region(extract.SourceSnapshot.read("residual_streams.py"), "controls", namespace)


@pytest.mark.parametrize("failure", [False, True])
def test_process_entry_when_cpu_probe_requested(failure: bool) -> None:
    # Given: fresh interpreter; no inherited models shim can hide an import.
    code = '''
import sys
import torch
from scale.experiments.nonlatent_iclr.initialization_loop import TinyLoopProbe
before = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith(("models.", "fla", "triton"))}
p = TinyLoopProbe(32)
p.model.attach_backbone_loop((16, 32), 1)
r = p.run(torch.ones(1, 2, 3))
assert (r.outer_invocations, r.base_calls, r.recycled_calls) == (1, 32, 16)
after = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith(("models.", "fla", "triton"))}
assert before == after
'''
    if failure:
        code += "p.model.attach_backbone_loop((16, 32), 1)\n"
    # When: drive the public evidence surface in a subprocess.
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
    # Then: success or an actual historical attachment error, never an import failure.
    assert result.returncode == (1 if failure else 0), result.stderr
    if failure:
        assert "RuntimeError: backbone loop already attached" in result.stderr
