"""Adapter tests: the efficiency registry must be explicit, portable and honest.

The paper's contribution 1 is an efficiency claim, so these tests are written
against the *mechanism* that keeps the comparison honest rather than the happy
path: that ``nfe`` cannot be defaulted, that the analytic state footprint grows
for an attention model and stays constant for a recurrent one, that a missing
checkpoint refuses by path instead of being scored, and that importing the module
is a pure table read that touches neither the disk nor CUDA.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final

import pytest

from scale.experiments.nonlatent_iclr.efficiency import adapters as efficiency

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
MODULE_PATH: Final = Path(efficiency.__file__)
#: Keys whose weights live under the *hub* models root, so a throwaway root can
#: stand in for the cluster.  BiRWKV's path is absolute (it ships in this repo).
HUB_KEYS: Final = ("llama-3.2-3b", "qwen2.5-3b", "llada-8b-base", "mamba2-2.7b",
                   "rwkv7-goose-world3-2.9b")


def _materialized_root(tmp_path: Path, *keys: str) -> Path:
    """A models root holding just enough directory structure for ``build`` to pass.

    ``build`` stats the recorded path, so materializing the model directories is
    what makes the analytic tests runnable without the multi-gigabyte weights.
    """
    root = tmp_path / "models"
    for key in keys:
        (root / efficiency.REGISTRY[key].path).mkdir(parents=True, exist_ok=True)
    return root


def _module_level_imports(path: Path) -> set[str]:
    """Every module imported at top level (import-time), excluding function bodies."""
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_an_incoherent_spec_is_refused_at_construction() -> None:
    """The registry test that used to live here inspected a constant and could not fail.

    It looped over REGISTRY and asserted each entry was well-formed -- but REGISTRY
    is a module constant written by the same hand as the test, so the assertion
    could only ever restate the fixture. A test that cannot go red reports coverage
    it does not have, which is worse than no test.

    What actually guards the invariant is ModelSpec.__post_init__, so that is what
    gets exercised: each of the four ways a spec can be incoherent is constructed
    here and must be refused.
    """
    from scale.experiments.nonlatent_iclr.efficiency.adapters import (
        AUTOREGRESSIVE, FAMILIES, MASKED_DIFFUSION, AdapterRefusal, ModelSpec)

    def spec(**over) -> ModelSpec:
        base = dict(key="k", family="transformer", objective=AUTOREGRESSIVE, nfe=1,
                    params=1, weights_bytes=1, path="/x")
        base.update(over)
        return ModelSpec(**base)

    # Given: a valid spec, so the refusals below are not vacuous.
    assert spec().nfe == 1
    # When/Then: each incoherence is refused rather than stored.
    with pytest.raises(AdapterRefusal, match="must carry a key"):
        spec(key="")
    with pytest.raises(AdapterRefusal, match="family"):
        spec(family="not_a_family")
    with pytest.raises(AdapterRefusal, match="objective"):
        spec(objective="not_an_objective")
    with pytest.raises(AdapterRefusal, match="nfe must be at least 1"):
        spec(nfe=0)
    # and the coherence rule between objective and nfe, which is the one that
    # keeps the efficiency table comparable to itself
    with pytest.raises(AdapterRefusal):
        spec(objective=AUTOREGRESSIVE, nfe=16)
    assert spec(objective=MASKED_DIFFUSION, nfe=16).nfe == 16
    assert FAMILIES  # the vocabulary the refusals are checked against


def test_every_registry_key_has_geometry() -> None:
    # Given: the registry and the geometry table.
    # When / Then: they cover exactly the same keys, so no model can be built
    # without the shape its analytic state bytes are computed from.
    assert set(efficiency._GEOMETRY) == set(efficiency.REGISTRY)


def test_nfe_is_required_and_cannot_contradict_the_objective() -> None:
    # Given: the required fields of a spec, with nfe deliberately omitted.
    required = dict(key="probe", family=efficiency.TRANSFORMER,
                    objective=efficiency.AUTOREGRESSIVE, params=1, weights_bytes=1,
                    path="probe")
    # When / Then: the dataclass refuses to be built without an explicit nfe...
    with pytest.raises(TypeError):
        efficiency.ModelSpec(**required)  # type: ignore[call-arg]
    # ...and refuses an autoregressive model that claims a multi-forward cost,
    with pytest.raises(efficiency.AdapterRefusal):
        efficiency.ModelSpec(**required, nfe=16)
    # ...and a masked-diffusion model that claims a single forward, which is the
    # exact silent default -- diffusion scored as an autoregressive decode -- the
    # spec exists to forbid.
    with pytest.raises(efficiency.AdapterRefusal):
        efficiency.ModelSpec(**{**required, "objective": efficiency.MASKED_DIFFUSION}, nfe=1)


def test_transformer_state_grows_with_context_and_recurrent_state_is_constant(tmp_path: Path) -> None:
    # Given: a transformer adapter and a recurrent adapter over a throwaway root.
    root = _materialized_root(tmp_path, "llama-3.2-3b", "rwkv7-goose-world3-2.9b")
    transformer = efficiency.build("llama-3.2-3b", models_root=str(root))
    recurrent = efficiency.build("rwkv7-goose-world3-2.9b", models_root=str(root))
    short, long = 4096, 65536
    # When / Then: the KV cache grows with context...
    assert transformer.analytic_state_bytes(long) > transformer.analytic_state_bytes(short)
    # ...while the recurrent state is constant, and non-zero (so equality is a
    # property of the formula and not of an accidentally-empty state).
    assert recurrent.analytic_state_bytes(long) == recurrent.analytic_state_bytes(short)
    assert recurrent.analytic_state_bytes(short) > 0


def test_transformer_state_matches_the_closed_form_with_gqa_shrink() -> None:
    # Given: two adapters that differ only in the KV width -- full multi-head and
    # grouped-query -- so the comparison isolates the GQA factor.
    spec = efficiency.ModelSpec(key="probe", family=efficiency.TRANSFORMER,
                                objective=efficiency.AUTOREGRESSIVE, nfe=1, params=1,
                                weights_bytes=1, path="probe")
    heads = efficiency.AttentionGeometry(
        n_layers=28, n_heads=24, hidden_size=3072, dtype_bytes=2, kv_heads=24)
    grouped = efficiency.AttentionGeometry(
        n_layers=28, n_heads=24, hidden_size=3072, dtype_bytes=2, kv_heads=8)
    mha = efficiency.CheckpointAdapter(spec=spec, geometry=heads, resolved_path=Path("probe"))
    gqa = efficiency.CheckpointAdapter(spec=spec, geometry=grouped, resolved_path=Path("probe"))
    context = 4096
    # When / Then: full multi-head equals the heads x head_dim x layers x 2 x
    # context x dtype_bytes form...
    assert mha.analytic_state_bytes(context) == (
        2 * 28 * (3072 * 24 // 24) * context * 2)
    # ...and GQA shrinks it by exactly kv_heads / n_heads.
    assert gqa.analytic_state_bytes(context) == mha.analytic_state_bytes(context) * 8 // 24


def test_build_present_key_returns_adapter_bound_to_the_registry_spec(tmp_path: Path) -> None:
    # Given: a models root in which the declared path exists.
    root = _materialized_root(tmp_path, "llama-3.2-3b")
    # When: the key is built.
    adapter = efficiency.build("llama-3.2-3b", models_root=str(root))
    # Then: the adapter carries the registry's own spec, not a copy that could drift.
    assert adapter.spec is efficiency.REGISTRY["llama-3.2-3b"]


def test_unknown_key_refuses_naming_the_key(tmp_path: Path) -> None:
    # Given: a key the registry does not declare.
    # When / Then: the refusal names the key, so a typo surfaces as the typo and
    # not as a mysterious missing directory.
    with pytest.raises(efficiency.AdapterRefusal) as excinfo:
        efficiency.build("not-a-registered-model", models_root=str(tmp_path))
    assert "not-a-registered-model" in str(excinfo.value)


def test_known_key_with_absent_weights_refuses_naming_the_path(tmp_path: Path) -> None:
    # Given: a known key and an empty models root, so its weights are absent.
    key = "llama-3.2-3b"
    expected = Path(tmp_path) / efficiency.REGISTRY[key].path
    assert not expected.exists()
    # When / Then: the refusal names the path it looked at, so a missing
    # checkpoint cannot be scored as a present one.
    with pytest.raises(efficiency.AdapterRefusal) as excinfo:
        efficiency.build(key, models_root=str(tmp_path))
    message = str(excinfo.value)
    assert str(expected) in message
    assert key in message


def test_module_level_imports_contain_no_torch_or_transformers() -> None:
    # Given: the adapter module's source.
    # When / Then: nothing CUDA-backed is imported at module scope; the heavy
    # imports sit inside the methods that need them.
    imported = _module_level_imports(MODULE_PATH)
    forbidden = {name for name in imported
                 if name.split(".")[0] in {"torch", "transformers", "safetensors"}}
    assert forbidden == set()


def test_import_does_not_touch_the_disk_or_import_cuda() -> None:
    # Given: the roots a lazy import must never reach, and a fresh interpreter in
    # which no earlier test has had the chance to import torch.
    guarded = [efficiency.RECORDED_MODELS_ROOT, str(REPO_ROOT / "release")]
    program = textwrap.dedent(
        """
        import json, sys
        guarded = json.loads(sys.argv[1])
        touched = []
        def audit(event, args):
            if event == "open" and args and isinstance(args[0], (str, bytes)):
                path = args[0].decode() if isinstance(args[0], bytes) else args[0]
                if path.endswith((".safetensors", ".bin", ".pt")) or any(
                        path.startswith(root) for root in guarded):
                    touched.append(path)
        sys.addaudithook(audit)
        import scale.experiments.nonlatent_iclr.efficiency.adapters  # noqa: F401
        heavy = sorted(m for m in ("torch", "transformers", "safetensors") if m in sys.modules)
        print("RESULT" + json.dumps({"touched": touched[:10], "heavy": heavy}))
        """
    ).strip()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")])
    # When: the module is imported in that interpreter.
    completed = subprocess.run(
        [sys.executable, "-c", program, json.dumps(guarded)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, check=False)
    # Then: the import neither read a weight file nor pulled a CUDA-backed package.
    assert completed.returncode == 0, completed.stderr
    line = next(line for line in completed.stdout.splitlines() if line.startswith("RESULT"))
    payload = json.loads(line.removeprefix("RESULT"))
    assert payload["touched"] == []
    assert payload["heavy"] == []
