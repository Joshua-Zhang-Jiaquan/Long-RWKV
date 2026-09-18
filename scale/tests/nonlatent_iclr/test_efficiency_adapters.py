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
import types
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


# --------------------------------------------------------------------------
# the measured checkpoint is not the shipped release
# --------------------------------------------------------------------------


def test_the_birwkv_release_can_be_redirected_to_the_measured_checkpoint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The matrix must be able to measure the model the paper is ABOUT.

    The default is the in-repo release path, which is the right default -- a path
    derived from the module's own location cannot be wrong about where the repository
    is. But the checkpoint this program measures is a training endpoint on the cluster,
    and the manuscript labels every number by the STEP that produced it. Without an
    override the matrix refuses to run against the model it is about.
    """
    # Given: a directory standing in for the measured endpoint.
    endpoint = tmp_path / "m4loop_ext_endpoint_ckpt"
    endpoint.mkdir()
    (endpoint / "config.json").write_text("{}")
    monkeypatch.setenv(efficiency.BIRWKV_RELEASE_ENV, str(endpoint))
    # When: it is resolved.
    adapter = efficiency.build(efficiency.BIRWKV_KEY, models_root=str(tmp_path))
    # Then: the override is what was used, and the resolved path is on the adapter so a
    # table entry can be read back to the checkpoint that produced it.
    assert adapter.resolved_path == endpoint


def test_the_override_applies_to_that_key_only(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """A redirect that could move ANY key would make the registry advisory.

    A table built from advisory paths cannot be checked against the models it names,
    because the registry would no longer be the thing that determined them.
    """
    # Given: the override set, and a hub model that exists under models_root.
    monkeypatch.setenv(efficiency.BIRWKV_RELEASE_ENV, str(tmp_path))
    hub = tmp_path / "Llama-3.2-3B"
    hub.mkdir()
    other = next(k for k in efficiency.REGISTRY if k != efficiency.BIRWKV_KEY)
    # When: the other key is resolved.
    adapter = efficiency.build(other, models_root=str(tmp_path))
    # Then: it came from the registry, NOT from the override.
    assert adapter.resolved_path != tmp_path
    assert adapter.resolved_path.name == efficiency.REGISTRY[other].path.split("/")[-1]


def test_without_the_override_the_default_path_is_used(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(efficiency.BIRWKV_RELEASE_ENV, raising=False)
    spec = efficiency.REGISTRY[efficiency.BIRWKV_KEY]
    assert efficiency._absolute_path(spec, str(tmp_path)) == Path(spec.path)


def test_the_absence_refusal_names_the_override_for_that_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The route out must be in the message, not only in the docstring."""
    monkeypatch.delenv(efficiency.BIRWKV_RELEASE_ENV, raising=False)
    with pytest.raises(efficiency.AdapterRefusal) as caught:
        efficiency.build(efficiency.BIRWKV_KEY, models_root=str(tmp_path))
    assert efficiency.BIRWKV_RELEASE_ENV in str(caught.value)


# --------------------------------------------------------------------------
# the BiRWKV endpoint has its own load path: HF geometry dir + flat model.pt
#
# The measured checkpoint is ``model.pt`` plus ``meta.json`` with no config.json, so the hub
# loader can never build it (commit 5038e1d). These tests exercise the BiRWKV path's CONTRACT
# without a 16 GB load: the heavy loader is monkeypatched and the assertions are about the
# dispatch, the named refusals, and the recorded step.
# --------------------------------------------------------------------------


class _FakeHubModel:
    """Stands in for ``AutoModelForCausalLM``'s return so the hub path is observable on CPU."""

    def to(self, device: object) -> _FakeHubModel:
        self.device = device
        return self

    def eval(self) -> _FakeHubModel:
        return self


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace the ``transformers`` module so a hub load records its path and needs no weights."""
    module = types.ModuleType("transformers")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path: str, trust_remote_code: bool = True) -> _FakeHubModel:
            calls.append(path)
            return _FakeHubModel()

    module.AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)


def _birwkv_endpoint(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, step: int = 9500
) -> tuple[Path, Path]:
    """A stand-in endpoint (model.pt + meta.json) and its HF geometry dir, wired through the env.

    No weight bytes are written -- the heavy loader is always monkeypatched in these tests -- but
    ``build`` stats the checkpoint directory and the step is read from ``meta.json``, so both must
    exist for the load path to run at all.
    """
    ckpt = tmp_path / "m4loop_ext_endpoint_ckpt"
    ckpt.mkdir()
    (ckpt / "model.pt").write_bytes(b"flat training state dict stands in for 16 GB")
    (ckpt / "meta.json").write_text(json.dumps({"step": step, "tokens_seen": 9_961_472_000.0}))
    geometry = tmp_path / "RWKV7-Goose-World3-2.9B-HF"
    geometry.mkdir()
    (geometry / "config.json").write_text("{}")
    monkeypatch.setenv(efficiency.BIRWKV_RELEASE_ENV, str(ckpt))
    monkeypatch.setenv(efficiency.BIRWKV_GEOMETRY_ENV, str(geometry))
    return ckpt, geometry


def test_a_birwkv_key_takes_the_birwkv_load_path_not_the_hub(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The endpoint cannot go through the hub loader, so the key must pick the BiRWKV path.

    A hub load of this checkpoint raises on the first cell -- no config.json, "Unrecognized
    model" -- so a dispatch that fell through to it would not merely be slow, it would measure
    nothing. Both loaders are observable here: the BiRWKV one is a recording fake, and the hub
    one is a fake ``transformers`` whose call list must stay empty.
    """
    ckpt, geometry = _birwkv_endpoint(tmp_path, monkeypatch)
    sentinel = object()
    seen: dict[str, object] = {}

    def fake_birwkv(**kwargs: object) -> object:
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(efficiency, "_load_birwkv_model", fake_birwkv)
    hub_calls: list[str] = []
    _install_fake_transformers(monkeypatch, hub_calls)

    adapter = efficiency.build(efficiency.BIRWKV_KEY, models_root=str(tmp_path))
    loaded = adapter.load("cpu")

    # Given/When/Then: the BiRWKV loader got the endpoint and the HF geometry, not the release.
    assert seen["ckpt_dir"] == str(ckpt)
    assert seen["model_dir"] == str(geometry)
    assert seen["device"] == "cpu"
    assert loaded.model is sentinel
    assert hub_calls == []


def test_an_unresolvable_birwkv_geometry_dir_refuses_naming_the_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong geometry dir is a different fault from absent weights, so it refuses by path.

    The checkpoint carries no config.json of its own; without the HF geometry dir the architecture
    cannot be built. The refusal must print the directory it looked for -- and name the env knob
    that redirects it -- so a reader can tell "model_dir is wrong" from "the checkpoint is not
    downloaded", which would otherwise both surface as a generic load failure.
    """
    _birwkv_endpoint(tmp_path, monkeypatch)
    missing = tmp_path / "not-downloaded" / "RWKV7-Goose-World3-2.9B-HF"
    monkeypatch.setenv(efficiency.BIRWKV_GEOMETRY_ENV, str(missing))
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        efficiency, "_load_birwkv_model", lambda **kwargs: calls.append(kwargs) or object())

    adapter = efficiency.build(efficiency.BIRWKV_KEY, models_root=str(tmp_path))
    with pytest.raises(efficiency.AdapterRefusal) as caught:
        adapter.load("cpu")

    message = str(caught.value)
    assert str(missing) in message
    assert efficiency.BIRWKV_GEOMETRY_ENV in message
    # The refusal happens before the heavy loader, so no 16 GB state dict was ever touched.
    assert calls == []


def test_the_checkpoint_step_from_meta_reaches_the_loaded_record(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every efficiency number is labelled by the step that produced it, so the step must travel.

    The endpoint's ``meta.json`` says step 9500; a row that cannot be read back to that step is not
    usable for the manuscript. The step is read from the file (asserted at the file too, so a
    hard-coded constant would not pass) and carried on both the loaded record and the adapter.
    """
    ckpt, _ = _birwkv_endpoint(tmp_path, monkeypatch, step=9500)
    monkeypatch.setattr(efficiency, "_load_birwkv_model", lambda **kwargs: object())

    adapter = efficiency.build(efficiency.BIRWKV_KEY, models_root=str(tmp_path))
    loaded = adapter.load("cpu")

    assert json.loads((ckpt / "meta.json").read_text())["step"] == 9500
    assert loaded.step == 9500
    assert adapter.step == 9500


def test_the_block_size_knob_reaches_the_birwkv_loader(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A block_t_cond checkpoint is only loadable with the training-time block size, unguessed.

    The knob is the route out the refusal names, so it must be read on the load path and passed to
    the loader; an adapter that dropped it would leave a conditioned checkpoint unloadable.
    """
    _birwkv_endpoint(tmp_path, monkeypatch)
    monkeypatch.setenv(efficiency.BIRWKV_BLOCK_SIZE_ENV, "64")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        efficiency, "_load_birwkv_model", lambda **kwargs: seen.update(kwargs) or object())

    adapter = efficiency.build(efficiency.BIRWKV_KEY, models_root=str(tmp_path))
    adapter.load("cpu")

    assert seen["block_size"] == 64


def test_block_t_cond_keys_without_a_block_size_are_refused() -> None:
    """The block size is a training-time contract; guessing it is refused, not defaulted.

    A checkpoint with ``block_t_cond.*`` keys trained with a per-block timestep. Loading without the
    arm attached makes those keys unexpected and kills the job (job-74707291); attaching with a
    guessed size misaligns every block's ``t`` and no downstream metric reveals it. The guard is
    exercised directly, so it must go red if the body is deleted.
    """
    guard = efficiency._refuse_block_t_cond_without_block_size

    # Given: a checkpoint with no timestep arm -- nothing to refuse at any block size.
    guard(["layers.0.attn_fwd.r_proj.weight"], 0)
    # Given: a timestep arm and its block size -- allowed.
    guard(["block_t_cond.trunk.0.weight"], 64)
    # When/Then: a timestep arm with no block size refuses and names the route out.
    with pytest.raises(efficiency.AdapterRefusal) as caught:
        guard(["layers.0.attn_fwd.r_proj.weight", "block_t_cond.trunk.0.weight"], 0)
    assert efficiency.BIRWKV_BLOCK_SIZE_ENV in str(caught.value)


def test_the_five_hub_models_still_take_the_hub_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding the BiRWKV path must not change where the five hub models resolve or load.

    The hub loader is a fake ``transformers`` recording every path it was asked to build, so a
    dispatch that routed a hub key through the BiRWKV loader (or resolved it elsewhere) turns this
    red rather than silently measuring the wrong thing.
    """
    hub_calls: list[str] = []
    _install_fake_transformers(monkeypatch, hub_calls)
    monkeypatch.setattr(
        efficiency, "_load_birwkv_model",
        lambda **kwargs: pytest.fail("the BiRWKV loader must not run for a hub key"))

    root = _materialized_root(tmp_path, *HUB_KEYS)
    for key in HUB_KEYS:
        adapter = efficiency.build(key, models_root=str(root))
        loaded = adapter.load("cpu")
        # A hub checkpoint has no training step of its own, so no step is fabricated for it.
        assert loaded.step is None

    assert set(hub_calls) == {str(root / efficiency.REGISTRY[key].path) for key in HUB_KEYS}


def test_the_birwkv_geometry_default_is_the_recorded_hub_backbone() -> None:
    """The default geometry dir is the recorded 2.9B backbone, not an invented path.

    The adapter's geometry default must point at a path that is actually used elsewhere in the
    repository (``qualification.calibration_arms.LARGE_MODEL_ROOT``), so a reader can verify the
    architecture the denoiser is built from without guessing where it lives.
    """
    from scale.experiments.nonlatent_iclr.qualification.calibration_arms import LARGE_MODEL_ROOT

    assert Path(efficiency.BIRWKV_GEOMETRY_DIR) == LARGE_MODEL_ROOT
