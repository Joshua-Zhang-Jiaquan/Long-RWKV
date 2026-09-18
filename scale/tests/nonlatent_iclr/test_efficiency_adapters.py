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
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pytest

from scale.experiments.nonlatent_iclr.efficiency import adapters as efficiency

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
MODULE_PATH: Final = Path(efficiency.__file__)
#: Registry keys that have their own load path and therefore never reach the generic hub loader,
#: read from the module's own ``*_KEY`` constants so a key that gains a loader (BiRWKV, then
#: mamba2, then LLaDA) drops out of :data:`HUB_KEYS` at the same moment its dispatch is added --
#: rather than leaving a stale assertion that it still goes through ``AutoModelForCausalLM``.
_SPECIAL_LOADER_KEYS: Final = frozenset(
    value for name, value in vars(efficiency).items()
    if name.endswith("_KEY") and isinstance(value, str))

#: Keys whose weights resolve under the *hub* models root AND reach the generic
#: ``AutoModelForCausalLM`` loader, so a throwaway root can stand in for the cluster.
HUB_KEYS: Final = tuple(
    sorted(key for key in efficiency.REGISTRY if key not in _SPECIAL_LOADER_KEYS))


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


def test_the_generic_hub_models_still_take_the_hub_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding loaders for BiRWKV and mamba2 must not change where the transformers models resolve.

    Every key that has its OWN loader is excluded from :data:`HUB_KEYS` by construction, so this
    test would go vacuous if that derivation ever emptied it; the three stock transformers models
    are therefore asserted present. The hub loader is a fake ``transformers`` recording every path
    it was asked to build, so a dispatch that routed one of these keys through a specialised loader
    (or resolved it elsewhere) turns this red rather than silently measuring the wrong thing.
    """
    assert {"llama-3.2-3b", "qwen2.5-3b", "rwkv7-goose-world3-2.9b"} <= set(HUB_KEYS)

    hub_calls: list[str] = []
    _install_fake_transformers(monkeypatch, hub_calls)
    monkeypatch.setattr(
        efficiency, "_load_birwkv_model",
        lambda **kwargs: pytest.fail("the BiRWKV loader must not run for a hub key"))
    monkeypatch.setattr(
        efficiency, "_load_mamba2_model",
        lambda **kwargs: pytest.fail("the mamba2 loader must not run for a hub key"))
    monkeypatch.setattr(
        efficiency, "_load_llada_model",
        lambda **kwargs: pytest.fail("the LLaDA loader must not run for a hub key"))

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


# --------------------------------------------------------------------------
# mamba2: the release config is the mamba_ssm format with no model_type, so the hub loader
# can never build it (commit 5038e1d). The architecture is rebuilt from HF's tensor-for-tensor
# port of the same mixer, against which every checkpoint tensor maps. These tests are CPU-only:
# the config->kwargs translation is a pure function, and the one heavy call is monkeypatched.
# --------------------------------------------------------------------------


def _mamba2_ssm_config(**over: object) -> dict[str, object]:
    """The mamba_ssm-format config the release ships, measured 2026-09-18.

    It names only ``ssm_cfg.layer``: none of the mixer geometry (d_state/headdim/expand/ngroups)
    is in it, which is exactly why the geometry has to come from the checkpoint's tensors.
    """
    config: dict[str, object] = {
        "d_model": 2560, "d_intermediate": 0, "n_layer": 64, "vocab_size": 50277,
        "ssm_cfg": {"layer": "Mamba2"}, "attn_layer_idx": [], "attn_cfg": {},
        "rms_norm": True, "residual_in_fp32": True, "fused_add_norm": True,
        "pad_vocab_size_multiple": 16, "tie_embeddings": True,
    }
    config.update(over)
    return config


def _mamba2_shapes(*, d_model: int = 2560, n_layer: int = 64, vocab: int = 50288,
                   num_heads: int = 80, head_dim: int = 64, state_size: int = 128,
                   n_groups: int = 1, conv_kernel: int = 4) -> dict[str, tuple[int, ...]]:
    """The tensor shapes of the real checkpoint, built from the same identities the loader checks."""
    d_inner = num_heads * head_dim
    group_width = 2 * n_groups * state_size
    conv_dim = d_inner + group_width
    in_proj = 2 * d_inner + group_width + num_heads
    shapes: dict[str, tuple[int, ...]] = {
        "backbone.embedding.weight": (vocab, d_model),
        "backbone.norm_f.weight": (d_model,),
        "lm_head.weight": (vocab, d_model),
    }
    for layer in range(n_layer):
        mixer = f"backbone.layers.{layer}.mixer"
        shapes[f"backbone.layers.{layer}.norm.weight"] = (d_model,)
        shapes[mixer + ".dt_bias"] = (num_heads,)
        shapes[mixer + ".A_log"] = (num_heads,)
        shapes[mixer + ".D"] = (num_heads,)
        shapes[mixer + ".in_proj.weight"] = (in_proj, d_model)
        shapes[mixer + ".conv1d.weight"] = (conv_dim, 1, conv_kernel)
        shapes[mixer + ".conv1d.bias"] = (conv_dim,)
        shapes[mixer + ".norm.weight"] = (d_inner,)
        shapes[mixer + ".out_proj.weight"] = (d_model, d_inner)
    return shapes


def _mamba2_model_dir(tmp_path: Path) -> Path:
    """A models root with the mamba2 directory and its config, enough for ``build`` to pass."""
    root = _materialized_root(tmp_path, "mamba2-2.7b")
    model_dir = root / efficiency.REGISTRY["mamba2-2.7b"].path
    (model_dir / efficiency.MAMBA2_CONFIG_FILE).write_text(
        json.dumps(_mamba2_ssm_config()), encoding="utf-8")
    return root


def _install_fake_mamba2(
        monkeypatch: pytest.MonkeyPatch, *, missing: Iterable[str] = (),
        unexpected: Iterable[str] = (), calls: dict[str, object] | None = None) -> None:
    """A ``transformers`` whose Mamba2 builder records what it was handed, with no weights."""
    module = types.ModuleType("transformers")

    class _FakeMamba2ForCausalLM:
        def __init__(self, config: object) -> None:
            if calls is not None:
                calls["config"] = config

        def to(self, *args: object, **kwargs: object) -> _FakeMamba2ForCausalLM:
            return self

        def eval(self) -> _FakeMamba2ForCausalLM:
            return self

        def load_state_dict(self, state_dict: dict[str, object],
                            strict: bool = True) -> tuple[list[str], list[str]]:
            if calls is not None:
                calls["keys"] = set(state_dict)
                calls["strict"] = strict
            return list(missing), list(unexpected)

    module.Mamba2Config = lambda **kwargs: types.SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.Mamba2ForCausalLM = _FakeMamba2ForCausalLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_mamba2_declares_a_constant_state_space_autoregressive_row() -> None:
    """The row's identity: STATE_SPACE, AUTOREGRESSIVE, and one forward per answer.

    A state-space model has no KV cache; scoring its state as growing, or its answer as more than
    one forward, would put a wrong row in the one table whose whole point is constant-versus-linear
    and forward-count.
    """
    spec = efficiency.REGISTRY["mamba2-2.7b"]
    assert spec.family == efficiency.STATE_SPACE
    assert spec.objective == efficiency.AUTOREGRESSIVE
    assert spec.nfe == efficiency.NFE_AUTOREGRESSIVE == 1


def test_mamba2_analytic_state_is_constant_and_rectangular(tmp_path: Path) -> None:
    """The state is constant in context AND is the rectangular head_dim x state_dim form.

    Constant because the SSD state does not grow; rectangular because collapsing state_dim onto
    head_dim (squaring it) is the specific mistake the RecurrentGeometry's two axes exist to
    prevent. The closed form is asserted as an exact product so either error moves this number.
    """
    adapter = efficiency.build("mamba2-2.7b", models_root=str(_mamba2_model_dir(tmp_path)))
    short, long = 4096, 65536
    assert adapter.analytic_state_bytes(long) == adapter.analytic_state_bytes(short) > 0
    # directions(1) x layers(64) x heads(80) x head_dim(64) x state_dim(128) x 2 bytes.
    assert adapter.analytic_state_bytes(short) == 1 * 64 * 80 * 64 * 128 * 2


def test_mamba2_ssm_config_and_shapes_translate_to_the_recorded_hf_geometry() -> None:
    """The mamba_ssm config plus the measured shapes give exactly the HF Mamba2 geometry.

    The vocabulary is the tensor's padded row count (50288), NOT the config's 50277: HF's
    Mamba2Config does not pad, so taking the config's number would build an embedding the
    checkpoint's tensor cannot load into. The mixer geometry comes from the shapes because the
    config does not carry it.
    """
    kwargs = efficiency._mamba2_hf_kwargs(
        _mamba2_ssm_config(), _mamba2_shapes(),
        geometry=efficiency._GEOMETRY["mamba2-2.7b"])
    assert kwargs == {
        "vocab_size": 50288, "hidden_size": 2560, "num_hidden_layers": 64, "num_heads": 80,
        "head_dim": 64, "state_size": 128, "n_groups": 1, "expand": 2, "conv_kernel": 4,
        "rms_norm": True, "residual_in_fp32": True, "tie_word_embeddings": True,
    }


def test_mamba2_refuses_shapes_that_contradict_the_recorded_geometry() -> None:
    """The geometry derived from the tensors is checked against the recorded one.

    The analytic state row and the built model must describe the SAME model; a geometry whose head
    count disagrees with the checkpoint's tensors is the concrete drift this guard catches, and it
    must refuse rather than measure a model different from the one in the table.
    """
    wrong = efficiency.RecurrentGeometry(
        n_layers=64, n_heads=64, head_dim=64, state_dim=128, directions=1, dtype_bytes=2)
    with pytest.raises(efficiency.AdapterRefusal, match="recorded geometry"):
        efficiency._mamba2_hf_kwargs(
            _mamba2_ssm_config(), _mamba2_shapes(), geometry=wrong)


def test_mamba2_refuses_a_hybrid_or_non_mamba2_config() -> None:
    """Only a uniform Mamba2 mixer can be built; anything else refuses rather than being forced.

    A Mamba1 layer, an interleaved attention layer, or a config that is not the mamba_ssm format
    each need a different builder -- forcing any of them would measure the wrong architecture.
    """
    geometry = efficiency._GEOMETRY["mamba2-2.7b"]
    with pytest.raises(efficiency.AdapterRefusal, match="Mamba2"):
        efficiency._mamba2_hf_kwargs(
            _mamba2_ssm_config(ssm_cfg={"layer": "Mamba1"}), _mamba2_shapes(), geometry=geometry)
    with pytest.raises(efficiency.AdapterRefusal, match="hybrid"):
        efficiency._mamba2_hf_kwargs(
            _mamba2_ssm_config(attn_layer_idx=[3]), _mamba2_shapes(), geometry=geometry)
    with pytest.raises(efficiency.AdapterRefusal, match="mamba_ssm"):
        efficiency._mamba2_hf_kwargs(
            {"model_type": "llama", "hidden_size": 4096}, _mamba2_shapes(), geometry=geometry)


def test_mamba2_refuses_a_state_dict_that_is_not_the_zxbcdt_packing() -> None:
    """The in-proj row count pins the z, x, B, C, dt packing; an off-by-one must refuse.

    A checkpoint whose in-proj does not decompose as 2*d_inner + 2*n_groups*d_state + num_heads
    packs its channels in an order this loader does not know, and loading it would silently
    scramble z/x/B/C/dt.
    """
    shapes = _mamba2_shapes()
    shapes["backbone.layers.0.mixer.in_proj.weight"] = (10576 + 1, 2560)
    with pytest.raises(efficiency.AdapterRefusal, match="in-proj"):
        efficiency._mamba2_hf_kwargs(
            _mamba2_ssm_config(), shapes, geometry=efficiency._GEOMETRY["mamba2-2.7b"])


def test_mamba2_refuses_a_layer_count_that_disagrees_with_the_config() -> None:
    """Every layer the config declares must be present in the state dict.

    Building fewer layers than the checkpoint holds would silently drop weights and measure a
    smaller model, so the tensor layer set is required to equal range(n_layer).
    """
    with pytest.raises(efficiency.AdapterRefusal, match="n_layer"):
        efficiency._mamba2_hf_kwargs(
            _mamba2_ssm_config(), _mamba2_shapes(n_layer=8),
            geometry=efficiency._GEOMETRY["mamba2-2.7b"])


def test_mamba2_key_rename_is_the_single_embedding_difference() -> None:
    """The checkpoint and the HF port agree on every name but the token embedding's.

    Asserted from the record so a future edit that adds a second rename (or drops this one) is
    visible; the map must also not collide two checkpoint names onto one HF name.
    """
    assert efficiency.MAMBA2_HF_KEY_RENAME == {
        "backbone.embedding.weight": "backbone.embeddings.weight"}
    shapes = _mamba2_shapes()
    renamed = {efficiency.MAMBA2_HF_KEY_RENAME.get(name, name) for name in shapes}
    assert "backbone.embedding.weight" not in renamed
    assert "backbone.embeddings.weight" in renamed
    assert len(renamed) == len(shapes)


def test_a_mamba2_key_takes_the_mamba2_load_path_not_the_hub(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The release cannot go through the hub loader, so the key must pick the mamba2 path.

    A hub load of this checkpoint raises on the first cell -- "Unrecognized model ... no
    model_type" (commit 5038e1d) -- so a dispatch that fell through to it would measure nothing.
    Both loaders are observable: the mamba2 one is a recording fake, the hub one is a fake
    ``transformers`` whose call list must stay empty.
    """
    root = _mamba2_model_dir(tmp_path)
    model_dir = root / efficiency.REGISTRY["mamba2-2.7b"].path
    sentinel = object()
    seen: dict[str, object] = {}

    def fake_mamba2(**kwargs: object) -> object:
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(efficiency, "_load_mamba2_model", fake_mamba2)
    hub_calls: list[str] = []
    _install_fake_transformers(monkeypatch, hub_calls)

    adapter = efficiency.build("mamba2-2.7b", models_root=str(root))
    loaded = adapter.load("cpu")

    assert seen["ckpt_dir"] == str(model_dir)
    assert seen["device"] == "cpu"
    # The recorded geometry travels into the builder, so the built model and the analytic state
    # row cannot drift apart.
    assert seen["geometry"] is efficiency._GEOMETRY["mamba2-2.7b"]
    assert loaded.model is sentinel
    assert loaded.step is None
    assert hub_calls == []


def test_mamba2_load_renames_the_embedding_and_refuses_an_unexpected_tensor(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader applies the single rename and rejects a checkpoint with a foreign tensor.

    The heavy builder is faked so the assertion is about the CONTRACT: the embedding arrives under
    HF's name, the load is not ``strict`` (so the mismatch is reported with names rather than as a
    raw error), and an unexpected tensor -- a checkpoint from another architecture -- refuses.
    """
    import torch

    root = _mamba2_model_dir(tmp_path)
    model_dir = root / efficiency.REGISTRY["mamba2-2.7b"].path
    (model_dir / efficiency.MAMBA2_WEIGHTS_FILE).write_bytes(b"stand-in for the 5.4 GB state dict")

    state = {"backbone.embedding.weight": torch.zeros(2, 2), "lm_head.weight": torch.zeros(2, 2)}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: dict(state))
    # The geometry translation is exercised by its own tests; stub it so this test isolates the
    # rename and the key check.
    monkeypatch.setattr(efficiency, "_mamba2_hf_kwargs", lambda *args, **kwargs: {})

    calls: dict[str, object] = {}
    _install_fake_mamba2(monkeypatch, calls=calls)
    efficiency._load_mamba2_model(
        ckpt_dir=str(model_dir), device="cpu", geometry=efficiency._GEOMETRY["mamba2-2.7b"])
    assert calls["keys"] == {"backbone.embeddings.weight", "lm_head.weight"}
    assert calls["strict"] is False

    _install_fake_mamba2(monkeypatch, unexpected=["foreign.weight"])
    with pytest.raises(efficiency.AdapterRefusal, match="mismatch"):
        efficiency._load_mamba2_model(
            ckpt_dir=str(model_dir), device="cpu", geometry=efficiency._GEOMETRY["mamba2-2.7b"])


def test_a_mamba2_without_transformers_refuses_naming_the_runtime(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing builder is a named refusal, not a raw ImportError a caller cannot classify.

    ``transformers`` supplies the only Mamba2ForCausalLM available (mamba_ssm is not installed), so
    its absence is the obstacle the refusal must name rather than letting a bare ImportError escape
    the adapter interface.
    """
    root = _mamba2_model_dir(tmp_path)
    adapter = efficiency.build("mamba2-2.7b", models_root=str(root))
    # Force the real loader's ``from transformers import ...`` to fail.
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(efficiency.AdapterRefusal) as caught:
        adapter.load("cpu")
    message = str(caught.value)
    # The dedicated runtime clause, not merely the generic wrapper: the wrapper would catch the
    # ImportError too and its message mentions the module only incidentally, so the builder's name
    # is asserted to pin the branch that classifies the fault.
    assert "transformers" in message
    assert "Mamba2ForCausalLM" in message


def test_a_mamba2_without_its_config_refuses_naming_the_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint directory without the mamba_ssm config cannot have its geometry read.

    The absence refuses by path -- "wrong model_dir" and "not downloaded" are different faults --
    and does so before the heavy loader is reached, so no weight file is touched.
    """
    root = _materialized_root(tmp_path, "mamba2-2.7b")  # the directory, but no config.json
    adapter = efficiency.build("mamba2-2.7b", models_root=str(root))
    touched: list[dict[str, object]] = []
    monkeypatch.setattr(
        efficiency, "_load_mamba2_model", lambda **kwargs: touched.append(kwargs) or object())
    missing = root / efficiency.REGISTRY["mamba2-2.7b"].path / efficiency.MAMBA2_CONFIG_FILE

    with pytest.raises(efficiency.AdapterRefusal) as caught:
        adapter.load("cpu")
    assert str(missing) in str(caught.value)
    assert touched == []


def test_a_mamba2_declared_with_an_attention_geometry_refuses_rather_than_caching(
        tmp_path: Path) -> None:
    """A state-space row built with a KV-cache geometry must refuse, not silently grow.

    The row's whole claim is a CONSTANT state; an attention geometry would turn ``analytic_state_bytes``
    linear and describe a model that does not exist, so the loader refuses before touching weights.
    The adapter is constructed directly because ``build`` always supplies the recorded geometry.
    """
    spec = efficiency.REGISTRY["mamba2-2.7b"]
    model_dir = _mamba2_model_dir(tmp_path) / spec.path
    adapter = efficiency.CheckpointAdapter(
        spec=spec,
        geometry=efficiency.AttentionGeometry(
            n_layers=64, n_heads=80, hidden_size=2560, dtype_bytes=2),
        resolved_path=model_dir,
    )
    with pytest.raises(efficiency.AdapterRefusal, match="recurrent"):
        adapter.load("cpu")


def test_a_mamba2_loader_failure_is_wrapped_with_the_checkpoint_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A load that dies in the builder surfaces as a named refusal naming the checkpoint.

    The raw exception is kept in the message so the defect stays diagnosable; a refusal the loader
    itself already made passes through unwrapped rather than being buried in a generic wrapper.
    """
    root = _mamba2_model_dir(tmp_path)
    model_dir = root / efficiency.REGISTRY["mamba2-2.7b"].path
    adapter = efficiency.build("mamba2-2.7b", models_root=str(root))

    def boom(**kwargs: object) -> object:
        raise RuntimeError("device-side assert")

    monkeypatch.setattr(efficiency, "_load_mamba2_model", boom)
    with pytest.raises(efficiency.AdapterRefusal) as caught:
        adapter.load("cpu")
    assert "device-side assert" in str(caught.value)
    assert str(model_dir) in str(caught.value)

    def refuse(**kwargs: object) -> object:
        raise efficiency.AdapterRefusal("geometry contradicts the checkpoint")

    monkeypatch.setattr(efficiency, "_load_mamba2_model", refuse)
    with pytest.raises(efficiency.AdapterRefusal) as caught2:
        adapter.load("cpu")
    assert str(caught2.value) == "geometry contradicts the checkpoint"


# --------------------------------------------------------------------------
# LLaDA: the checkpoint ships its own modeling code (``auto_map`` -> modeling_llada.LLaDAModelLM)
# written for transformers 4.x, which the installed 5.3.0 cannot finish loading -- it raises
# ``all_tied_weights_keys`` AFTER every tensor has been read. The loader restores the 5.x load
# contract and then REFUSES unless the load reports every tensor mapped, so a shim that merely
# silenced the crash cannot put a half-loaded model into the table. CPU-only: the checkpoint's
# remote base class and the tensor report are fakes, so no 8 GB of weights is ever read.
# --------------------------------------------------------------------------


def _fake_llada_base(*, loading_info: dict[str, object]) -> tuple[type, dict[str, object]]:
    """A stand-in for the 4.x-era ``LLaDAModelLM``: no ``all_tied_weights_keys``, a no-arg ``tie_weights``.

    This is the shape the real remote class has under transformers 5.x -- ``post_init`` is never run
    for it and its ``tie_weights`` predates the two-argument signature -- so a compat wrapper that
    does not restore both fails here exactly as the real one did at load.

    ``from_pretrained`` CONSTRUCTS the model through ``cls``, as the real classmethod does, so the
    subclass's ``__init__`` -- and therefore ``post_init`` -- runs on the loader's path too. A fake
    that returned a pre-built stand-in would exercise the wrapper only in the direct compat test and
    leave the wiring the loader actually depends on untested end to end, which is the coverage a
    test that cannot go red claims and does not have. ``loading_info`` is returned unchanged and the
    call is recorded, so the clean-load check is judged against the report the test supplied.
    """
    records: dict[str, object] = {"post_init": 0, "tie_weights": 0, "from_pretrained": []}

    class _FakeBase:
        def __init__(self, config: object, *args: object, **kwargs: object) -> None:
            self.config = config
            self.device: object | None = None
            self.eval_called = False

        def post_init(self) -> None:
            records["post_init"] = int(records["post_init"]) + 1
            self.all_tied_weights_keys = {}
            # 5.x calls tie_weights(recompute_mapping=False); the 4.x body takes no arguments.
            self.tie_weights(recompute_mapping=False)

        def tie_weights(self, **kwargs: object) -> None:  # the 4.x no-arg body the wrapper widens
            records["tie_weights"] = int(records["tie_weights"]) + 1

        def to(self, device: object) -> _FakeBase:
            self.device = device
            return self

        def eval(self) -> _FakeBase:
            self.eval_called = True
            return self

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> tuple[object, dict[str, object]]:
            cast_records = records["from_pretrained"]
            assert isinstance(cast_records, list)
            cast_records.append((path, kwargs))
            # Construct through cls, as the real from_pretrained does, so the compat subclass's
            # __init__ runs post_init here too; a stand-in object returned around it would hide a
            # wrapper that forgot post_init.
            return cls(types.SimpleNamespace()), dict(loading_info)

    return _FakeBase, records


_CLEAN_LLADA_REPORT: Final = {
    "missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set()}


def test_an_llada_key_takes_the_llada_load_path_not_the_hub(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 4.x remote code cannot be finished by the hub loader, so the key must pick the LLaDA path.

    The unmodified class raises ``all_tied_weights_keys`` while ``from_pretrained`` finalizes the load,
    so a dispatch that fell through to the generic hub path would measure nothing. Both loaders are
    observable: the LLaDA one is a recording fake, the hub one a fake ``transformers`` whose call list
    must stay empty. The composable ``*_KEY`` derivation keeps LLaDA out of :data:`HUB_KEYS`.
    """
    assert efficiency.LLADA_KEY not in HUB_KEYS
    root = _materialized_root(tmp_path, efficiency.LLADA_KEY)
    model_dir = root / efficiency.REGISTRY[efficiency.LLADA_KEY].path
    sentinel = object()
    seen: dict[str, object] = {}

    def fake_llada(**kwargs: object) -> object:
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(efficiency, "_load_llada_model", fake_llada)
    hub_calls: list[str] = []
    _install_fake_transformers(monkeypatch, hub_calls)

    adapter = efficiency.build(efficiency.LLADA_KEY, models_root=str(root))
    loaded = adapter.load("cpu")

    assert seen["model_dir"] == str(model_dir)
    assert seen["device"] == "cpu"
    assert loaded.model is sentinel
    assert loaded.step is None
    assert hub_calls == []


def test_the_llada_load_refuses_a_tensor_report_that_is_not_clean() -> None:
    """A load that could not map every tensor must refuse -- a half-loaded model is a wrong number.

    ``from_pretrained`` reports missing, unexpected and shape-mismatched tensors. Past the declared
    allowlists any of the three means the shim silenced a crash without making the 4.x code load, so
    the load is refused and the launcher records a named hole instead of a plausible-looking row.
    """
    # Given: a clean report, so the refusals below are not vacuous.
    efficiency._require_clean_llada_load(efficiency.LLADA_KEY, dict(_CLEAN_LLADA_REPORT))
    # When/Then: each nonzero class refuses and names the offending tensor.
    with pytest.raises(efficiency.AdapterRefusal) as missing:
        efficiency._require_clean_llada_load(
            efficiency.LLADA_KEY,
            {**_CLEAN_LLADA_REPORT, "missing_keys": {"model.transformer.wte.weight"}})
    assert "model.transformer.wte.weight" in str(missing.value)
    assert "1 missing" in str(missing.value)
    with pytest.raises(efficiency.AdapterRefusal) as unexpected:
        efficiency._require_clean_llada_load(
            efficiency.LLADA_KEY,
            {**_CLEAN_LLADA_REPORT, "unexpected_keys": {"foreign.weight"}})
    assert "foreign.weight" in str(unexpected.value)
    with pytest.raises(efficiency.AdapterRefusal) as mismatched:
        efficiency._require_clean_llada_load(
            efficiency.LLADA_KEY,
            {**_CLEAN_LLADA_REPORT,
             "mismatched_keys": {("model.transformer.ff_out.weight", (1, 2), (3, 4))}})
    assert "model.transformer.ff_out.weight" in str(mismatched.value)


def test_the_llada_base_class_resolves_through_the_checkpoints_own_auto_map(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The model built is the CHECKPOINT's code, named by its ``auto_map`` and resolved against its dir.

    ``_llada_base_model_class`` must hand transformers' dynamic-module resolver the checkpoint's own
    ``AutoModelForCausalLM`` entry (``modeling_llada.LLaDAModelLM``) together with the checkpoint
    directory: resolving any other reference, or the right reference against the wrong directory,
    builds a stock architecture or an unrelated module, and the clean-load refusal downstream would
    then be judging the wrong model. The resolver is faked, so this exercises the contract on CPU
    without importing the remote code -- the import failure it could otherwise raise is classified by
    :meth:`CheckpointAdapter._load_llada`.
    """
    sentinel = object()
    seen: dict[str, object] = {}

    dynamic = types.ModuleType("transformers.dynamic_module_utils")

    def fake_get_class_from_dynamic_module(
            reference: str, model_dir: str, **kwargs: object) -> object:
        seen["reference"] = reference
        seen["model_dir"] = model_dir
        return sentinel

    dynamic.get_class_from_dynamic_module = fake_get_class_from_dynamic_module  # type: ignore[attr-defined]
    transformers = types.ModuleType("transformers")
    transformers.dynamic_module_utils = dynamic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dynamic)

    model_dir = tmp_path / "LLaDA-8B-Base"
    model_dir.mkdir()
    resolved = efficiency._llada_base_model_class(str(model_dir))

    assert resolved is sentinel
    # Pinned to the literal the checkpoint's config.json declares under ``auto_map.AutoModelForCausalLM``,
    # so a mistyped constant is caught here rather than surviving to a real 8 B load.
    assert seen["reference"] == efficiency.LLADA_MODEL_REFERENCE == "modeling_llada.LLaDAModelLM"
    assert seen["model_dir"] == str(model_dir)


def test_the_llada_compat_subclass_restores_the_transformers5_contract() -> None:
    """The wrapper must supply the two pieces of the 5.x contract the 4.x class predates.

    Under 5.3.0 the checkpoint's class has no ``all_tied_weights_keys`` (set by ``post_init``, which it
    never runs) and its ``tie_weights`` takes no arguments (5.x passes two). The wrapper calls
    ``post_init`` after construction and widens ``tie_weights``, delegating to the checkpoint's own
    body so its config-driven tying is preserved.
    """
    base, records = _fake_llada_base(loading_info=dict(_CLEAN_LLADA_REPORT))
    # Given: the 4.x-shaped base really lacks the attribute the 5.x load reaches for -- otherwise the
    # assertions below would pass for a subclass that changed nothing.
    assert "all_tied_weights_keys" not in vars(base)
    # When: a model is built through the compat wrapper.
    compat = efficiency._llada_transformers5_compat_class(base)
    instance = compat(config=object())
    # Then: post_init ran, setting the attribute, and delegated to the 4.x tying body...
    assert records["post_init"] == 1
    assert instance.all_tied_weights_keys == {}
    assert records["tie_weights"] == 1
    # ...and tie_weights now accepts 5.x's two arguments, delegating instead of raising TypeError.
    instance.tie_weights(missing_keys={"x"}, recompute_mapping=False)
    assert records["tie_weights"] == 2


def test_the_llada_loader_refuses_a_dirty_report_and_returns_a_clean_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader must inspect the tensor report, refuse a dirty one, and complete a clean one.

    This is the wiring the whole shim rests on: ``output_loading_info=True`` is requested so the report
    can be judged, a report with an unmapped tensor refuses before the model is moved, and a clean
    report yields a model whose dropped ``use_cache`` flag has been restored from the checkpoint's own
    config so the first forward can run.

    The fake ``from_pretrained`` CONSTRUCTS the model through the class the loader resolved -- the
    compat subclass -- so this test observes ``post_init`` running on the loader's own path, not only
    when the compat test instantiates the subclass directly. A wrapper whose ``__init__`` forgot
    ``post_init`` leaves the returned model without ``all_tied_weights_keys``, which is the exact
    attribute the 5.x finalize step reaches for, and turns this red.
    """
    import torch

    model_dir = tmp_path / "LLaDA-8B-Base"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"use_cache": False}), encoding="utf-8")

    # Given: a load that reports a tensor it could not map.
    dirty_base, _ = _fake_llada_base(
        loading_info={"missing_keys": {"model.transformer.wte.weight"},
                      "unexpected_keys": set(), "mismatched_keys": set()})
    monkeypatch.setattr(efficiency, "_llada_base_model_class", lambda _dir: dirty_base)
    with pytest.raises(efficiency.AdapterRefusal) as caught:
        efficiency._load_llada_model(model_dir=str(model_dir), device="cpu")
    assert "model.transformer.wte.weight" in str(caught.value)

    # Given: a load that mapped every tensor.
    clean_base, records = _fake_llada_base(loading_info=dict(_CLEAN_LLADA_REPORT))
    monkeypatch.setattr(efficiency, "_llada_base_model_class", lambda _dir: clean_base)
    result = efficiency._load_llada_model(model_dir=str(model_dir), device="cpu")

    # Then: the report was requested (inspection, not assumption); post_init ran on the CONSTRUCTED
    # instance, restoring the attribute the 5.x load reaches for; the dropped generation flag was
    # restored from the checkpoint's own config; and the model was moved and set to eval.
    requested = records["from_pretrained"]
    assert isinstance(requested, list) and requested[0][1]["output_loading_info"] is True
    assert isinstance(result, clean_base)
    assert records["post_init"] == 1
    assert result.all_tied_weights_keys == {}
    assert result.device == torch.device("cpu")
    assert result.config.use_cache is False
    assert result.eval_called is True

    # The loader threads its ``device`` argument to ``.to`` unchanged, so on the pod the same path
    # moves the model to a CUDA device. ``torch.device`` construction needs no GPU, so the plumbing
    # is testable on CPU; the CUDA execution itself is exercised only by the pod run.
    cuda_result = efficiency._load_llada_model(model_dir=str(model_dir), device="cuda:0")
    assert cuda_result.device == torch.device("cuda:0")


def test_the_llada_loader_refuses_a_checkpoint_that_asks_for_caching(tmp_path: Path) -> None:
    """The restored ``use_cache`` is read from the checkpoint, never guessed; caching is refused.

    transformers 5.x pops ``use_cache`` from the config, so it must be restored for the forward to run.
    A checkpoint declaring caching is refused rather than silently run uncached, because caching changes
    what an answer costs and the table's unit is a fixed number of forwards.
    """
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"use_cache": False}), encoding="utf-8")
    config = types.SimpleNamespace()
    efficiency._restore_dropped_use_cache(config, config_path)
    assert config.use_cache is False

    config_path.write_text(json.dumps({"use_cache": True}), encoding="utf-8")
    with pytest.raises(efficiency.AdapterRefusal, match="use_cache"):
        efficiency._restore_dropped_use_cache(types.SimpleNamespace(), config_path)


def test_the_llada_load_has_no_missing_or_unexpected_allowance() -> None:
    """The declared allowlist is empty: this checkpoint and its remote code are a matched pair.

    A nonempty allowance would be the place a version mismatch hides, so it is asserted empty; a future
    justified exception must edit this test and the constant together, which is the review it deserves.
    """
    assert efficiency.LLADA_MISSING_ALLOWLIST == frozenset()
    assert efficiency.LLADA_UNEXPECTED_ALLOWLIST == frozenset()
