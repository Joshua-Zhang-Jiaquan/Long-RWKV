"""Launcher tests for the measured efficiency matrix -- the refusals, not the CUDA run.

The grid itself can only be measured on a GPU, so these tests pin the launcher's
*fail-closed* surface: the single-device rule, the unknown-key / absent-path distinction,
the contexts parse, and the mode allowlist.  Each is exercised by running the real script
under bash, so a guard that is silently dropped -- the multi-GPU rule being the one this
suite exists to keep -- turns a green test red rather than passing on the happy path.

The one end-to-end test drives a *fake* scale package (a stand-in adapters registry and
probe_inference) so the grid loop, the per-cell JSON naming, the OOM-keeps-going contract
and the load-failure abort can be observed without a checkpoint or a device.  A python3
stand-in reports a CUDA device only for the launcher's CUDA preflight, so the same run
path is taken as on a GPU host while no actual CUDA is required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.efficiency import adapters as efficiency

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "scale" / "qz" / "launch_efficiency_probe.sh"

#: The environment the launcher reads.  Each `_run` starts from a clean copy of it so a
#: value left in the pytest process cannot make a refusal test pass for the wrong reason.
LAUNCHER_ENV_VARS = (
    "MODE", "NGPUS", "OUTDIR", "MODEL_KEYS", "CONTEXTS", "MODELS_ROOT", "IMPORT_ROOT",
    "DEVICE", "DRY_RUN", "PYTHON_BIN", "FAKE_OOM_AT", "FAKE_LOAD_FAIL",
)


def _run(
    tmp_path: Path, overrides: dict[str, str], *, out_name: str = "out", timeout: int = 240
) -> subprocess.CompletedProcess[str]:
    """Run the launcher once with a controlled environment and OUTDIR."""
    env = dict(os.environ)
    for name in LAUNCHER_ENV_VARS:
        env.pop(name, None)
    env["OUTDIR"] = str(tmp_path / out_name)
    env.update(overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER)], capture_output=True, text=True,
        cwd=str(tmp_path), env=env, timeout=timeout)


def _combined(result: subprocess.CompletedProcess[str]) -> str:
    """Everything the launcher said, so a test cannot miss a message on the wrong stream."""
    return result.stdout + result.stderr


def _materialize_models(tmp_path: Path, *keys: str) -> Path:
    """A models root holding the declared directories of ``keys``, nothing else.

    ``build`` stats the recorded path, so creating the directories is what makes the
    structural preflight pass without the multi-gigabyte weights.
    """
    root = tmp_path / "models"
    for key in keys:
        (root / efficiency.REGISTRY[key].path).mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------
# the guard most likely to be dropped: one GPU, or no measurement
# --------------------------------------------------------------------------


def test_ngpus_other_than_one_is_refused_with_its_reason(tmp_path: Path) -> None:
    # Given: a registry key whose weights are present and a contexts string that parses, so
    # the ONLY reason this run can fail is the device count.
    models = _materialize_models(tmp_path, "llama-3.2-3b")
    base = {"MODEL_KEYS": "llama-3.2-3b", "MODELS_ROOT": str(models),
            "CONTEXTS": "4096", "DRY_RUN": "1"}
    control = _run(tmp_path, {**base, "NGPUS": "1"}, out_name="control")
    assert control.returncode == 0, _combined(control)

    # When: the same run is asked for eight devices.
    result = _run(tmp_path, {**base, "NGPUS": "8"}, out_name="refused")
    combined = _combined(result)

    # Then: it fails for the single-device reason -- the phrase names WHY, so removing the
    # guard (which would otherwise let the run fall through and succeed) fails this test --
    # and it fails before OUTDIR is even created, so nothing was planned or measured.
    assert result.returncode == 2, combined
    assert "NGPUS=8" in combined
    assert "single-device" in combined
    assert not (tmp_path / "refused").exists()


# --------------------------------------------------------------------------
# unknown key vs absent weights: a reader must be able to tell them apart
# --------------------------------------------------------------------------


def test_an_unknown_model_key_is_refused_by_name(tmp_path: Path) -> None:
    # Given: a key the registry does not declare.
    models = _materialize_models(tmp_path, "llama-3.2-3b")

    # When: it is requested.
    result = _run(tmp_path, {"MODEL_KEYS": "no-such-model-xyz", "MODELS_ROOT": str(models),
                             "DRY_RUN": "1"})
    combined = _combined(result)

    # Then: the refusal names the key and says it is unknown -- so a typo surfaces as the
    # typo rather than as a missing directory, which is the other failure entirely.
    assert result.returncode == 2
    assert "no-such-model-xyz" in combined
    assert "unknown model key" in combined


def test_a_known_key_with_absent_weights_is_refused_by_path(tmp_path: Path) -> None:
    # Given: a declared key and an empty models root, so its weights are absent.
    key = "llama-3.2-3b"
    models = tmp_path / "empty_models"
    models.mkdir()
    expected = models / efficiency.REGISTRY[key].path
    assert not expected.exists()

    # When: it is requested.
    result = _run(tmp_path, {"MODEL_KEYS": key, "MODELS_ROOT": str(models), "DRY_RUN": "1"})
    combined = _combined(result)

    # Then: the refusal names the path it looked at, so a model that was never downloaded
    # cannot be scored as a present one -- the distinction the matrix depends on.
    assert result.returncode == 2
    assert str(expected) in combined
    assert key in combined


def test_a_registry_that_cannot_be_imported_is_refused(tmp_path: Path) -> None:
    # Given: an import root with no scale package in it, so the registry cannot be loaded.
    empty_root = tmp_path / "no_such_tree"
    empty_root.mkdir()

    # When: the launcher preflight runs.
    result = _run(tmp_path, {"IMPORT_ROOT": str(empty_root), "MODEL_KEYS": "llama-3.2-3b",
                             "MODELS_ROOT": str(tmp_path), "DRY_RUN": "1"})
    combined = _combined(result)

    # Then: the failure is reported as an import failure rather than falling through to a
    # grid that would have no registry to build.
    assert result.returncode == 2
    assert "cannot import the adapters registry" in combined


# --------------------------------------------------------------------------
# contexts must be a strictly ascending, positive ladder
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("contexts", "reason"),
    [
        ("4096,,8192", "positive integers"),
        ("abc", "positive integers"),
        ("0,4096", "positive integers"),
        ("-1,4096", "positive integers"),
        ("8192,4096", "strictly ascend"),
        ("4096,4096", "strictly ascend"),
        ("", "named no context"),
    ],
)
def test_a_malformed_contexts_string_is_refused(
    tmp_path: Path, contexts: str, reason: str
) -> None:
    # Given: a valid model key, so the only defect is the contexts string.
    models = _materialize_models(tmp_path, "llama-3.2-3b")

    # When: the malformed string is supplied.
    result = _run(tmp_path, {"MODEL_KEYS": "llama-3.2-3b", "MODELS_ROOT": str(models),
                             "CONTEXTS": contexts, "DRY_RUN": "1"})
    combined = _combined(result)

    # Then: it is refused with the specific reason, rather than passed to a probe that would
    # have to discover the defect after the model was loaded.
    assert result.returncode == 2, combined
    assert reason in combined


# --------------------------------------------------------------------------
# the mode allowlist
# --------------------------------------------------------------------------


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    # Given: a mode this launcher does not implement, plus an otherwise broken run so that
    # the mode guard -- not some later refusal -- is what must stop it.
    result = _run(tmp_path, {"MODE": "measure", "MODEL_KEYS": "no-such-model-xyz",
                             "MODELS_ROOT": str(tmp_path), "DRY_RUN": "1"})
    combined = _combined(result)

    # Then: it is refused by name, before OUTDIR is created.
    assert result.returncode == 2
    assert "unknown MODE" in combined
    assert "MODE=measure" in combined
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------
# the dry run, which is where the grid shape is observable without a device
# --------------------------------------------------------------------------


def test_dry_run_lists_every_planned_cell_and_touches_no_device(tmp_path: Path) -> None:
    # Given: two registry keys present on disk and a two-rung contexts string.
    models = _materialize_models(tmp_path, "llama-3.2-3b", "qwen2.5-3b")

    # When: the launcher runs dry.
    result = _run(tmp_path, {"MODEL_KEYS": "llama-3.2-3b,qwen2.5-3b", "MODELS_ROOT": str(models),
                             "CONTEXTS": "4096,16384", "DRY_RUN": "1"})
    combined = _combined(result)

    # Then: it succeeds, plans four cells, and says it skipped the CUDA check rather than
    # failing it -- a dry run that demanded a GPU could not plan a GPU experiment from a
    # login node.
    assert result.returncode == 0, combined
    assert "cells planned" in combined
    assert "llama-3.2-3b_4096.json" in combined
    assert "qwen2.5-3b_16384.json" in combined
    assert "DRY_RUN=1 does not touch a device" in combined
    assert "PREFLIGHT FAIL" not in combined
    # The boot log is still written (it is the run's record), but no cell is measured.
    out = tmp_path / "out"
    assert list(out.glob("*.boot.txt"))
    assert not list(out.glob("*_4096.json"))


def test_the_launcher_is_a_single_device_probe_not_a_distributed_launch() -> None:
    # Given: the launcher source.
    text = LAUNCHER.read_text(encoding="utf-8")

    # Then: it pins one GPU and never reaches for a process-group launcher -- a torchrun
    # here would silently turn a per-device measurement into a multi-device one.
    assert "readonly REQUIRED_NGPUS=1" in text
    assert "torchrun" not in text


# --------------------------------------------------------------------------
# end to end against a fake scale package: the grid loop, OOM rows, load failures
# --------------------------------------------------------------------------

#: A stand-in adapters registry with the same ``build`` contract the real one enforces:
#: an unknown key and an absent path are different refusals.  ``FAKE_LOAD_FAIL`` names a
#: key whose ``load`` raises, so the launcher's "abort on load failure" path is reachable.
FAKE_ADAPTERS = '''\
"""Test double for efficiency.adapters: two keys, no weights, no CUDA."""
import os
from pathlib import Path


class AdapterRefusal(ValueError):
    pass


class _Spec:
    def __init__(self, key, path):
        self.key = key
        self.family = "transformer"
        self.objective = "autoregressive"
        self.nfe = 1
        self.weights_bytes = 1_000_000
        self.path = path


REGISTRY = {"fake-a": _Spec("fake-a", "Fake-A"), "fake-b": _Spec("fake-b", "Fake-B")}


class _Adapter:
    def __init__(self, spec, resolved_path):
        self.spec = spec
        self.resolved_path = resolved_path

    def load(self, device):
        if os.environ.get("FAKE_LOAD_FAIL") == self.spec.key:
            raise RuntimeError(f"fake load failure for {self.spec.key}")
        return self

    def analytic_state_bytes(self, context):
        return 1024

    def forward(self, tokens):
        return None


def build(key, *, models_root):
    spec = REGISTRY.get(key)
    if spec is None:
        raise AdapterRefusal(
            f"unknown model key {key!r}; the registry declares {sorted(REGISTRY)}")
    resolved = Path(spec.path) if Path(spec.path).is_absolute() else Path(models_root) / spec.path
    if not resolved.exists():
        raise AdapterRefusal(f"weights for {key!r} are absent at {resolved}; check models_root")
    return _Adapter(spec, resolved)
'''

#: A stand-in probe_inference.  It *asserts* the arm it is handed satisfies the
#: InferenceAdapter shape the real runner relies on, so a driver that forwards the raw
#: CheckpointAdapter (whose ``analytic_state_bytes`` is a method, not an int) fails here
#: rather than producing a row from the wrong object.  A context at or above ``FAKE_OOM_AT``
#: is recorded as an oom row, matching the protocol.
FAKE_PROBE = '''\
"""Test double for efficiency.probe_inference: writes an artifact, records oom rows."""
import json
import os
from pathlib import Path

STATUS_OK = "ok"
STATUS_OOM = "oom"
PROBE_FILENAME = "inference_probe.json"


def _oom_at():
    return int(os.environ.get("FAKE_OOM_AT", str(1 << 60)))


def probe_ladder(adapter, *, contexts, device, out_dir):
    assert isinstance(getattr(adapter, "analytic_state_bytes", None), int), (
        "the arm must carry a context-independent analytic_state_bytes int")
    assert hasattr(adapter, "analytic_weights_bytes"), "the arm must carry analytic_weights_bytes"
    assert isinstance(getattr(adapter, "nfe", None), int), "the arm must carry an int nfe"
    assert callable(getattr(adapter, "forward", None)), "the arm must carry forward()"

    rows = [
        {"model": adapter.model, "context": context,
         "status": STATUS_OOM if context >= _oom_at() else STATUS_OK}
        for context in contexts
    ]
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": adapter.model,
        "device": device,
        "contexts_requested": list(contexts),
        "contexts_measured": [row["context"] for row in rows if row["status"] == STATUS_OK],
        "stopped_at_oom": bool(rows) and rows[-1]["status"] == STATUS_OOM,
        "rows": rows,
    }
    (directory / PROBE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return rows
'''

FAKE_CUDA_WRAPPER_TEMPLATE = """#!/usr/bin/env bash
# Stand-in for python3: report a CUDA device for the launcher's CUDA preflight, and delegate
# every other invocation (registry preflight, payload) to the real interpreter.  This keeps
# the launcher's real code path while letting the grid be driven without a GPU.
for arg in "$@"; do
  case "$arg" in
    *cuda.is_available*)
      echo "torch 0.0.0-fake cuda True"
      exit 0
      ;;
  esac
done
exec __PYTHON__ "$@"
"""


@pytest.fixture
def fake_scale_tree(tmp_path: Path) -> Path:
    """A fake ``scale.experiments.nonlatent_iclr.efficiency`` package on disk."""
    root = tmp_path / "fake_import_root"
    package = root / "scale" / "experiments" / "nonlatent_iclr" / "efficiency"
    package.mkdir(parents=True)
    for init in (root / "scale", root / "scale" / "experiments",
                 root / "scale" / "experiments" / "nonlatent_iclr", package):
        (init / "__init__.py").write_text("", encoding="utf-8")
    (package / "adapters.py").write_text(FAKE_ADAPTERS, encoding="utf-8")
    (package / "probe_inference.py").write_text(FAKE_PROBE, encoding="utf-8")
    return root


@pytest.fixture
def fake_cuda_python(tmp_path: Path) -> Path:
    """A python3 stand-in that fakes only the CUDA preflight."""
    wrapper = tmp_path / "python3-cuda-stand-in"
    wrapper.write_text(
        FAKE_CUDA_WRAPPER_TEMPLATE.replace("__PYTHON__", sys.executable), encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def _fake_grid_env(tmp_path: Path, fake_scale_tree: Path, fake_cuda_python: Path,
                   **overrides: str) -> dict[str, str]:
    models = tmp_path / "fake_models"
    for key in ("Fake-A", "Fake-B"):
        (models / key).mkdir(parents=True, exist_ok=True)
    env = {
        "IMPORT_ROOT": str(fake_scale_tree),
        "MODELS_ROOT": str(models),
        "PYTHON_BIN": str(fake_cuda_python),
        "MODEL_KEYS": "fake-a,fake-b",
        "CONTEXTS": "4096,16384",
        "DRY_RUN": "0",
    }
    env.update(overrides)
    return env


def test_a_real_invocation_with_a_fake_probe_writes_one_json_per_cell(
    tmp_path: Path, fake_scale_tree: Path, fake_cuda_python: Path
) -> None:
    # Given: two fake models and two contexts.
    env = _fake_grid_env(tmp_path, fake_scale_tree, fake_cuda_python)

    # When: the launcher runs for real (CUDA faked only for the preflight).
    result = _run(tmp_path, env)
    combined = _combined(result)

    # Then: it succeeds and wrote exactly one JSON per cell, named by key and context.
    assert result.returncode == 0, combined
    out = tmp_path / "out"
    for key in ("fake-a", "fake-b"):
        for context in (4096, 16384):
            cell = out / f"{key}_{context}.json"
            assert cell.exists(), f"{cell} missing\n{combined}"
            payload = json.loads(cell.read_text(encoding="utf-8"))
            assert payload["rows"][0]["status"] == "ok"
    assert "SUMMARY: cells written=4 cells failed=0" in combined


def test_an_oom_row_is_kept_and_does_not_abort_the_grid(
    tmp_path: Path, fake_scale_tree: Path, fake_cuda_python: Path
) -> None:
    # Given: one model whose second rung does not fit.
    env = _fake_grid_env(tmp_path, fake_scale_tree, fake_cuda_python,
                         MODEL_KEYS="fake-a", CONTEXTS="4096,16384,32768", FAKE_OOM_AT="16384")

    # When: the grid runs.
    result = _run(tmp_path, env)
    combined = _combined(result)

    # Then: every rung still produced a file -- the OOM is a recorded row, not a gap -- and
    # the grid continued to the larger context.  This is the protocol the table rests on:
    # four contexts attempted, not two measured.
    assert result.returncode == 0, combined
    out = tmp_path / "out"
    assert (out / "fake-a_4096.json").exists()
    assert (out / "fake-a_16384.json").exists()
    assert (out / "fake-a_32768.json").exists()
    oom = json.loads((out / "fake-a_16384.json").read_text(encoding="utf-8"))
    assert oom["rows"][0]["status"] == "oom"
    assert "SUMMARY: cells written=1 cells failed=2" in combined


def test_a_load_failure_is_recorded_and_the_grid_continues(
    tmp_path: Path, fake_scale_tree: Path, fake_cuda_python: Path
) -> None:
    r"""A load failure must not cost the cells the OTHER models would have produced.

    This test used to assert the opposite -- that the grid aborts, "because a load failure
    is a defect in the run". The first real run showed what that policy costs: it measured
    all four BiRWKV contexts, hit a load failure on LLaDA, and aborted, losing the twenty
    cells the other five models would have produced. The aborting policy reported one
    defect and destroyed five models' worth of evidence to do it.

    So the failure is now a row: the model gets a `load_failed` cell per requested context
    carrying the refusal text, and the run exits ZERO so the next model is attempted.
    """
    # Given: two models, the second of which cannot be loaded.
    env = _fake_grid_env(tmp_path, fake_scale_tree, fake_cuda_python, FAKE_LOAD_FAIL="fake-b")

    # When: the grid runs.
    result = _run(tmp_path, env)
    combined = _combined(result)

    # Then: the run completes -- the grid CONTINUES past the failure...
    assert result.returncode == 0, combined
    assert "LOAD FAIL" in combined
    assert "fake-b" in combined
    assert "aborting the grid" not in combined
    out = tmp_path / "out"
    # ...the model that loaded has its cells...
    assert (out / "fake-a_4096.json").exists()
    # ...and the model that did not has a cell recording WHY, rather than a hole.
    failed = out / "fake-b_4096.json"
    assert failed.exists(), "a load failure must leave a row, not an absence"
    import json as _json
    document = _json.loads(failed.read_text())
    row = document["rows"][0]
    assert row["status"] == "load_failed"
    assert row["error"], "the hole must carry its reason"
    assert row["tokens_per_second"] is None, "no number may be printed where a load failed"
    assert row["peak_allocated_bytes"] is None


def test_the_campaign_code_root_is_on_the_import_path() -> None:
    """The BiRWKV backbone lives in the campaign tree, not in the scale package.

    `models.birwkv7_diffusion` is shared with the trainer and the eval lanes, so it is not
    owned by this experiment and does not ship inside `scale`. Without the campaign root on
    sys.path the adapter refuses with a named error -- correct behaviour, useless
    deployment: the grid aborts on the one model the paper is about, having measured
    nothing.
    """
    script = (Path(__file__).resolve().parents[3]
              / "scale" / "qz" / "launch_efficiency_probe.sh").read_text()
    # Then: the root is declared, defaulted relative to the repo, and exported on the path.
    assert "DAN_CODE_ROOT" in script
    assert "DEFAULT_DAN_CODE_ROOT" in script
    assert "export PYTHONPATH=" in script
    path_line = next(l for l in script.splitlines() if l.startswith("export PYTHONPATH="))
    assert "${DAN_CODE_ROOT}" in path_line
    # and IMPORT_ROOT comes FIRST, so `scale` resolves to the tree under test rather than
    # to a copy that happens to sit inside the campaign tree
    assert path_line.index("${IMPORT_ROOT}") < path_line.index("${DAN_CODE_ROOT}")
    # the BOOT log records it, so a reader can tell which backbone tree a row came from
    assert "DAN_CODE_ROOT=$DAN_CODE_ROOT" in script


def test_a_load_failure_is_a_row_not_an_abort() -> None:
    """The policy that cost twenty cells to report one.

    The first real run of this grid measured all four BiRWKV contexts, then hit a load
    failure on LLaDA and aborted -- losing the twenty cells the other five models would
    have produced. A model that cannot be loaded is a fact about the ARTIFACT, and the
    other models' measurements do not depend on it, so the honest output is a table with a
    named hole rather than no table.

    The distinction the original policy was reaching for is real but was drawn one level
    too coarse: an IMPORT failure still aborts, because every cell after it would be
    meaningless.
    """
    script = (Path(__file__).resolve().parents[3]
              / "scale" / "qz" / "launch_efficiency_probe.sh").read_text()
    lines = script.splitlines()
    start = next(i for i, l in enumerate(lines) if "<<'PYPROBE'" in l) + 1
    end = next(i for i, l in enumerate(lines) if l.strip() == "PYPROBE" and i > start)
    driver = "\n".join(lines[start:end])

    # Then: the load failure writes rows and exits ZERO, so the grid continues.
    assert 'status": "load_failed"' in driver
    assert "raise SystemExit(0)" in driver, "a load failure must not abort the grid"
    assert "raise SystemExit(3)" not in driver, "the abort is what lost twenty cells"
    # the import failure still aborts -- that IS a run defect
    assert "raise SystemExit(2)" in driver
    # and the error text reaches the row, so the hole has a reason
    assert '"error": message' in driver
    # the header's stated policy must match the behaviour
    assert "does NOT abort it" in script
