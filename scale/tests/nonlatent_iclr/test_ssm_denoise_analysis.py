"""Task 10: the E1-E4 spectral validation must fail loudly.

The 2026-09-03 run of ``ssm_denoise_analysis`` raised at E3 and its wrapper still
printed ``===SSM DONE exit=0===``, so a crashed analysis was indistinguishable
from a completed one to anything reading the wrapper.  Two things therefore need
pinning, and neither needs a GPU:

* a crash leaves ``ssm_FAILED.json`` behind and exits non-zero, whatever the
  caller reports;
* the E3 trajectory runs at all, on a stub, with the canvas pinned to the
  model's device -- the failure was a mixed-device copy inside
  ``canvas_embed``/``reverse_trajectory``.

The geometry here is tiny but the vocabulary is not: ``reverse_trajectory``
builds a ``[n_mask, V]`` canvas with ``V = 65536`` hard-coded, so the stub has to
carry the real vocabulary width for the shapes to line up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from scale.experiments import ssm_denoise_analysis as ssm

VOCAB = ssm.V if hasattr(ssm, "V") else 65536
HIDDEN = 8


class _StubDenoiser(torch.nn.Module):
    """A denoiser stub carrying the interface ``SoftMap`` drives.

    Its ``forward`` reads a registered buffer rather than ``self.embeddings``:
    ``SoftMap._forward_logits`` swaps ``embeddings`` for a canvas stub, so a
    forward that touched it would break on the swap rather than on the maths.
    """

    def __init__(self, vocab: int = VOCAB, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab, hidden)
        self.register_buffer(
            "field", torch.cos(torch.arange(vocab, dtype=torch.float32) * 0.0007))
        self.vocab = vocab

    def forward(self, ids: torch.Tensor, force_forward: bool) -> torch.Tensor:  # noqa: ARG002
        batch, length = ids.shape
        return self.field.view(1, 1, -1).expand(batch, length, self.vocab).clone()


def _fixture(length: int = 16, n_mask: int = 4):
    model = _StubDenoiser()
    smap = ssm.SoftMap(model, "cpu")
    generator = torch.Generator().manual_seed(7)
    clean = torch.randint(0, VOCAB, (length,), generator=generator)
    mask_idx = torch.arange(6, 6 + n_mask)
    return smap, clean, mask_idx


# --------------------------------------------------------------------------
# the canvas and the simplex point
# --------------------------------------------------------------------------


def test_the_canvas_is_pinned_to_the_model_device_and_dtype() -> None:
    # Given: a document and a mask range, with the mask indices passed as a
    # different integer dtype than the model's tensors.
    smap, clean, mask_idx = _fixture()
    X = ssm.make_start(clean, mask_idx, 0.5, torch.device("cpu"))
    # When: the canvas is built.
    h = smap.canvas_embed(X, ssm._vis_ids(clean, mask_idx), mask_idx.to(torch.int32))
    # Then: it carries the embedding's device and dtype -- the property whose
    # absence caused the E3 crash.
    assert h.device == smap.emb_w.device
    assert h.dtype == smap.emb_w.dtype
    assert h.shape == (1, clean.numel(), smap.emb_w.shape[1])
    # and the visible positions really are the document's embeddings
    visible = ssm._vis_ids(clean, mask_idx)
    n_visible_before_mask = int(mask_idx.min())
    assert torch.allclose(
        h[0, :n_visible_before_mask].to(torch.float32),
        smap.emb_w[visible[:n_visible_before_mask]].to(torch.float32))


def test_the_canvas_accepts_a_float64_canvas_without_a_device_error() -> None:
    # Given: a canvas at a dtype the model does not use.
    smap, clean, mask_idx = _fixture()
    X = ssm.make_start(clean, mask_idx, 0.5, torch.device("cpu")).double()
    # When/Then: it is promoted rather than copied into a mismatched buffer.
    h = smap.canvas_embed(X, ssm._vis_ids(clean, mask_idx), mask_idx)
    assert h.dtype == smap.emb_w.dtype


def test_probs_returns_a_simplex_point_on_the_model_device() -> None:
    # Given: a start canvas.
    smap, clean, mask_idx = _fixture()
    X = ssm.make_start(clean, mask_idx, 0.9, torch.device("cpu"))
    # When: G(X) is evaluated.
    probs = smap.probs(X, ssm._vis_ids(clean, mask_idx), mask_idx)
    # Then: it is a probability simplex point of the right shape and device.
    assert probs.shape == (mask_idx.numel(), VOCAB)
    assert probs.device == smap.emb_w.device
    assert torch.allclose(probs.sum(dim=-1), torch.ones(mask_idx.numel()), atol=1e-4)


# --------------------------------------------------------------------------
# the E3 trajectory -- the stage that crashed
# --------------------------------------------------------------------------


def test_the_reverse_trajectory_runs_and_stays_finite() -> None:
    # Given: a soft map and a document whose tensor is on the host.
    smap, clean, mask_idx = _fixture()
    anchor = ssm.make_start(clean, mask_idx, 1.0, torch.device("cpu"))
    # When: the reverse schedule is run for a few steps.
    result = ssm.reverse_trajectory(smap, clean, mask_idx, 4, anchor)
    # Then: it completes, every recorded quantity is finite, and the trajectory
    # is retained on the host so downstream stacking cannot mix devices.
    assert result["final_acc"] >= 0.0
    for key in ("dist_anchor_l1_per_pos", "conf_mean", "schedule_residual"):
        values = result[key]
        assert len(values) == 4
        assert all(v == v and abs(v) != float("inf") for v in values), (key, values)
    assert len(result["traj"]) == 4
    assert all(t.device.type == "cpu" for t in result["traj"])
    # and the commit gates were produced for both families, which is what the
    # distance-threshold bug prevented
    assert set(result["commit_gates"]) == {
        "conf0.5", "conf0.9", "conf0.99", "dist0.2", "dist0.1", "dist0.05"}


def test_the_trajectory_result_keys_are_the_ones_the_analysis_reads() -> None:
    # Given: a completed trajectory.
    smap, clean, mask_idx = _fixture()
    anchor = ssm.make_start(clean, mask_idx, 1.0, torch.device("cpu"))
    result = ssm.reverse_trajectory(smap, clean, mask_idx, 2, anchor)
    # Then: the keys the JSON assembly and E4 consume are all present -- a
    # renamed key would otherwise surface only as a KeyError mid-run.
    for key in ("dist_anchor_l1_per_pos", "conf_mean", "schedule_residual",
                "final_acc", "commit_gates", "traj"):
        assert key in result


# --------------------------------------------------------------------------
# failing loudly, which is the whole point
# --------------------------------------------------------------------------


def test_a_crash_writes_a_failure_receipt_and_exits_nonzero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a main that raises, as it did at E3.
    def _explode(*_args, **_kwargs):
        raise RuntimeError("boom at E3")

    monkeypatch.setattr(ssm, "main", _explode)
    outdir = tmp_path / "exp"
    monkeypatch.setattr(sys, "argv", ["ssm", "--outdir", str(outdir)])
    # When: the guarded entry point runs.
    code = ssm._guarded_main()
    # Then: it exits non-zero AND leaves a receipt naming the failure, so a
    # wrapper that reports exit=0 cannot make the crash disappear.
    assert code == 1
    receipt = outdir / "ssm_FAILED.json"
    assert receipt.is_file()
    document = json.loads(receipt.read_text())
    assert document["status"] == "FAILED"
    assert "boom at E3" in document["traceback"]


def test_a_crash_without_an_outdir_still_exits_nonzero(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: an invocation with no --outdir to write beside.
    def _explode(*_args, **_kwargs):
        raise RuntimeError("boom before the output dir is known")

    monkeypatch.setattr(ssm, "main", _explode)
    monkeypatch.setattr(sys, "argv", ["ssm"])
    # When/Then: the exit code is still non-zero; the receipt is a bonus, not
    # the mechanism.
    assert ssm._guarded_main() == 1


def test_a_clean_run_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a main that succeeds.
    monkeypatch.setattr(ssm, "main", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["ssm", "--outdir", str(tmp_path)])
    # When/Then: nothing is written and the exit code is zero, so the guard has
    # not turned success into failure.
    assert ssm._guarded_main() == 0
    assert not (tmp_path / "ssm_FAILED.json").exists()


def test_the_completed_receipt_is_stamped_with_its_stages() -> None:
    # Given: the source of the completed-run assembly.
    source = Path(ssm.__file__).read_text(encoding="utf-8")
    # Then: completion is stamped rather than inferred from which keys happen to
    # be present -- the crashed run left no JSON at all, so absence was the only
    # signal a reader had.
    assert '"status": "COMPLETE"' in source
    assert '"stages_completed"' in source
