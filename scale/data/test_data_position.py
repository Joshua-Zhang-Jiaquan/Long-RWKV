"""The resume position must follow the TRAINER's precedence, not the flag's.

The bug these tests pin was in a first launcher patch: it preferred
``--resume-from`` while the trainer prefers its own newest checkpoint
(``train_birwkv_diffusion.py:1119`` before ``:1160``).  On a run that has both --
a warm start on the command line and its own checkpoints from earlier segments --
the model and the sampler would be positioned at different steps, and nothing
would fail: both numbers are individually valid, so the only symptom is that the
run re-reads data it already trained on.  That is the same class of defect as the
one the positioning exists to remove, at a different offset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scale.data.data_position import (  # noqa: E402
    PositionRefusal, main, newest_checkpoint, resolve_resume_step, step_of)
from scale.data.fineweb4096_sampler import resume_position  # noqa: E402


def _checkpoint(root: Path, name: str, step: int, *, complete: bool = True) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    if complete:
        (path / "model.pt").write_bytes(b"x")
    return path


# --------------------------------------------------------------------------
# the precedence the peer caught
# --------------------------------------------------------------------------


def test_the_own_checkpoint_wins_when_both_sources_exist(tmp_path: Path) -> None:
    """The case that would have mispositioned a real run."""
    # Given: a run with its own checkpoint at 15500 and a warm start at 9500.
    save_root = tmp_path / "out"
    _checkpoint(save_root / "m4-loop-2p9b", "step_00015500", 15500)
    warm = _checkpoint(tmp_path / "warm", "endpoint", 9500)
    # When: the position is resolved the way the launcher resolves it.
    step, source = resolve_resume_step(save_root=save_root, run_name="m4-loop-2p9b",
                                       resume_from=str(warm))
    # Then: it is the OWN checkpoint's step, because that is the one the trainer
    # loads -- the flag is only a fallback.
    assert step == 15500
    assert source.startswith("own-dir")


def test_the_flag_is_used_only_when_the_run_has_no_checkpoint(tmp_path: Path) -> None:
    # Given: an empty run directory and a warm start.
    save_root = tmp_path / "out"
    (save_root / "fresh-run").mkdir(parents=True)
    warm = _checkpoint(tmp_path / "warm", "endpoint", 9500)
    # When: the position is resolved.
    step, source = resolve_resume_step(save_root=save_root, run_name="fresh-run",
                                       resume_from=str(warm))
    # Then: the warm start is the run's starting point, which is correct on a
    # first launch and only on a first launch.
    assert (step, source) == (9500, "resume-from:endpoint")


def test_a_first_launch_with_no_flag_is_fresh(tmp_path: Path) -> None:
    step, source = resolve_resume_step(save_root=tmp_path / "out",
                                       run_name="nothing-here", resume_from=None)
    assert (step, source) == (0, "fresh")


def test_a_missing_run_directory_is_fresh_not_an_error(tmp_path: Path) -> None:
    # Given: a save root that does not exist yet.
    # When/Then: a fresh run, because that is what a first launch looks like.
    step, source = resolve_resume_step(save_root=tmp_path / "absent",
                                       run_name="r", resume_from=None)
    assert (step, source) == (0, "fresh")


def test_the_resolved_step_feeds_the_sampler_in_the_same_units() -> None:
    """The peer's second question: per-rank rows on both sides."""
    # Given: a position from the launcher and a per-rank epoch length.
    step, _ = 15500, None
    per_rank_rows_per_step = 8 * 4          # microbatch x grad_accum, one rank
    per_rank_rows_per_epoch = 24_390_000 // 4
    # When: the sampler converts it.
    epoch, skip = resume_position(start_step=step, rows_per_step=per_rank_rows_per_step,
                                  rows_per_epoch=per_rank_rows_per_epoch)
    # Then: the split is consistent -- the two units are both per-rank, so the
    # epoch/offset split cannot be off by the world size.
    assert epoch * per_rank_rows_per_epoch + skip == step * per_rank_rows_per_step


# --------------------------------------------------------------------------
# which checkpoint counts
# --------------------------------------------------------------------------


def test_the_newest_complete_checkpoint_is_chosen(tmp_path: Path) -> None:
    # Given: three checkpoints at increasing steps.
    for step in (14000, 15000, 15500):
        _checkpoint(tmp_path, f"step_{step:08d}", step)
    # When/Then: the latest is chosen, matching `find_resume`'s ordering.
    assert newest_checkpoint(tmp_path).name == "step_00015500"
    assert step_of(newest_checkpoint(tmp_path)) == 15500


def test_a_half_written_checkpoint_is_ignored(tmp_path: Path) -> None:
    """A wall-cap stop can leave a directory with no model.pt behind."""
    # Given: a complete 15000 and an incomplete 15500.
    _checkpoint(tmp_path, "step_00015000", 15000)
    _checkpoint(tmp_path, "step_00015500", 15500, complete=False)
    # When/Then: the complete one, because the trainer's own test is model.pt --
    # choosing the incomplete one would position against a checkpoint that is
    # never loaded.
    assert newest_checkpoint(tmp_path).name == "step_00015000"


def test_a_sidecar_directory_is_not_mistaken_for_a_checkpoint(tmp_path: Path) -> None:
    """``find_resume`` globs exactly eight digits."""
    # Given: a real checkpoint and a suffixed probe copy beside it.
    _checkpoint(tmp_path, "step_00015000", 15000)
    _checkpoint(tmp_path, "step_00015000_probe_copy", 99999)
    # When/Then: the probe copy is ignored -- the trainer would never load it, so
    # positioning against it would be silently wrong.
    assert newest_checkpoint(tmp_path).name == "step_00015000"


def test_an_empty_directory_has_no_checkpoint(tmp_path: Path) -> None:
    assert newest_checkpoint(tmp_path) is None


def test_a_checkpoint_without_meta_is_refused(tmp_path: Path) -> None:
    # Given: a checkpoint directory with no meta.json.
    path = tmp_path / "step_00015000"
    path.mkdir()
    (path / "model.pt").write_bytes(b"x")
    # When/Then: refused rather than silently read as step 0, which would place
    # the sampler at the head of the data for a run that is 15000 steps in.
    with pytest.raises(PositionRefusal, match="no meta.json"):
        step_of(path)


def test_a_non_integer_step_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ck"
    path.mkdir()
    (path / "meta.json").write_text(json.dumps({"step": "later"}), encoding="utf-8")
    with pytest.raises(PositionRefusal, match="no integer step"):
        step_of(path)


def test_a_flag_pointing_nowhere_with_no_checkpoint_is_refused(tmp_path: Path) -> None:
    # Given: a run with no checkpoint and a --resume-from that does not exist.
    # When/Then: refused. The trainer would fail to resume here, so inventing a
    # position of 0 would position a run that cannot start.
    with pytest.raises(PositionRefusal, match="not a directory"):
        resolve_resume_step(save_root=tmp_path / "out", run_name="r",
                            resume_from=str(tmp_path / "absent"))


# --------------------------------------------------------------------------
# the CLI the launcher calls
# --------------------------------------------------------------------------


def test_the_cli_prints_step_and_source(tmp_path: Path, capsys) -> None:
    # Given: a run with its own checkpoint.
    save_root = tmp_path / "out"
    _checkpoint(save_root / "r", "step_00015500", 15500)
    # When: the launcher's CLI form is invoked.
    code = main(["--save-root", str(save_root), "--run-name", "r",
                 "--resume-from", ""])
    # Then: exactly "<step> <source>", which is what the shell's
    # `${DP_POS%% *}` and `${DP_POS#* }` splits expect.
    assert code == 0
    out = capsys.readouterr().out.strip()
    step, _, source = out.partition(" ")
    assert step == "15500"
    assert source.startswith("own-dir")


def test_an_empty_flag_string_means_absent(tmp_path: Path, capsys) -> None:
    # Given: an invocation whose flag is the empty string, which is how the
    # launcher expresses "the flag was not given".
    code = main(["--save-root", str(tmp_path / "out"), "--run-name", "r",
                 "--resume-from", ""])
    # When/Then: treated as absent rather than as a path.
    assert code == 0
    assert capsys.readouterr().out.strip() == "0 fresh"
