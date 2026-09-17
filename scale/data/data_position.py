"""Where a resumed run actually is, in the trainer's own precedence.

The launcher has to know the step a run resumes from in order to position the
sampler, and it must agree with the trainer about *which* checkpoint that is.  It
does not follow from the command line, because the trainer consults two sources
in a fixed order and the flag is the SECOND one:

* ``train_birwkv_diffusion.py:1119`` -- ``resume = find_resume(save_dir)``, the
  newest complete checkpoint in the run's own directory, and
* ``train_birwkv_diffusion.py:1160`` -- ``elif args.resume_from``, used only when
  the own directory has none.

A first version of the launcher derivation preferred the flag.  On a run that has
both -- a warm-start checkpoint passed by ``--resume-from`` and its own
checkpoints from earlier segments -- that disagrees with the trainer, and it
disagrees precisely at the wall-cap resubmission boundaries the positioning
exists for: the model would resume at step 15,500 while the sampler was placed at
the warm start's step 9,500, re-reading a slice it had already trained on.  Both
numbers are individually valid, so nothing fails; the data exposure is simply
wrong.

So the rule lives here, in one place, tested against the trainer's ordering
rather than restated in shell.
"""

from __future__ import annotations

from pathlib import Path

#: ``find_resume``'s glob, exactly: eight digits.  A looser ``step_*`` would also
#: match sidecar directories such as ``step_00015000_probe_copy``, which the
#: trainer never considers, so the launcher could position against a checkpoint
#: the run does not load.
STEP_GLOB = "step_????????"


class PositionRefusal(ValueError):
    """The resume position cannot be established."""


def newest_checkpoint(save_dir: Path) -> Path | None:
    """The newest complete checkpoint, mirroring ``find_resume`` exactly.

    "Complete" means ``model.pt`` exists, which is the trainer's own test -- a
    directory left half-written by a wall-cap stop must not be chosen, because
    the trainer would skip it and the launcher would then be positioned against
    a checkpoint that is not loaded.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        return None
    candidates = sorted(
        path for path in save_dir.glob(STEP_GLOB) if (path / "model.pt").is_file())
    return candidates[-1] if candidates else None


def step_of(checkpoint: Path) -> int:
    """The step recorded in a checkpoint's ``meta.json``."""
    import json

    meta_path = Path(checkpoint) / "meta.json"
    if not meta_path.is_file():
        raise PositionRefusal(f"{checkpoint} has no meta.json")
    try:
        document = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PositionRefusal(f"{meta_path} is unreadable: {exc}") from exc
    step = document.get("step")
    if not isinstance(step, int):
        raise PositionRefusal(f"{meta_path} carries no integer step: {step!r}")
    return step


def resolve_resume_step(*, save_root: Path, run_name: str,
                        resume_from: str | None) -> tuple[int, str]:
    """``(start_step, source)`` under the TRAINER's precedence.

    Own-directory checkpoint first, ``--resume-from`` only as a fallback.  The
    source is returned so the launcher can print which rule fired: when the two
    disagree, the only visible symptom is a number in a log line, and a reader
    needs to know whether that number came from the run's history or from its
    warm start.
    """
    own = newest_checkpoint(Path(save_root) / run_name)
    if own is not None:
        return step_of(own), f"own-dir:{own.name}"
    if resume_from:
        warm = Path(resume_from)
        if not warm.is_dir():
            msg = (f"--resume-from {resume_from} is not a directory and the run "
                   f"has no checkpoint of its own; the trainer would fail here, "
                   f"so the position cannot be established")
            raise PositionRefusal(msg)
        return step_of(warm), f"resume-from:{warm.name}"
    return 0, "fresh"


def main(argv: list[str] | None = None) -> int:
    """CLI for the launcher: print ``"<step> <source>"`` and nothing else.

    An empty ``--resume-from`` means the flag was absent.  Shell cannot easily
    omit an argument conditionally, so the absence is expressed as an empty
    string and handled here rather than in the launcher's quoting.
    """
    import argparse

    parser = argparse.ArgumentParser(description="resolve the data resume position")
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume-from", default="")
    args = parser.parse_args(argv)
    step, source = resolve_resume_step(
        save_root=Path(args.save_root), run_name=args.run_name,
        resume_from=args.resume_from or None)
    print(f"{step} {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
