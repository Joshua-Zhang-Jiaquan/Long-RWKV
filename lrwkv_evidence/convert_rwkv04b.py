"""Convert an ``outputs_rwkv04b`` bundled checkpoint into a loadable step dir.

The rwkv04b trainer saved ``step_00001908.pt`` as a *training* checkpoint:

    {"model": {...867 tensors...}, "optimizer": {...}, "scaler": {}, "step": int,
     "record": {...}}

``eval/capability/birwkv_diffusion_model.load_birwkv_diffusion`` wants the other
shape -- a directory containing a **flat** ``model.pt`` that is itself the state
dict -- because that is what the task7/2.9B trainers wrote.  Both are legitimate;
they are simply different, and the eval harness only reads one.  This writes the
second from the first so the five rwkv04b entries can be evaluated through the
*same* code path as every other BiRWKV checkpoint, which is the only way their
numbers are comparable.

Two details that are not incidental:

``weights_only=True`` needs one allowlisted global
    The bundle's ``record`` carries a ``torch.torch_version.TorchVersion``, which
    the PyTorch>=2.6 default refuses.  These are this project's own artifacts, so
    the global is allowlisted explicitly rather than by dropping to
    ``weights_only=False`` -- the narrow allowance keeps the safety property for
    everything else in the file.

The optimizer state is dropped, and that is the point
    It is ~3.8 GB of the 5.4 GB payload and no eval reads it.  Dropping it is
    recorded in the sidecar so nobody later mistakes the derived dir for a
    resumable training checkpoint.

The extracted weights are written with their source sha256 recorded, so a paper
number traced to the derived dir still resolves to the original bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from . import tracks
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lrwkv_evidence import tracks

#: Where derived step dirs go.  Beside the source run, not inside it: the run dir
#: is the trainer's output and a reader must be able to tell the two apart.
DERIVED_ROOT = tracks.G / "derived_ckpts_rwkv04b"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def convert(track: tracks.Track, out_root: Path = DERIVED_ROOT,
            force: bool = False) -> dict:
    import torch

    src = track.path
    if not src.is_file():
        raise SystemExit(f"{track.track_id}: no bundle at {src}")
    out_dir = out_root / track.track_id / f"step_{track.step:08d}"
    target = out_dir / "model.pt"
    if target.is_file() and not force:
        return {"track_id": track.track_id, "status": "already_present",
                "ckpt_dir": str(out_dir)}

    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    bundle = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise SystemExit(
            f"{track.track_id}: {src} is not a bundle with a 'model' key "
            f"(keys: {list(bundle)[:8] if isinstance(bundle, dict) else type(bundle)}); "
            f"refusing to guess which tensor group is the state dict")
    state = bundle["model"]
    step = int(bundle.get("step", -1))
    if track.step is not None and step != track.step:
        raise SystemExit(
            f"{track.track_id}: the bundle says step {step} but the registry says "
            f"{track.step}. One of them is describing a different checkpoint, and "
            f"a token count attributed to the wrong step is a wrong measurement.")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)
    meta = {
        "step": step,
        "tokens_seen": track.tokens_seen,
        "derived_from": str(src),
        "derived_from_sha256": track.sha256 or sha256_file(src),
        "extracted_key": "model",
        "tensors": len(state),
        "dropped": ["optimizer", "scaler", "record"],
        "dropped_note": (
            "This is an EVAL-ONLY checkpoint. The optimizer and scaler state were "
            "not copied, so it cannot resume training; the source bundle remains "
            "the resumable artifact."),
        "loader": "eval.capability.birwkv_diffusion_model.load_birwkv_diffusion",
        "model_dir": track.model_dir,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                       encoding="utf-8")
    return {"track_id": track.track_id, "status": "converted",
            "ckpt_dir": str(out_dir), "tensors": len(state),
            "model_pt_sha256": sha256_file(target), **meta}


def rwkv04b_tracks() -> list[tracks.Track]:
    return [t for t in tracks.ALL_TRACKS
            if t.model_kind == "birwkv_diffusion" and t.path.is_file()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", action="append",
                   help="track_id to convert; repeatable. Default: all bundles.")
    ap.add_argument("--out-root", type=Path, default=DERIVED_ROOT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    todo = rwkv04b_tracks()
    if args.track:
        wanted = set(args.track)
        todo = [t for t in todo if t.track_id in wanted]
        missing = wanted - {t.track_id for t in todo}
        if missing:
            raise SystemExit(f"not bundled rwkv04b tracks: {sorted(missing)}")
    if args.list:
        print(json.dumps([t.track_id for t in todo], indent=2))
        return 0
    report = [convert(t, args.out_root, args.force) for t in todo]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
