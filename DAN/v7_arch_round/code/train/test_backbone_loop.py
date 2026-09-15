"""CPU checks for v7 W-A0 — the weight-tied backbone loop arm.

Reuses the fake-block AST loader from ``test_residual_streams.py`` (the
``test_block_timestep.py`` pattern: the real ``BiRWKV7ForMaskedDiffusion``
class compiled without fla's Triton kernels).

Properties under test (the loop arm's whole safety case):

1. **Exact identity at init.** The per-pass gate is exactly 0 (``tanh(0)``),
   so any number of extra weight-tied passes must leave the logits bit-
   identical to the legacy single-pass forward — including the
   depth-extrapolation path (reps > trained).
2. **Grads flow to the gate AT init** — no bilinear zero, no one-step delay
   (the GRPO/LoRA lesson: gate on gradient flow, never on loss != 0).
3. **Bounded gate** (house discipline: tanh·loop_scale), **range validation**,
   and the ``loop.`` state-dict prefix that ``_merge_latent_keys``
   whitelists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_SCALE_DIR = Path(__file__).resolve().parents[1]
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))
_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

from models.residual_streams import BackboneLoopControl  # noqa: E402
from test_residual_streams import (  # noqa: E402
    _Cfg,
    _probe_ids,
    check,
    load_model_cls,
)

FAIL: list[str] = []


def main() -> int:
    torch.manual_seed(0)

    print("u1 gate identity and bounds")
    lp = BackboneLoopControl(2, 6, reps=1)
    check("gates() is exactly 0 at init", float(lp.gates().abs().max()) == 0.0,
          f"{lp.gates().tolist()}")
    with torch.no_grad():
        lp.gates_raw.fill_(100.0)
    check("gate bounded by loop_scale at large raw",
          float(lp.gates().abs().max()) == 1.0, f"{lp.gates().tolist()}")
    lp2 = BackboneLoopControl(0, 4, reps=2, loop_scale=0.5)
    with torch.no_grad():
        lp2.gates_raw.fill_(-100.0)
    check("two reps, custom scale", tuple(lp2.gates().shape) == (2,)
          and float(lp2.gates().abs().max()) == 0.5, f"{lp2.gates().tolist()}")
    check("reps property", lp2.reps == 2, f"{lp2.reps}")

    print("u2 validation")
    for bad_range in ((6, 2), (-1, 3)):
        try:
            BackboneLoopControl(bad_range[0], bad_range[1], 1)
            check(f"range={bad_range} rejected at construction", False, "no raise")
        except ValueError:
            check(f"range={bad_range} rejected at construction", True, "ValueError")
    try:
        BackboneLoopControl(0, 4, 3)
        check("reps=3 rejected", False, "no raise")
    except ValueError:
        check("reps=3 rejected", True, "ValueError")

    print("u3 state-dict round trip: lo/hi/gates_raw persist")
    lp3 = BackboneLoopControl(16, 32, reps=2)
    sd = lp3.state_dict()
    check("standalone keys are {lo,hi,gates_raw}", sorted(sd) == ["gates_raw", "hi", "lo"],
          f"{sorted(sd)}")
    check("lo/hi persist as int64", sd["lo"].dtype == torch.int64
          and int(sd["lo"]) == 16 and int(sd["hi"]) == 32, f"{sd['lo'].tolist()}")

    # ----------------------------------------------------------------------
    # Integration: the REAL forward
    # ----------------------------------------------------------------------
    print("i1 loop ON at init == legacy forward (torch.equal)")
    cls = load_model_cls()
    cfg = _Cfg(vocab=65536, hidden=64, layers=6)
    ids = _probe_ids(cfg.vocab_size)
    torch.manual_seed(0)
    m_legacy = cls(cfg)
    with torch.no_grad():
        ref = m_legacy(ids)
    torch.manual_seed(0)
    m = cls(cfg)
    _ = m.attach_backbone_loop((3, 6), loop_reps=2)  # loop the last 3 of 6 layers
    with torch.no_grad():
        on = m(ids)
    check("logits bit-identical to legacy at init", torch.equal(ref, on),
          f"max|d|={float((ref - on).abs().max()):.3e}")

    print("i2 depth extrapolation (reps > trained) is ALSO identity at init")
    with torch.no_grad():
        ext4 = m(ids, loop_reps_override=4)
        ext9 = m(ids, loop_reps_override=9)
    check("override=4 bit-identical", torch.equal(ref, ext4), "")
    check("override=9 bit-identical", torch.equal(ref, ext9), "")
    check("override=0 disables the loop", torch.equal(ref, m(ids, loop_reps_override=0)), "")

    print("i3 grads flow to gates_raw AT init (gate on grad flow, not loss)")
    torch.manual_seed(0)
    m = cls(cfg)
    _ = m.attach_backbone_loop((3, 6), loop_reps=1)
    m(ids).sum().backward()
    g = m.loop.gates_raw.grad
    check("gates_raw.grad exists and is nonzero",
          g is not None and float(g.abs().sum()) > 0.0,
          f"{None if g is None else float(g.abs().sum()):.3e}")

    print("i4 the loop actually changes the output once the gate opens")
    torch.manual_seed(0)
    m = cls(cfg)
    _ = m.attach_backbone_loop((3, 6), loop_reps=1)
    with torch.no_grad():
        m.loop.gates_raw.fill_(2.0)  # tanh(2) ~= 0.96
        live = m(ids)
    check("open gate changes the output", not torch.equal(ref, live),
          f"max|d|={float((ref - live).abs().max()):.3e}")

    print("i5 weight-tying: the looped range re-uses the SAME modules")
    torch.manual_seed(0)
    m = cls(cfg)
    _ = m.attach_backbone_loop((3, 6), loop_reps=2)
    block_params = sum(id(p) for p in m._blocks[3].parameters())  # noqa: SLF001
    check("loop adds ZERO new block modules",
          len(list(m._blocks)) == 6  # noqa: SLF001
          and sum(p.numel() for p in m.loop.parameters()) == 2,
          "same 6 blocks; only the [r] gate vector is new")

    print("i6 model-level validation mirrors the unit checks")
    torch.manual_seed(0)
    m = cls(cfg)
    try:
        _ = m.attach_backbone_loop((0, 99), 1)
        check("attach range outside [0,L) rejected", False, "no raise")
    except ValueError:
        check("attach range outside [0,L) rejected", True, "ValueError")
    try:
        _ = m.attach_backbone_loop((0, 3), 1)
        _ = m.attach_backbone_loop((0, 3), 1)
        check("double attach rejected", False, "no raise")
    except RuntimeError:
        check("double attach rejected", True, "RuntimeError")

    print()
    if FAIL:
        print(f"BACKBONE-LOOP SUITE: {len(FAIL)} FAIL -> {FAIL}")
        return 1
    print("BACKBONE-LOOP SUITE: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
