"""Add the executed 0.4B adaptation phase to the paper's protocol and run matrix.

The manuscript ships a 48-row ``training_run_matrix.csv`` in which every row is
``not_run``, and ``validate_protocol.py`` enforces that an unrun row carries no
alleged measurement.  This module adds a **new** phase for work that was actually
done -- the twelve task7 0.4B adaptation runs -- and leaves all 48 original rows
untouched.  That separation is the point: the paper's core claims rest on the
from-scratch matrix, and nothing here may be mistaken for it.

Why this lineage and no other
-----------------------------
``validate_protocol.py:43-46`` requires a phase's rows to be *exactly* the cross
product of its arms and the protocol's three global training seeds.  Only the
task7 matrix satisfies that: 4 arms x {17,29,43}, all twelve finished at the same
step with the same token budget.  The 2.9B lineage has one seed, and the second
0.4B lineage has four finished runs out of twelve.  Neither can enter a phase
without either inventing rows or widening the seed rule, so both stay in the
descriptive table that :mod:`lrwkv_evidence.tracks` writes.

Why the loop arms get their own arm ids
--------------------------------------
a3 and a5 recycle layers [12,24) for an extra pass.  No protocol arm has a
recycled-depth field, so folding them into C6/C5 would let the validator certify
a phase in which two rows labelled C6 execute different amounts of compute.  They
are declared as ``C6_loop``/``C5_loop`` with ``derived_from`` pointing at the arm
they deviate from, which keeps the deviation machine-readable instead of buried
in a footnote.

What is deliberately left blank
-------------------------------
``task7-a2-s17`` has no surviving training log (the other eleven are in
``outputs_task7_matrix/logs/``), so its ``measured_gpu_hours`` is empty.  Its
siblings ran 36.7 and 37.0 GPU-h and interpolating would be indistinguishable
from a measurement in the CSV.  The validator permits a blank here -- it only
forbids measurements on *unrun* rows -- so the honest cell is the empty one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Final

try:
    from . import tracks
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lrwkv_evidence import tracks

PROTOCOL_DIR: Final = Path(__file__).resolve().parent.parent / "paper" / "protocol"

#: Wall-clock GPU-hours per run, derived from each run's own log: the span from
#: the first timestamped line to the last, times the 8 ranks the log declares
#: (``world=8``).  Each log was checked for exactly one ``BiRWKV masked-diffusion:
#: run=`` banner and one ``training complete``, so no row double-counts a restart.
#: This is *occupancy*, not utilization, which is the quantity a compute-budget
#: table should report.
GPU_HOURS: Final = {
    ("a1", 17): 26.620, ("a1", 29): 26.967, ("a1", 43): 26.900,
    ("a2", 17): None,   ("a2", 29): 36.996, ("a2", 43): 36.738,
    ("a3", 17): 53.976, ("a3", 29): 52.516, ("a3", 43): 52.847,
    ("a5", 17): 38.836, ("a5", 29): 38.998, ("a5", 43): 39.091,
}

#: Median ``step/s`` from each arm's logs.  Recorded because it is an independent
#: check on the arm mapping rather than a performance number: a1 (forward-only)
#: runs 1.41x faster than a2 (bidirectional), and a5 (forward+loop) 1.40x faster
#: than a3 (bidirectional+loop), at identical parameter counts.  A mislabelled
#: direction would break that pattern, so these rates are evidence that the
#: forward-only arms genuinely skip the backward mixer.
MEDIAN_STEP_S: Final = {"a1": 0.1653, "a2": 0.1185, "a3": 0.0817, "a5": 0.1123}

NEW_ARMS: Final = {
    "C5_loop": {
        "mixer": "rwkv7",
        "objective": "absorbing_diffusion",
        "direction": "forward",
        "derived_from": "C5",
        "deviation": "weight-tied depth recycling over layers [12,24), one extra "
                     "pass; increases compute per token at fixed parameter count",
    },
    "C6_loop": {
        "mixer": "rwkv7",
        "objective": "absorbing_diffusion",
        "direction": "bidirectional",
        "derived_from": "C6",
        "deviation": "weight-tied depth recycling over layers [12,24), one extra "
                     "pass; increases compute per token at fixed parameter count",
    },
}

PHASE: Final = {
    "id": "adaptation_0p4b",
    "required_for_core_claims": False,
    "arms": ["C5", "C6", "C5_loop", "C6_loop"],
    "parameter_target": 591_054_848,
    "tokens_per_run": 2_000_683_008,
    "runs": 12,
    "aggregate_token_exposures": 12 * 2_000_683_008,
    "source": "adaptation_from_released_rwkv7_0p4b",
    "track": "adaptation",
    "initialized_from": "models/rwkv7-0.4B",
    "exposure_disclosure": tracks.EXPOSURE_DISCLOSURE,
    "not_a_substitute_for": "core",
    "scope_note": (
        "Executed adaptation runs, not the controlled from-scratch matrix. Each "
        "run continues a released RWKV-7 0.4B checkpoint for 1908 steps x "
        "1,048,576 tokens on 8xH100, so a contrast against an autoregressive "
        "reference here is 'adaptation vs released', never equal-exposure. The "
        "core / scale / cross_corpus phases remain not_run and the paper's core "
        "claims are scoped accordingly."
    ),
}

#: task7 arm label -> (phase arm id, actual stored parameters).  The loop arms
#: store three extra scalars (``loop.lo``, ``loop.hi``, ``loop.gates_raw``), which
#: is why ``actual_parameters`` differs from ``parameter_target`` by exactly 3 on
#: those rows -- a real measurement, not a rounding artifact.
ARM_ROWS: Final = {
    "a1": ("C5", 591_054_848),
    "a2": ("C6", 591_054_848),
    "a3": ("C6_loop", 591_054_851),
    "a5": ("C5_loop", 591_054_851),
}


def build_rows() -> list[dict[str, str]]:
    """The twelve CSV rows, with every measurement sourced from the registry."""
    by_id = {t.track_id: t for t in tracks.TRACKS_0P4B}
    rows = []
    for arm, (phase_arm, actual_params) in ARM_ROWS.items():
        for seed in (17, 29, 43):
            t = by_id[f"a04_{arm}_s{seed}"]
            if t.tokens_seen != PHASE["tokens_per_run"]:
                raise ValueError(
                    f"{t.track_id}: registry says {t.tokens_seen} tokens but the "
                    f"phase declares {PHASE['tokens_per_run']}; the validator "
                    f"compares them and would refuse")
            if t.params_stored != actual_params:
                raise ValueError(
                    f"{t.track_id}: registry stores {t.params_stored} parameters, "
                    f"this table says {actual_params}")
            hours = GPU_HOURS[(arm, seed)]
            rows.append({
                "run_id": f"adaptation_0p4b_{phase_arm}_s{seed}",
                "phase": PHASE["id"],
                "arm": phase_arm,
                "seed": str(seed),
                "parameter_target": str(PHASE["parameter_target"]),
                "token_budget": str(PHASE["tokens_per_run"]),
                "source": PHASE["source"],
                "status": "ok",
                "actual_parameters": str(actual_params),
                # Blank, not zero and not interpolated, when the log is gone.
                "measured_gpu_hours": "" if hours is None else f"{hours:.3f}",
                "checkpoint_sha256": t.sha256,
            })
    return rows


def patch_protocol(path: Path, dry_run: bool = False) -> dict:
    """Add the arms and the phase.  Idempotent: re-running changes nothing.

    Two *disjoint* sets of arms are added, and the difference matters:

    * :data:`NEW_ARMS` (``C5_loop``/``C6_loop``) are **phase arms**.  They appear in
      ``PHASE["arms"]``, so ``validate_protocol.py:43-46`` requires one matrix row
      per arm per training seed -- :func:`build_rows` writes exactly those.
    * :data:`tracks.BASELINE_ARMS` (the ``*_ar`` ids) are **descriptive arms** for
      released checkpoints.  They belong to no phase and have no matrix row.  They
      are needed only because ``validate_tracks`` refuses a registry row whose
      ``paper_arm`` is unknown, and because a later phase or a rendered table that
      names one must find its mixer/objective/direction somewhere machine-readable.

    Putting a descriptive arm into a phase would make the validator demand three
    seeded rows for a checkpoint nobody trained, so that is checked here rather
    than left to be discovered by ``--self-test``.
    """
    p = json.loads(path.read_text(encoding="utf-8"))
    baseline = tracks.BASELINE_ARMS
    overlap = set(NEW_ARMS) & set(baseline)
    if overlap:
        raise ValueError(
            f"{sorted(overlap)} is declared both as a phase arm and as a "
            f"descriptive baseline arm; one of the two tables is wrong")
    added_arms = [a for a in NEW_ARMS if a not in p["arms"]]
    for arm in added_arms:
        p["arms"][arm] = dict(NEW_ARMS[arm])
    added_baseline = []
    for arm, spec in baseline.items():
        if p["arms"].get(arm) != spec:
            p["arms"][arm] = dict(spec)
            added_baseline.append(arm)
    # A descriptive arm in any phase would be a demand for matrix rows that do not
    # and should not exist.  PHASE is ours; the other phases came with the paper.
    for ph in [*p["phases"], PHASE]:
        claimed = sorted(set(ph["arms"]) & set(baseline))
        if claimed:
            raise ValueError(
                f"phase {ph['id']} lists descriptive baseline arm(s) {claimed}. "
                f"validate_protocol.py requires one matrix row per (arm, seed) for "
                f"every phase arm, and these are released checkpoints with no "
                f"training runs, so the phase could never be satisfied")
    unknown = [t.paper_arm for t in tracks.ALL_TRACKS
               if t.paper_arm not in p["arms"]]
    if unknown:
        raise ValueError(
            f"registry rows name arm(s) {sorted(set(unknown))} that this patch does "
            f"not add to protocol.json; validate_protocol.py:50 would refuse any "
            f"matrix row naming them")
    existing = {ph["id"] for ph in p["phases"]}
    added_phase = PHASE["id"] not in existing
    if added_phase:
        p["phases"].append(dict(PHASE))
    else:
        for i, ph in enumerate(p["phases"]):
            if ph["id"] == PHASE["id"]:
                p["phases"][i] = dict(PHASE)
    if not dry_run:
        path.write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
    return {"arms_added": added_arms, "baseline_arms_added": added_baseline,
            "phase_added": added_phase,
            "phases": [ph["id"] for ph in p["phases"]],
            "arms": sorted(p["arms"])}


def patch_matrix(path: Path, dry_run: bool = False) -> dict:
    """Append the twelve rows, replacing any previous copy of this phase.

    The 48 original rows are rewritten byte-for-byte from what was read, so a
    second run cannot drift them.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        old = list(reader)
    if fields is None:
        raise ValueError(f"{path}: no header")
    kept = [r for r in old if r["phase"] != PHASE["id"]]
    new = build_rows()
    unknown = [k for k in new[0] if k not in fields]
    if unknown:
        raise ValueError(f"{path}: new rows carry undeclared columns {unknown}")
    out = kept + new
    if not dry_run:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in out:
                w.writerow(r)
    return {"original_rows_kept": len(kept), "rows_written": len(new),
            "replaced_existing": len(old) - len(kept),
            "total_rows": len(out),
            "not_run_rows": sum(1 for r in out if r["status"] == "not_run")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol-dir", type=Path, default=PROTOCOL_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    tracks.check_ids_unique()
    tracks.check_directional_accounting()
    report = {
        "protocol": patch_protocol(args.protocol_dir / "protocol.json", args.dry_run),
        "matrix": patch_matrix(args.protocol_dir / "training_run_matrix.csv", args.dry_run),
        "median_step_s": MEDIAN_STEP_S,
        "gpu_hours_total": round(sum(v for v in GPU_HOURS.values() if v), 1),
        "gpu_hours_missing_log": [f"task7-{a}-s{s}" for (a, s), v
                                  in GPU_HOURS.items() if v is None],
        "dry_run": args.dry_run,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
