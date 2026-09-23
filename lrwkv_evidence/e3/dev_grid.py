"""The E3 dev-only decoder grid, and the freeze that must precede any confirmation run.

The plan's Q2 requires the sampler operating point -- steps x tau x commit rule -- to be
chosen on **dev data** and frozen into ``protocol.json["quality"]`` *before* a single
confirmation cell is scored.  The reason is not bookkeeping: the 324-cell grid is the
confirmation set, so choosing the sampler on it would be choosing it on the test data,
and the resulting number would be an operating point selected for the panel it is
reported on.  Unlike the T=0 batching gate, this hazard is not invisible -- it is
invisible *after the fact*, because two runs at different operating points render as the
same cell.

Everything here is dev-only by construction:

* the cells are built under the generator's own ``dev`` split, whose data seeds are
  (201, 202, 203) and whose key tag is ``dv`` -- disjoint from the confirmation grid's
  ``iclr2027_grid`` split (seeds 101-105, tag ``g7``).  A dev instance therefore cannot
  be reproduced as a confirmation instance even by accident.
* :func:`panel_cells` deliberately reuses the *same* three families, length and
  difficulty axes as the confirmation grid.  A calibration panel drawn from easier cells
  would pick the operating point that suits the easy cells.

The selection rule is stated here, before any measurement exists, and
:func:`select_decoder` implements exactly it.  A rule written after seeing the sweep
table is a rule fitted to it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final, Iterable

from lrwkv_evidence.e3 import grid324 as G

#: The split the calibration draws from.  The generator's own; not registered here.
DEV_SPLIT: Final = "dev"

#: The confirmation grid's lengths are 16K/32K/64K.  Calibration runs at the *smallest*
#: primary length because the decoder's behaviour at 16K is the one that has to hold
#: for the panel to be interpretable, and because a 64K sweep costs 4x as much for an
#: operating point that is then applied at 16K.
DEV_LENGTH: Final = 16384

#: One representative difficulty setting per family: mid-canvas evidence with the
#: hardest distractor class and a mid load.  Chosen once, from the confirmation grid's
#: own axes, so the panel is not a second tuning surface.
DEV_SETTINGS: Final = ((50, 8, "similar"),)

#: Instances per (family, setting, seed).  Small on purpose: the sweep is
#: steps x tau x sampler, and the panel's job is to *rank* configurations, not to
#: measure accuracy to a tight interval.
DEV_INSTANCES_PER_SEED: Final = 8

#: The dev split's declared seeds.  Read from the generator rather than restated: a
#: hard-coded (201, 202, 203) would keep working after the generator reassigned them,
#: and would then be drawing from whatever split now owns those seeds.
def dev_seeds() -> tuple[int, ...]:
    hop, _, _ = G._import_generator()
    seeds = tuple(hop.SPLIT_DATA_SEEDS[DEV_SPLIT])
    if not seeds:
        raise SystemExit(f"the generator declares no data seeds for {DEV_SPLIT!r}")
    return seeds


#: The sweep.  ``steps`` is NFE and is the whole cost of a diffusion arm; ``tau`` is the
#: temperature the sampler reveals at; ``sampler`` is the reveal law (the theorem's
#: matched-Bernoulli law versus the confidence-greedy one the legacy runs used).
DECODER_GRID: Final = {
    "sampler": ("matched_bernoulli", "confidence"),
    "steps": (8, 16, 32, 64),
    "tau": (0.7, 1.0),
}
DECODER_KEYS: Final = ("sampler", "steps", "tau", "commit_order")

#: Fixed, not swept: the commit order the banked legacy runs used, so the sweep varies
#: the three axes the plan names and nothing else.
COMMIT_ORDER: Final = "confidence"


def sweep_configs() -> list[dict]:
    """Every configuration in the declared grid, in a stable order."""
    out = []
    for sampler in DECODER_GRID["sampler"]:
        for steps in DECODER_GRID["steps"]:
            for tau in DECODER_GRID["tau"]:
                out.append({"sampler": sampler, "steps": int(steps), "tau": float(tau),
                            "commit_order": COMMIT_ORDER})
    return out


def config_id(config: dict) -> str:
    return (f"{config['sampler']}_s{int(config['steps'])}"
            f"_t{config['tau']:g}_{config['commit_order']}")


def panel_cells() -> list[G.Cell]:
    """The calibration panel's cells: one per family per declared setting."""
    cells = []
    for family in G.PAPER_FAMILIES:
        for position, load, distractor in DEV_SETTINGS:
            cells.append(G.Cell(family, DEV_LENGTH, position, load, distractor))
    return cells


def build_panel(out: Path, *, instances_per_seed: int = DEV_INSTANCES_PER_SEED,
                write: bool = True) -> dict:
    """Build the dev panel and, optionally, write it.

    Refuses if any cell does not build: an ``unsupported`` cell in a *calibration* panel
    is not reported the way a confirmation cell is -- it would silently shrink the panel
    the operating point is chosen on, and the chosen point would then be the one that
    suits the cells that happened to build.
    """
    hop, token_prompt, registry = G._import_generator()
    encoder = registry.TrieEncoder.from_vocab(str(G.VOCAB))
    pools = G._PermutingPoolFactory(hop._Pool)
    seeds = dev_seeds()
    records = []
    for cell in panel_cells():
        record = G.build_cell(hop, token_prompt, encoder, cell,
                              instances_per_seed, pools, DEV_SPLIT, seeds)
        if record["status"] != G.STATUS_OK:
            raise SystemExit(
                f"{cell.cell_id}: the dev panel's cell is "
                f"{record['status']} ({record.get('unsupported_reason')}). A "
                f"calibration panel is not allowed to lose a family -- the operating "
                f"point would then be chosen on the families that built.")
        records.append(record)
    report = {
        "split": DEV_SPLIT, "length": DEV_LENGTH,
        "seeds": list(dev_seeds()), "instances_per_seed": instances_per_seed,
        "cells": [r["cell_id"] for r in records],
        "instances": sum(r["total_instances"] for r in records),
        "settings": [list(s) for s in DEV_SETTINGS],
        "configs": [config_id(c) for c in sweep_configs()],
        "wrote": bool(write),
    }
    if write:
        out.mkdir(parents=True, exist_ok=True)
        for record in records:
            (out / f"{record['cell_id']}.json").write_text(
                json.dumps(record, indent=1) + "\n", encoding="utf-8")
        (out / "dev_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def select_decoder(measurements: Iterable[dict]) -> dict:
    """The frozen operating point, by the rule declared above.

    The rule: **highest mean dev accuracy wins; ties go to the fewer steps, then to the
    matched-Bernoulli sampler, then to the lower tau.**  Steps first because it is the
    only axis that costs GPU time, and a tie is not evidence for paying 8x for it.  The
    sampler preference is the theorem's reveal law over the legacy confidence-greedy one
    -- stated here so that a tie does not silently decide which law the paper reports.

    Ties are not hypothetical at this panel size, which is exactly why the order is
    fixed in advance: a rule chosen after reading the table is fitted to the table.
    """
    rows = list(measurements)
    if not rows:
        raise SystemExit("no decoder measurements to select from")
    for row in rows:
        missing = [k for k in DECODER_KEYS if row.get(k) in (None, "")]
        if missing:
            raise SystemExit(
                f"a measurement row is missing {missing}; it cannot be ranked against "
                f"a complete one, and defaulting the gap would invent a configuration.")
        if row.get("accuracy") is None:
            raise SystemExit(
                f"{config_id(row)} has no accuracy. A configuration that did not run is "
                f"not a configuration that scored zero; ranking it as one would prefer "
                f"whatever ran first.")

    sampler_rank = {name: i for i, name in enumerate(DECODER_GRID["sampler"])}

    def key(row: dict) -> tuple:
        return (-float(row["accuracy"]), int(row["steps"]),
                sampler_rank.get(row["sampler"], len(sampler_rank)),
                float(row["tau"]), config_id(row))

    ranked = sorted(rows, key=key)
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return {
        **{k: winner[k] for k in DECODER_KEYS},
        "selected_by": "highest dev accuracy; ties: fewer steps, then matched_bernoulli,"
                       " then lower tau",
        "dev_accuracy": float(winner["accuracy"]),
        "dev_runner_up": ({"config": config_id(runner_up),
                           "accuracy": float(runner_up["accuracy"])}
                          if runner_up else None),
        "dev_margin": (float(winner["accuracy"]) - float(runner_up["accuracy"])
                       if runner_up else None),
        "dev_panel": "lrwkv_evidence.e3.dev_grid",
    }


def freeze(protocol_path: Path, decoder: dict) -> dict:
    """Write the operating point into ``protocol["quality"]["decoder"]`` and read back.

    Read back from the file rather than from the dict that was written: the artifact is
    what a later run will consult, and a return value that echoes the intent is the
    [[an-artifact-on-disk-is-not-evidence-about-the-code]] shape.
    """
    path = Path(protocol_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    quality = protocol.setdefault("quality", {})
    if quality.get("decoder") not in (None, {}):
        raise SystemExit(
            f"{path} already carries quality.decoder="
            f"{json.dumps(quality['decoder'])[:200]}. The operating point is frozen "
            f"before the confirmation run; overwriting it after one has been scored is "
            f"how a sampler gets chosen on the test panel.")
    quality["decoder"] = dict(decoder)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    written = json.loads(path.read_text(encoding="utf-8"))["quality"]["decoder"]
    if written != decoder:
        raise SystemExit(f"{path}: wrote {decoder} and read back {written}")
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=G.OUT_ROOT.parent / "lc_dev_panel")
    ap.add_argument("--instances-per-seed", type=int, default=DEV_INSTANCES_PER_SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and check the panel, write nothing")
    ap.add_argument("--freeze", type=Path, default=None,
                    help="a JSON list of measurement rows; selects and writes the "
                         "operating point into --protocol")
    ap.add_argument("--protocol", type=Path,
                    default=Path(__file__).resolve().parents[2] / "paper" / "protocol"
                    / "protocol.json")
    args = ap.parse_args(argv)

    if args.freeze is not None:
        rows = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
        decoder = select_decoder(rows)
        written = freeze(args.protocol, decoder)
        print(json.dumps({"selected": decoder, "written": written}, indent=2))
        return 0

    report = build_panel(args.out, instances_per_seed=args.instances_per_seed,
                         write=not args.dry_run)
    report["configs"] = sweep_configs()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
