"""Collect every measured reading this project has, with provenance, into one record.

The paper cites numbers from six different kinds of artifact produced over weeks by
different harnesses, and the failure this module exists to prevent is a number in the
manuscript that cannot be traced back to a file -- or that traces back to a file whose
``n`` is not what the table implies.  So the record is **generated, never transcribed**:
re-running this module rebuilds it from the artifacts, and ``--check`` re-hashes every
source so a record that no longer describes the bytes on disk is caught rather than
trusted.

What it deliberately records alongside each value:

* **``n`` from the file**, plus a ``subset`` hazard when ``n`` is below the benchmark's
  canonical item count.  A mean over a prefix and a mean over the benchmark render as the
  same table cell and are different statistics
  ([[a-mean-over-a-subset-is-a-different-statistic]]).
* **the scoring arm**, because a reading labelled with the wrong arm came from a
  different code path even when the number looks ordinary.
* **the file's sha256**, so the record names bytes rather than a path that may be
  rewritten by a later run into the same directory.
* **the ``unpinned`` provenance triple** as a recorded fact, not as a verdict: the prompt
  tree and label convention are hash-pinned in the tree manifests, while the harness
  revision those runs used is not.

Nothing here interprets.  Hazards are recorded as data and summarised; deciding what
they mean for a claim belongs to whoever writes the claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

#: Every mount this collector reads.  ``G`` only: the pods see nothing else, so a
#: reading that lives outside it could not be reproduced by the same machinery.
G: Final = Path(os.environ.get("LRWKV_G", "/inspire/hdd/global_user/zhangjiaquan-253108540222"))
OUTPUTS: Final = G / "outputs_long_rwkv_iclr2027"

#: Canonical item counts, used only to flag a subset.  Sourced from the tree manifests
#: and the benchmark cards; a mismatch is reported, never silently corrected.
CANONICAL_N: Final = {
    "mmlu_test": 14042, "mmlu_validation": 1531, "hellaswag": 10042, "race": 4887,
    "mmlu_redux": 5700, "siqa": 1954, "story_cloze": 1871, "piqa": 1838,
    "commonsenseqa": 1221, "sciq": 1000, "openbookqa": 500, "gpqa_diamond": 198,
    "gsm8k": 1319, "humanevalplus": 164, "mbppplus": 378,
}

#: A directory name is ``e1_<dataset>_<track>`` or ``e1b_<dataset>_<track>``.  The
#: dataset token is matched longest-first so ``mmlu_redux`` does not parse as ``mmlu``.
_DATASET_TOKENS: Final = sorted(CANONICAL_N, key=len, reverse=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_run_dir(name: str) -> tuple[str, str, str] | None:
    """``e1_mmlu_test_a29_c6_f2_s14000`` -> ``(tag, dataset, track)``."""
    m = re.match(r"^(e1b?)_(.+)$", name)
    if not m:
        return None
    tag, rest = m.group(1), m.group(2)
    for dataset in _DATASET_TOKENS:
        if rest.startswith(dataset + "_"):
            return tag, dataset, rest[len(dataset) + 1:]
    return None


def _metrics(merged: dict) -> list[dict]:
    out = []
    for m in merged.get("metrics") or []:
        out.append({"arm": m.get("arm"), "metric": m.get("metric"),
                    "value": m.get("mean"), "ci95": m.get("ci95"),
                    "n_documents": m.get("n_documents")})
    return out


def _hazards(record: dict, merged: dict, canonical: int | None) -> list[str]:
    hz = []
    n = merged.get("record_count") or 0
    if canonical and n and n < canonical:
        hz.append(f"subset: {n} of {canonical} canonical items")
    # The arm check is scoped to multichoice on purpose. `fwdce`/`raw` are the two names
    # for the one *multichoice* statistic; the likelihood panels legitimately use
    # `mc070`, `mc_elbo`, `iter16`, `ddpm100` and a dozen sampler arms, and flagging
    # those as mislabelled produced 1,100 false positives on the first run -- which is
    # exactly how a hazard report trains its reader to ignore it.
    if (record.get("source_class") == "campaign_multichoice"
            and record.get("scoring_arm") not in ("fwdce", "raw")):
        hz.append(f"scoring arm {record.get('scoring_arm')!r} is neither "
                  f"fwdce nor raw")
    for key in ("registry_hash", "profile_sha256", "condition_profile_hash"):
        if str(merged.get(key, "")).startswith("unpinned"):
            hz.append(f"{key}=unpinned")
            break
    if re.search(r"probe_copy", str(record["path"])):
        hz.append("source filename carries a probe_copy provenance marker")
    return hz


def collect_campaign(root: Path = OUTPUTS) -> tuple[list[dict], list[str]]:
    """Every merged multichoice aggregate under the campaign outputs."""
    readings, notes = [], []
    for d in sorted(p for p in root.glob("e1*_*") if p.is_dir()):
        parsed = split_run_dir(d.name)
        if not parsed:
            notes.append(f"{d.name}: not a <tag>_<dataset>_<track> directory")
            continue
        tag, dataset, track = parsed
        for mfile in sorted((d / "merged").glob("*.json")):
            try:
                merged = json.loads(mfile.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                notes.append(f"{mfile}: unreadable ({exc})")
                continue
            for m in _metrics(merged):
                rec = {
                    "source_class": "campaign_multichoice",
                    "tag": tag, "dataset": dataset, "track": track,
                    "scoring_arm": m["arm"], "metric": m["metric"],
                    "value": m["value"], "ci95": m["ci95"],
                    "n": merged.get("record_count"),
                    "task_dir": (merged.get("header") or {}).get("task_dir")
                    or merged.get("task_dir"),
                    "checkpoint": merged.get("checkpoint"),
                    "path": str(mfile), "sha256": sha256_file(mfile),
                }
                rec["hazards"] = _hazards(rec, merged, CANONICAL_N.get(dataset))
                if merged.get("timing"):
                    rec["timing"] = merged["timing"]
                readings.append(rec)
    return readings, notes


def collect_probes(root: Path = OUTPUTS) -> tuple[list[dict], list[str]]:
    """Q1 qualification receipts: existence is the receipt, so it is recorded as one."""
    out, notes = [], []
    for d in sorted(root.glob("q1probe_*")):
        if not d.is_dir():
            continue
        hits = [p for p in d.glob("*.probe.json") if p.stat().st_size > 0]
        out.append({
            "source_class": "campaign_probe",
            "track": d.name.replace("q1probe_mmlu_validation_", ""),
            "qualified": bool(hits),
            "path": str(sorted(hits)[-1]) if hits else None,
            "sha256": sha256_file(sorted(hits)[-1]) if hits else None,
            "n_receipts": len(hits),
            "hazards": ([] if hits else ["no non-empty probe.json: not qualified"]),
        })
        if not hits:
            notes.append(f"{d.name}: no non-empty probe.json (not qualified)")
    return out, notes


def collect_systems(root: Path = G / "outputs_efficiency_probe") -> tuple[list[dict], list[str]]:
    """The single-device memory/latency probe, one row per (model, context)."""
    out, notes = [], []
    if not root.is_dir():
        return out, [f"{root} absent"]
    for f in sorted(root.glob("*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{f}: unreadable ({exc})")
            continue
        for row in payload.get("rows") or []:  # noqa: PLR1702
            out.append({
                "source_class": "systems_probe", "path": str(f),
                "sha256": sha256_file(f), "model": row.get("model"),
                "family": row.get("family"), "objective": row.get("objective"),
                "nfe": row.get("nfe"), "context": row.get("context"),
                "status": row.get("status"), "wall_ms": row.get("wall_ms"),
                "tokens_per_second": row.get("tokens_per_second"),
                "peak_reserved_bytes": row.get("peak_reserved_bytes"),
                "peak_allocated_bytes": row.get("peak_allocated_bytes"),
                "hazards": ([] if row.get("status") == "ok"
                            else [f"status={row.get('status')}"]),
            })
    return out, notes


def collect_banked(root: Path = G) -> tuple[list[dict], list[str]]:
    """Older, separately-produced readings.

    Kept in the record but never merged with the campaign panels: they come from a
    different harness revision and, for MMLU, from a tree the manuscript's own
    convention rules out.  Recording them with that fact attached is the point --
    omitting them would make the record silent about numbers that exist and have been
    cited before.
    """
    out, notes = [], []
    patterns = ["cap_eval_*/merged/*.json", "cap_lm_*/merged/*.json",
                "cap_lm_*/*/merged/*.json", "m2_baseline_triangle/*/merged/*.json",
                "m2_baseline_triangle/*/*.json"]
    seen: set[str] = set()
    for pat in patterns:
        for f in sorted(root.glob(pat)):
            if str(f) in seen or not f.is_file():
                continue
            seen.add(str(f))
            try:
                payload = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a non-JSON file in a data tree is normal
                continue
            metrics = _metrics(payload)
            if not metrics:
                continue
            for m in metrics:
                rec = {
                    "source_class": "banked_readings", "path": str(f),
                    "sha256": sha256_file(f),
                    "model": payload.get("checkpoint") or f.parts[-4],
                    "metric": m["metric"], "scoring_arm": m["arm"],
                    "value": m["value"], "ci95": m["ci95"],
                    "n": payload.get("record_count"),
                    "task_dir": (payload.get("header") or {}).get("task_dir")
                    or payload.get("task_dir"),
                    "hazards": [],
                }
                td = str(rec["task_dir"] or "")
                if "preprocessed_data/mmlu/validation" in td:
                    rec["hazards"].append(
                        "task_dir is the multi-token option-pseudolikelihood tree, "
                        "which the manuscript's one-token convention rules out")
                rec["hazards"] += _hazards(rec, payload, None)
                out.append(rec)
    return out, notes


def collect_inventories(root: Path = G / "capability_eval_data") -> dict:
    """Artifact inventories that are not scores but are cited as facts."""
    inv: dict[str, Any] = {}
    for name, rel in (("grid324", "lc_grid324/grid_manifest.json"),
                      ("dev_panel", "lc_dev_panel/dev_manifest.json")):
        p = root / rel
        if p.is_file():
            inv[name] = {"path": str(p), "sha256": sha256_file(p),
                         **json.loads(p.read_text(encoding="utf-8"))}
    return inv


#: Hazards that belong to a *source file*, not to one reading out of it.  Every reading
#: the file yields carries the same one, so reporting them per row turns the report into
#: a wall of identical lines -- measured on the first full run: 2,941 of 2,999 readings
#: flagged `unpinned` and 1,302 flagged `probe_copy`, which is how a hazard report
#: teaches its reader to skip it.  These are lifted to :func:`source_facts` and removed
#: from the per-reading hazards, whose remaining entries are genuinely about the row.
_FILE_LEVEL_HAZARDS: Final = ("registry_hash=unpinned", "profile_sha256=unpinned",
                              "condition_profile_hash=unpinned",
                              "source filename carries a probe_copy provenance marker")


def source_facts(readings: list[dict]) -> dict:
    """Per-source-file facts that are identical for every reading the file yields."""
    files: dict[str, dict] = {}
    for r in readings:
        path = r.get("path")
        if not path:
            continue
        facts = files.setdefault(path, {"source_class": r.get("source_class"),
                                        "readings": 0, "facts": []})
        facts["readings"] += 1
        for h in r.get("hazards") or []:
            if h in _FILE_LEVEL_HAZARDS and h not in facts["facts"]:
                facts["facts"].append(h)
        r["hazards"] = [h for h in (r.get("hazards") or [])
                        if h not in _FILE_LEVEL_HAZARDS]
    return files


def collect() -> dict:
    campaign, n1 = collect_campaign()
    probes, n2 = collect_probes()
    systems, n3 = collect_systems()
    banked, n4 = collect_banked()
    record = {
        "schema": "lrwkv_readings_v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "roots": {"G": str(G), "outputs": str(OUTPUTS)},
        "counts": {"campaign_multichoice": len(campaign), "campaign_probe": len(probes),
                   "systems_probe": len(systems), "banked_readings": len(banked)},
        "readings": campaign + probes + systems + banked,
        "inventories": collect_inventories(),
        "notes": n1 + n2 + n3 + n4,
    }
    sources = source_facts(record["readings"])   # strips the file-level ones first
    record["hazard_counts"] = _hazard_counts(record["readings"])
    record["sources"] = {
        "n_files": len(sources),
        "with_unpinned_provenance": sum(
            1 for s in sources.values() if any("unpinned" in f for f in s["facts"])),
        "with_probe_copy_marker": sum(
            1 for s in sources.values()
            if any("probe_copy" in f for f in s["facts"])),
        "files": {p: s for p, s in sources.items() if s["facts"]},
    }
    return record


def _hazard_counts(readings: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in readings:
        for h in r.get("hazards") or []:
            key = h.split(":")[0].split("=")[0]
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def check(record: dict) -> dict:
    """Re-hash every source and report any that changed since the record was written."""
    changed, missing, ok = [], [], 0
    for r in record["readings"]:
        p = Path(r["path"]) if r.get("path") else None
        if p is None:
            continue
        if not p.is_file():
            if r.get("source_class") == "campaign_probe":
                continue          # an unqualified probe legitimately has no file
            missing.append(str(p))
            continue
        if sha256_file(p) != r.get("sha256"):
            changed.append(str(p))
        else:
            ok += 1
    return {"verified": ok, "changed": changed, "missing": missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "evidence"
                    / "descriptive_readings.json")
    ap.add_argument("--check", action="store_true",
                    help="re-hash every recorded source against the file on disk")
    ap.add_argument("--summary", action="store_true", help="counts only")
    args = ap.parse_args(argv)

    if args.check:
        if not args.out.is_file():
            raise SystemExit(f"{args.out} does not exist; run without --check first")
        report = check(json.loads(args.out.read_text(encoding="utf-8")))
        print(json.dumps(report, indent=2))
        return 0 if not report["changed"] and not report["missing"] else 1

    record = collect()
    if args.summary:
        print(json.dumps({k: record[k] for k in ("counts", "hazard_counts")}, indent=2))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "counts": record["counts"],
                      "hazard_counts": record["hazard_counts"],
                      "notes": len(record["notes"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
