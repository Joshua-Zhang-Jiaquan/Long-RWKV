#!/usr/bin/env python3
"""v7 loop-arm gate analysis at the s4750 endpoint (all paired, ci95).

Usage: python3 wl_gate_analysis.py [endpoint_step=4750]

Reads:
  PRIMARY    cap_lm_m4loop_s4750{,_reps{2,3,4}} vs cap_lm_n2_s{4000,6000}
  CO-PRIMARY sampler_gate_m4_loop_s4750{,_reps{2,3,4}} vs sampler_gate_m4_n2_{s4000,s6000}
  W-A5       reps curve within the loop endpoint (paired vs r1)
  MMLU       m2_baseline_triangle/ability_curve_m4loop.csv
"""
import glob
import json
import math
import os
import sys

G = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
STEP = sys.argv[1] if len(sys.argv) > 1 else "4750"


def load_caplm(base):
    """corpus -> {doc_id: nll} for arm='causal'."""
    out = {}
    for cd in sorted(glob.glob(f"{base}/*")):
        if not os.path.isdir(cd):
            continue
        corpus = os.path.basename(cd)
        docs = {}
        for f in glob.glob(f"{cd}/lm.*.shard*.json"):
            d = json.load(open(f))
            for r in d.get("records", []):
                if r.get("arm") == "causal" and r.get("failure") is None:
                    docs[r["document_id"]] = r["metrics"]["nll"]
        out[corpus] = docs
    return out


def load_sampler(base):
    recs = []
    for f in sorted(glob.glob(f"{base}/sampler.shard*of*.json")):
        recs += json.load(open(f)).get("records", [])
    return recs


def paired_mean_ci(deltas):
    n = len(deltas)
    if n < 2:
        return n, float("nan"), float("nan")
    m = sum(deltas) / n
    se = math.sqrt(sum((x - m) ** 2 for x in deltas) / (n - 1) / n)
    return n, m, 1.96 * se


def verdict(m, ci):
    if ci == float("nan"):
        return "NO DATA"
    if m - ci > 0:
        return "loop WORSE (CI)"
    if m + ci < 0:
        return "loop BETTER (CI)"
    return "wash"


def caplm_compare(tag, a, b):
    print(f"\n== PPL: {tag} (per-doc paired, causal arm, nats)")
    better = worse = 0
    for corpus in sorted(set(a) & set(b)):
        ids = set(a[corpus]) & set(b[corpus])
        n, m, ci = paired_mean_ci([a[corpus][i] - b[corpus][i] for i in ids])
        v = verdict(m, ci)
        if "BETTER" in v:
            better += 1
        if "WORSE" in v:
            worse += 1
        print(f"  {corpus:<12} n={n:<6} delta={m:+.4f} ci95=±{ci:.4f}  {v}")
    print(f"  -> better={better} worse={worse} (PRIMARY needs >=2 better, no delta > +0.03)")
    return better, worse


def sampler_compare(tag, recs_a, recs_b):
    print(f"\n== DECODE: {tag} (paired by document_id+seed)")
    arms = sorted({r["arm"] for r in recs_a} & {r["arm"] for r in recs_b})
    for arm in arms:
        for key in ("em", "max_run_frac", "distinct_frac"):
            da = {(r["document_id"], r["seed"]): r["metrics"][key]
                  for r in recs_a if r["arm"] == arm}
            db = {(r["document_id"], r["seed"]): r["metrics"][key]
                  for r in recs_b if r["arm"] == arm}
            common = set(da) & set(db)
            n, m, ci = paired_mean_ci([da[k] - db[k] for k in common])
            if key == "em" or abs(m) > 0.002:
                print(f"  {arm:9s} {key:<13} n={n:<5} delta={m:+.4f} ci95=±{ci:.4f}  "
                      f"{'BETTER' if m - ci > 0 else ('WORSE' if m + ci < 0 else '~0')}")


def main():
    ep = f"{G}/cap_lm_m4loop_s{STEP}"
    # PRIMARY vs both N2 brackets
    for ref in ("s4000", "s6000"):
        if os.path.isdir(f"{G}/cap_lm_n2_{ref}"):
            caplm_compare(f"loop-s{STEP} - N2-{ref}",
                          load_caplm(ep), load_caplm(f"{G}/cap_lm_n2_{ref}"))
        else:
            print(f"[skip] cap_lm_n2_{ref} not merged yet")
    # W-A5 depth extrapolation (paired vs r1 within the endpoint)
    r1 = load_caplm(ep)
    for r in (2, 3, 4):
        p = f"{G}/cap_lm_m4loop_s{STEP}_reps{r}"
        if os.path.isdir(p):
            caplm_compare(f"loop-s{STEP} reps{r} - reps1 (W-A5)", load_caplm(p), r1)
        else:
            print(f"[skip] reps{r} panel not merged yet")
    # CO-PRIMARY decode vs both brackets
    epd = f"{G}/sampler_gate_m4_loop_s{STEP}"
    if glob.glob(f"{epd}/sampler.shard*"):
        recs_ep = load_sampler(epd)
        for ref in ("s4000", "s6000"):
            p = f"{G}/sampler_gate_m4_n2_{ref}"
            if glob.glob(f"{p}/sampler.shard*"):
                sampler_compare(f"loop-s{STEP} - N2-{ref}", recs_ep, load_sampler(p))
            else:
                print(f"[skip] sampler n2_{ref} not done")
    else:
        print("[skip] loop endpoint sampler not done")
    # W-A5 decode reps curve
    for r in (2, 3, 4):
        p = f"{G}/sampler_gate_m4_loop_s{STEP}_reps{r}"
        if glob.glob(f"{p}/sampler.shard*") and glob.glob(f"{epd}/sampler.shard*"):
            sampler_compare(f"loop-s{STEP} reps{r} - reps1 (W-A5 decode)",
                            load_sampler(p), load_sampler(epd))
    # MMLU curve
    print("\n== MMLU curve (tracked, non-gating)")
    for line in open(f"{G}/m2_baseline_triangle/ability_curve_m4loop.csv").read().splitlines():
        f = line.split(",")
        print(f"  step {f[0]:>5}  mmlu {f[3]}")
    print("\nN2 MMLU refs: s2000=46.5 s4000=47.5 s6000=48.0")


if __name__ == "__main__":
    main()
