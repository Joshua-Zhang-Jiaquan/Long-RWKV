#!/usr/bin/env python3
"""W-A3/W-A5 endpoint battery for m4-loop-2p9b at s4750.

Builds and submits (after ListJobs duplicate checks):
  1. cap_lm 5-corpus panel on the s4750 endpoint, loop reps = trained (1)
  2. cap_lm panels with LOOP_REPS_OVERRIDE in {2,3,4}  (W-A5 depth extrapolation)
  3. decode-gate sampler harness on: loop-s4750 (reps 1,2,3,4), N2-s4000, N2-s6000
     (both N2 brackets pre-registered since 4.98B tokens sits between them)

The MMLU probe at s4750 is owned by the host prober daemon (sent_00004750).

Conventions copied verbatim from the m4 tracker's caplm instantiation and the
peer's decode-gate submissions (decode-gate-{mhc,n2}-s9500).
"""
import json
import os
import re
import subprocess
import sys

Q = "/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/scale/qz"
G = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
ENDPOINT = f"{G}/m2_baseline_triangle/m4loop_endpoint_ckpt"
SR = f"{G}/outputs_birwkv_diffusion"


def qz(*args):
    return subprocess.run(["qz", *args], capture_output=True, text=True)


def already_submitted(name):
    r = qz("train", "ListJobs", "--data", json.dumps({"keyword": name, "page_size": 5}), "-o", "yaml")
    return name in (r.stdout + r.stderr)


def submit(spec, dry=False):
    raw = json.dumps(spec)
    bad = [t for t in ("__CKPT__", "__OUTDIR__", "__NAME__", "__PLACEHOLDER__") if t in raw]
    assert not bad, bad
    assert "UNSET" not in spec["name"], spec["name"]
    assert "UNSET" not in spec["command"], "unsubstituted UNSET in command"
    if dry:
        print(f"DRY {spec['name']}: placeholders clean")
        return None
    if already_submitted(spec["name"]):
        print(f"SKIP {spec['name']}: already on the scheduler")
        return None
    r = qz("train", "CreateJob", "--data", raw, "-o", "yaml")
    m = re.search(r"job_id: (job-[a-f0-9-]+)", r.stdout + r.stderr)
    print(f"SUBMIT {spec['name']}: {m.group(1) if m else (r.stdout + r.stderr)[:300]}")
    return m.group(1) if m else None


def caplm_spec(ckpt, outdir, name, reps=None):
    spec = json.load(open(f"{Q}/m4_caplm_TEMPLATE.json"))
    cmd = spec["command"]
    assert "__CKPT__" in cmd and "__OUTDIR__" in cmd
    cmd = cmd.replace("__CKPT__", ckpt).replace("__OUTDIR__", outdir)
    if reps is not None:
        # eval-time depth extrapolation; the loader wraps model.forward
        cmd = cmd.replace(
            "export PROBE_ONLY=0;",
            f"export PROBE_ONLY=0; export LOOP_REPS_OVERRIDE={reps};")
        assert f"LOOP_REPS_OVERRIDE={reps}" in cmd
    spec["command"] = cmd
    spec["name"] = name
    spec["tb_summary_path"] = f"{G}/m2_baseline_triangle/caplm_tb_m4/{name}"
    tag = f" reps={reps} (W-A5 depth extrapolation)" if reps is not None else ""
    spec["description"] = ("v7 loop-arm L3 fixed-yardstick cap_lm panel on the s4750 "
                           f"endpoint{tag}. Paired vs N2 brackets (s4000/s6000).")
    return spec


def decode_spec(ckpt_abs, outdir, name, reps=None):
    spec = json.load(open(f"{Q}/birwkv_sampler_highmask_job.json"))
    outdir_abs = outdir if outdir.startswith("/") else f"{G}/{outdir}"
    if reps is not None:
        single = (f'echo "===SAMPLER ENDPOINT reps={reps}==="; '
                  f'export LOOP_REPS_OVERRIDE={reps}; '
                  f'CKPT_DIR={ckpt_abs} OUTDIR={outdir_abs} '
                  f'bash $DAN_SCALE_DIR/qz/launch_sampler_eval.sh')
    else:
        single = (f'echo "===SAMPLER ENDPOINT==="; '
                  f'CKPT_DIR={ckpt_abs} OUTDIR={outdir_abs} '
                  f'bash $DAN_SCALE_DIR/qz/launch_sampler_eval.sh')
    m = re.search(r"for ARM in [^;]+; do .*?done", spec["command"])
    assert m, "arm loop not found in base spec"
    spec["command"] = spec["command"].replace(m.group(0), single)
    spec["command"] = spec["command"].replace("===SAMPLER_GATE0_DONE===",
                                              f"===SAMPLER_GATE_{name}_DONE===")
    assert "codecpt" not in spec["command"] and "codegen4b" not in spec["command"]
    spec["name"] = name
    spec["description"] = ("v7 loop-arm CO-PRIMARY DECODE GATE: paired sampler "
                           f"harness, ckpt={ckpt_abs}"
                           + (f", loop reps={reps} (W-A5)" if reps is not None else "")
                           + ". em@k across mask ratios 0.7/0.95/0.99/1.0 + repetition.")
    return spec


def main():
    dry = "--dry-run" in sys.argv
    assert os.path.isfile(f"{ENDPOINT}/model.pt"), f"{ENDPOINT}/model.pt missing"
    assert os.path.isfile(f"{ENDPOINT}/meta.json"), f"{ENDPOINT}/meta.json missing"

    jobs = []
    # cap_lm panels (r=1 endpoint + reps 2,3,4)
    jobs.append(("caplm-r1", caplm_spec(ENDPOINT, f"{G}/cap_lm_m4loop_s4750",
                                        "m4-loop-lmeval-s4750")))
    for r in (2, 3, 4):
        jobs.append((f"caplm-r{r}",
                     caplm_spec(ENDPOINT, f"{G}/cap_lm_m4loop_s4750_reps{r}",
                                f"m4-loop-lmeval-s4750-reps{r}", reps=r)))
    # decode gate — loop endpoint at all four depths + both N2 brackets
    for r in (1, 2, 3, 4):
        reps = None if r == 1 else r
        outdir = "sampler_gate_m4_loop_s4750" if r == 1 else f"sampler_gate_m4_loop_s4750_reps{r}"
        jobs.append((f"decode-r{r}",
                     decode_spec(ENDPOINT, outdir, f"decode-gate-loop-s4750-r{r}", reps=reps)))
    for tag, step in (("s4000", "step_00004000_probe_copy"),
                      ("s6000", "step_00006000_probe_copy")):
        ck = f"{SR}/n2-knowpt-2p9b/{step}"
        assert os.path.isfile(f"{ck}/model.pt"), f"N2 bracket ckpt missing: {ck}"
        jobs.append((f"decode-n2-{tag}",
                     decode_spec(ck, f"sampler_gate_m4_n2_{tag}", f"decode-gate-n2-{tag}")))

    for _tag, spec in jobs:
        submit(spec, dry=dry)
    print("battery done:", len(jobs), "jobs")


if __name__ == "__main__":
    main()
