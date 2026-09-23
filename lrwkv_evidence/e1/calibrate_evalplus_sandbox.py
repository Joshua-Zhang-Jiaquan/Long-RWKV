"""Measure whether the pinned sandbox can run the EvalPlus *expanded* suites at all.

Why this has to be measured before the task files are built
-----------------------------------------------------------
``sandbox.run_tests`` executes each test block in ``python -I`` with
``env={}`` and two hard rlimits (``sandbox.py:113-126``): ``RLIMIT_AS`` at
``mem_mb`` MiB and ``RLIMIT_CPU`` at ``ceil(timeout_s)`` seconds, with
``eval_code``'s defaults being ``mem_mb=256`` / ``timeout_s=5``.  Those defaults
were set for base MBPP, whose test block is a single ``assert``.

The EvalPlus expanded blocks are a different kind of program: ~77 KB
(HumanEval+) and ~9.5 KB (MBPP+) of inlined inputs, they ``import numpy``, and
they loop over hundreds of cases.  numpy alone reserves a large virtual mapping,
and ``RLIMIT_AS`` counts *address space*, not resident pages.

The failure mode this guards against is the dangerous one: a resource refusal is
scored as a **wrong answer**, not as an error.  ``_run_single_test`` returns
``(False, stderr, False)`` for any nonzero exit, so a MemoryError from the rlimit
and a genuinely incorrect program both land as ``passed=False`` and get the
``tests_failed`` tag.  A whole arm would read as 0.0 % pass@1 on HumanEval+ and
nothing in the output would say the grader never ran.

So this script runs the benchmark's own **canonical solutions** through the real
sandbox.  Ground truth must score ~1.0; whatever it actually scores is the
ceiling any model can reach, and the gap is pure harness artifact.  The measured
ceiling is written to a JSON receipt so the E1 bodies can be built against the
settings that were shown to work, and so a later reader can tell a model failure
from a sandbox failure.

What it found (2026-09-20)
-------------------------
At the harness defaults, **0 of 40** canonical solutions passed.  Not one failed
on correctness: ``RLIMIT_NPROC (0, 0)`` (sandbox.py:124) forbids thread
creation, so OpenBLAS aborts at ``import numpy``, and every EvalPlus test body
imports numpy (163/164 and 378/378) while no prompt or canonical solution does.

Two independent corrections followed, both verified here:

* the thread-cap preamble in ``evalplus_tasks.THREAD_CAP_PREAMBLE``, which fixes
  the import and is also ~15x faster (1693 ms of failed spawning -> ~110 ms);
* ``--code_mem_mb 4096 --code_timeout_s 30``, which clears the 4 rows that are
  genuinely resource-hungry (HumanEval/15, HumanEval/139, Mbpp/255, Mbpp/599 are
  SIGKILLed at 256 MiB/5 s).  The flags default to the old 256/5, so base
  humaneval/mbpp runs are unchanged.

One row cannot be fixed at any setting: **HumanEval/32**'s own EvalPlus test is
malformed -- ``assert _poly(*candidate(*inp), inp) <= 0.0001`` splats the float
that ``candidate`` returns, raising ``TypeError`` before any comparison.  That is
an upstream defect, independent of the model, so the honest ceiling for
HumanEval+ under this harness is 163/164 and that row must be reported
``unsupported`` rather than scored.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Final

G = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")
SCALE = G / "qz_stage_traj4096_v7" / "scale"
RAW = G / "capability_eval_data" / "raw"
TASKS_DIR = G / "capability_eval_data" / "tasks_evalplus"

#: Which ``code.py`` suite each built file is run as.  Mirrors
#: ``evalplus_tasks.RUN_AS_SUITE``; kept local so this script can run standalone.
RUN_AS_SUITE: Final = {"humanevalplus": "humanevalplus", "mbppplus": "mbppplus"}


#: The one row whose EvalPlus test is itself malformed: it splats a float
#: (``_poly(*candidate(*inp), inp)``), so it raises TypeError before comparing
#: anything. No sandbox setting fixes it and no model can pass it.
KNOWN_BROKEN_UPSTREAM: Final = {"HumanEval/32": "evalplus test splats a float return value"}


def load_rows(name: str) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(RAW / f"evalplus_{name}" / "data" / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet for {name} under {RAW}")
    rows: list[dict] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def load_built(name: str, tasks_dir: Path) -> list[dict]:
    """The JSONL the E1 jobs will actually read.

    Measuring the built file rather than the parquet is the point: the thread-cap
    preamble and the prompt/grader split live in the JSONL, so a receipt taken
    from the parquet would certify a file nobody runs.
    """
    path = tasks_dir / f"{name}.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist: run evalplus_tasks.py first. The ceiling "
            f"must be measured on the file the jobs will read.")
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def measure(name: str, rows: list[dict], mem_mb: int, timeout_s: float,
            limit: int | None) -> dict:
    """Score the canonical solutions through ``code.py``'s own scorer.

    Uses ``normalize_problem``/``build_candidate_code`` rather than re-deriving
    the assembly here: the routing itself is a thing that can be wrong (MBPP+
    through the humaneval rule produced 100 % ``syntax_error``), so a receipt
    built from a private copy of the rules would certify the wrong code path.
    """
    sys.path.insert(0, str(SCALE))
    from eval.capability.code import build_candidate_code, normalize_problem
    from eval.capability.sandbox import _active_isolation_mode, run_tests

    suite = RUN_AS_SUITE[name]
    subset = rows if limit is None else rows[:limit]
    results = []
    for row in subset:
        prob = normalize_problem(suite, row)
        program = build_candidate_code(suite, prob["prompt"],
                                       str(row.get("canonical_solution", "")))
        res = run_tests(program, prob["test_cases"], timeout_s=timeout_s,
                        mem_mb=mem_mb, entry_point=prob["entry_point"] or None)
        tag = ("passed" if res.passed else
               "timed_out" if res.timed_out else
               "syntax_error" if res.syntax_error else
               "entry_point_missing" if res.entry_point_missing else "failed")
        results.append({"task_id": str(row["task_id"]), "tag": tag,
                        "wall_ms": res.wall_ms,
                        "error": (res.error or "")[:400]})
    n = len(results)
    passed = sum(1 for r in results if r["tag"] == "passed")
    tags: dict[str, int] = {}
    for r in results:
        tags[r["tag"]] = tags.get(r["tag"], 0) + 1
    # Separate the failure kinds: a resource kill, an upstream-broken task and a
    # genuinely wrong answer are all `tests_failed` in the artifact, and the
    # message is the only place they differ.
    reasons: dict[str, int] = {}
    for r in results:
        if r["tag"] == "passed":
            continue
        err = r["error"]
        key = ("upstream_broken_task" if r["task_id"] in KNOWN_BROKEN_UPSTREAM else
               "openblas_thread_limit" if "blas_thread_init" in err else
               "sigkill_resource_limit" if "exit code -9" in err else
               "MemoryError" if "MemoryError" in err else
               "timeout" if r["tag"] == "timed_out" else
               err.strip().splitlines()[-1][:120] if err.strip() else "empty_error")
        reasons[key] = reasons.get(key, 0) + 1
    unexplained = [r["task_id"] for r in results
                   if r["tag"] != "passed" and r["task_id"] not in KNOWN_BROKEN_UPSTREAM]
    measured_ids = {r["task_id"] for r in results}
    broken_here = {k: v for k, v in KNOWN_BROKEN_UPSTREAM.items() if k in measured_ids}
    return {
        "suite_file": name,
        "run_as_suite": suite,
        "isolation_mode": _active_isolation_mode(),
        "mem_mb": mem_mb,
        "timeout_s": timeout_s,
        "n_measured": n,
        "ground_truth_pass_rate": (passed / n) if n else 0.0,
        "scorable_items": n - len(broken_here),
        "tag_counts": tags,
        "failure_reasons": reasons,
        "known_broken_upstream": broken_here,
        "unexplained_failures": unexplained,
        "slowest_ms": max((r["wall_ms"] for r in results), default=0),
        "median_ms": sorted(r["wall_ms"] for r in results)[n // 2] if n else 0,
        "total_wall_s": round(sum(r["wall_ms"] for r in results) / 1000, 1),
        "per_task": results,
        "interpretation": (
            "ground_truth_pass_rate is the CEILING for this suite at these "
            "sandbox settings. Anything in unexplained_failures is a harness "
            "defect that would be attributed to the model: those rows must not "
            "be scored as model failures. known_broken_upstream rows are "
            "unscorable by any model and are reported 'unsupported'."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", action="append",
                    choices=("humanevalplus", "mbppplus"),
                    help="repeatable; default both")
    ap.add_argument("--tasks-dir", type=Path, default=TASKS_DIR,
                    help="dir holding the built <suite>.jsonl files")
    ap.add_argument("--mem-mb", type=int, default=4096,
                    help="4096 is the calibrated value; 256 is eval_code's "
                         "default and scores 0/40 (see module docstring)")
    ap.add_argument("--timeout-s", type=float, default=30.0,
                    help="30 is the calibrated value; 5.0 is eval_code's default")
    ap.add_argument("--limit", type=int, default=None,
                    help="measure only the first N rows (quick sweeps)")
    ap.add_argument("--from-parquet", action="store_true",
                    help="measure the raw parquet instead of the built JSONL "
                         "(diagnostic only: it has no thread-cap preamble, so it "
                         "reproduces the 0%% baseline)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    names = args.suite or ["humanevalplus", "mbppplus"]
    report = []
    for name in names:
        rows = (load_rows(name) if args.from_parquet
                else load_built(name, args.tasks_dir))
        rep = measure(name, rows, args.mem_mb, args.timeout_s, args.limit)
        rep["source"] = "parquet" if args.from_parquet else str(
            args.tasks_dir / f"{name}.jsonl")
        report.append(rep)
        brief = {k: v for k, v in rep.items() if k != "per_task"}
        print(json.dumps(brief, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
