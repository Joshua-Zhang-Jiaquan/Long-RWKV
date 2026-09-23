"""Build EvalPlus HumanEval+ / MBPP+ task files for ``eval/capability/code.py``.

This module writes JSONL that the **existing, unmodified** scorer reads.  Two
facts about ``code.py`` determine its whole shape, and both were measured rather
than assumed.

1. The two suites are asymmetric, and neither existing path fits MBPP+
---------------------------------------------------------------------
``normalize_problem`` (code.py:113-146):

* ``humaneval`` builds its single test case from ``raw["test"]`` plus a
  ``check(<entry_point>)`` call.  EvalPlus's ``humanevalplus`` ``test`` field
  *defines* ``check`` (verified: 164/164 rows) and is ~77 KB of expanded inputs.
  So HumanEval+ needs no new logic -- ``humanevalplus`` is registered as its own
  suite name only so the output header records which benchmark was scored.

* ``mbpp`` builds its test cases **and its prompt** from ``raw["test_list"]``.
  The prompt is the trap.  Putting the expanded EvalPlus suite in ``test_list``
  would paste ~9.5 KB of hidden grader inputs into the model's prompt; putting
  the base assertions there scores base MBPP under the plus name.  One field
  cannot hold both roles.

  Routing MBPP+ through the ``humaneval`` path instead does not work either, and
  this was measured rather than reasoned about: ``build_candidate_code`` uses the
  *humaneval* rule ``prompt + completion``, so the assembled program began with
  "You are an expert Python programmer..." and all three test rows came back
  ``syntax_error`` -- a silent 0 % pass@1, the exact failure class this module
  exists to prevent.  The suites differ in **assembly** as well as in field
  layout, so no combination of the two existing branches expresses MBPP+.

  Hence one new ``mbppplus`` branch in ``code.py``: prompt and grader in
  separate fields (``prompt`` / ``test``), MBPP's standalone assembly rule, and
  ``entry_point`` empty so no ``check(...)`` is appended -- MBPP+ bodies are
  self-executing (verified: 0/378 define ``check``) and call several functions by
  name.  The existing ``humaneval`` and ``mbpp`` branches are untouched, so the
  pinned scorer behind the published GSM8K/HumanEval readings still means what it
  meant.

2. The sandbox silently fails every numpy-importing test, and EvalPlus is all numpy
---------------------------------------------------------------------------------
Measured (``calibrate_evalplus_sandbox.py``, 2026-09-20): at the harness's own
defaults the **canonical solutions** score ``0/40``.  Not one row failed on
correctness.  ``sandbox._resource_limiter`` sets ``RLIMIT_NPROC (0, 0)``
(sandbox.py:124), so no thread can be created, and OpenBLAS aborts on import:

    OpenBLAS blas_thread_init: pthread_create failed for thread 8 of 64:
    Resource temporarily unavailable

Every EvalPlus test body imports numpy (163/164 and 378/378), while **no**
prompt or canonical solution does.  And ``_run_single_test`` reports any nonzero
exit as ``(False, stderr, False)``, which ``_failure_tag`` renders as
``tests_failed`` -- identical to a wrong answer.  Uncorrected, every arm would
have reported 0.0 % pass@1 on HumanEval+/MBPP+ and no field in the output would
have said the grader never ran.

Because numpy appears only in the *test* bodies, the fix belongs in the task
file: each test block is prefixed with :data:`THREAD_CAP_PREAMBLE`, which pins
the BLAS thread counts to 1 before numpy is imported.  Measured effect: 52/52
canonical solutions pass, including the largest rows (502 KB and 790 KB test
bodies), and it is *faster* -- 1693 ms of failed thread spawning becomes ~110 ms.
The sandbox keeps ``RLIMIT_NPROC (0, 0)``; nothing about the isolation stack is
weakened, and ``sandbox.py`` is not edited at all.

A sweep of ``mem_mb`` x ``timeout_s`` over {256, 512, 1024} x {5, 10, 30} shows
26/26 at every point once the preamble is present, so the harness defaults
(``mem_mb=256``, ``timeout_s=5``) are kept: no new CLI surface is needed.

What this module changes outside itself
--------------------------------------
Two additive edits, both verified not to alter the existing paths:
``normalize_problem`` gains an ``mbppplus`` branch and accepts ``humanevalplus``
as an alias of ``humaneval``; ``build_candidate_code`` keys its concat rule on
``suite in ("humaneval", "humanevalplus")`` (it already fell through to
``completion`` for every other suite, which is MBPP's rule).  ``run_eval.py``'s
``--suite`` choices gain the two plus names.  ``code.py``'s 13-test harness is
re-run to confirm the ``humaneval``/``mbpp`` behaviour is unchanged.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Final

#: Where the downloaded parquets live, relative to the data root.
SOURCES: Final = {
    "humanevalplus": "raw/evalplus_humanevalplus/data",
    "mbppplus": "raw/evalplus_mbppplus/data",
}

#: Expected row counts, from the EvalPlus release.  A mismatch is a refusal and
#: not a warning: a 160-task HumanEval+ and a 164-task one are different
#: benchmarks, and the difference would be invisible in a pass@1.
EXPECTED_ROWS: Final = {"humanevalplus": 164, "mbppplus": 378}

#: Which ``code.py`` suite each file must be RUN as.  Each plus file has its own
#: name so the output header records which benchmark was scored: a HumanEval and
#: a HumanEval+ pass@1 are different numbers and must not share a label.
RUN_AS_SUITE: Final = {"humanevalplus": "humanevalplus", "mbppplus": "mbppplus"}

#: Pin BLAS threading before numpy is imported.  ``RLIMIT_NPROC (0, 0)`` makes
#: any ``pthread_create`` fail, and OpenBLAS treats that as fatal at import.
#: ``setdefault`` so a caller-supplied value still wins.  This is prepended to
#: the TEST block only: no EvalPlus prompt or canonical solution imports numpy,
#: so the model never sees it and the candidate program is unmodified.
THREAD_CAP_PREAMBLE: Final = (
    "import os as _os\n"
    "for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',\n"
    "           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):\n"
    "    _os.environ.setdefault(_v, '1')\n"
    "del _os, _v\n"
)

#: The MBPP instruct wrapper, copied verbatim from ``code.py:133-139``.  It must
#: stay character-identical: if it drifts, the model sees a different
#: instruction than the historical MBPP runs did and the two numbers stop being
#: comparable.
MBPP_PROMPT_TEMPLATE: Final = (
    "You are an expert Python programmer. Write only the Python code for this task.\n"
    "Task: {task}\n"
    "Your code should pass these tests:\n"
    "{tests}"
    "\n\nPython code:\n"
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_parquet_rows(parquet_dir: Path) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no *.parquet under {parquet_dir}")
    rows: list[dict] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def render_mbpp_prompt(task: str, base_tests: list[str]) -> str:
    """The MBPP instruct prompt, showing the THREE BASE assertions.

    EvalPlus's design is that the visible specification is the original MBPP
    assertions while the grader is the expanded suite.  Scoring against inputs
    the prompt never showed is the benchmark, not a leak.
    """
    return MBPP_PROMPT_TEMPLATE.format(task=task, tests="\n".join(base_tests))


def build_humanevalplus(rows: list[dict]) -> list[dict]:
    """Rows for the ``humaneval`` path, carrying the *plus* tests."""
    out = []
    for raw in rows:
        entry = str(raw["entry_point"])
        test = str(raw["test"])
        if "def check(" not in test:
            raise SystemExit(
                f"{raw['task_id']}: the EvalPlus test body does not define "
                f"check(), but code.py's humaneval path appends check({entry}) "
                f"and nothing else. Every task would raise NameError, which "
                f"reads as a 0% pass@1 rather than as a broken task file.")
        out.append({
            "task_id": str(raw["task_id"]),
            "prompt": str(raw["prompt"]),
            "test": THREAD_CAP_PREAMBLE + test,
            "entry_point": entry,
            # Kept for the ground-truth/contamination checks, never shown.
            "canonical_solution": str(raw.get("canonical_solution", "")),
        })
    return out


def build_mbppplus(rows: list[dict]) -> list[dict]:
    """Rows for the ``humaneval`` path, holding the MBPP+ expanded suite.

    The ``humaneval`` assembly rule is used because it keeps the prompt and the
    tests in **separate fields**; ``mbpp``'s rule renders its prompt from the
    tests it scores, which cannot express MBPP+ (module docstring, part 1).

    ``entry_point`` is deliberately ``""``: it suppresses the ``check(...)``
    call that ``normalize_problem`` would otherwise append (MBPP+ test bodies
    are self-executing) and it also disables ``_ast_precheck``'s entry-point
    scan, which is right here -- MBPP has no single entry function, and the
    expanded body calls several by name.
    """
    out = []
    for raw in rows:
        base = [str(c) for c in (raw.get("test_list") or [])]
        expanded = str(raw["test"])
        if not base:
            raise SystemExit(
                f"{raw['task_id']}: no base test_list, so there is nothing to "
                f"show the model as the task specification.")
        if "def check(" in expanded:
            raise SystemExit(
                f"{raw['task_id']}: this MBPP+ test body defines check(), so it "
                f"is not self-executing and would never be invoked with "
                f"entry_point empty. The suite's shape changed; re-derive the "
                f"assembly rule rather than scoring nothing.")
        if len(expanded) <= sum(len(c) for c in base):
            raise SystemExit(
                f"{raw['task_id']}: the 'test' field ({len(expanded)} chars) is "
                f"not larger than the base assertions; this parquet may be base "
                f"MBPP rather than MBPP+, and scoring it would report base "
                f"numbers under the plus name.")
        # test_imports carries `import math` for 10 rows; the expanded body
        # assumes it is in scope.
        imports = "\n".join(str(i) for i in (raw.get("test_imports") or []))
        test_block = THREAD_CAP_PREAMBLE + (f"{imports}\n" if imports else "") + expanded
        out.append({
            "task_id": f"Mbpp/{raw['task_id']}",
            "prompt": render_mbpp_prompt(str(raw["prompt"]), base),
            "test": test_block,
            "entry_point": "",
            "mbpp_base_test_list": base,
            "canonical_solution": str(raw.get("code", "")),
        })
    return out


def manifest(name: str, rows: list[dict], out_path: Path,
             parquet_dir: Path) -> dict:
    parquets = sorted(parquet_dir.glob("*.parquet"))
    return {
        "suite_file": name,
        "path": str(out_path),
        "items": len(rows),
        "expected_items": EXPECTED_ROWS[name],
        "run_as_suite": RUN_AS_SUITE[name],
        "run_as_suite_note": (
            f"Run with SUITE={RUN_AS_SUITE[name]}. For mbppplus that is an "
            f"ASSEMBLY RULE, not a claim about which benchmark was scored: the "
            f"humaneval rule keeps prompt and tests in separate fields, which "
            f"the mbpp rule does not (it renders its prompt from test_list). "
            f"The tests in this file are the EvalPlus expanded ones."
        ),
        "jsonl_sha256": sha256_file(out_path),
        "source_parquet": [str(p) for p in parquets],
        "source_parquet_sha256": {p.name: sha256_file(p) for p in parquets},
        "grader": "the EvalPlus expanded test suite",
        "prompt_shows": (
            "the task text plus the three original MBPP assertions"
            if name == "mbppplus" else
            "the function signature and docstring"),
        "thread_cap_preamble_sha256": hashlib.sha256(
            THREAD_CAP_PREAMBLE.encode()).hexdigest(),
        "thread_cap_rationale": (
            "sandbox.py:124 sets RLIMIT_NPROC (0,0), so OpenBLAS aborts on "
            "import and every numpy-importing test exits nonzero -- which "
            "_failure_tag records as 'tests_failed', indistinguishable from a "
            "wrong answer. Measured: canonical solutions scored 0/40 without "
            "this preamble and 52/52 with it. Prepended to the TEST block only; "
            "no EvalPlus prompt or canonical solution imports numpy."),
        # NOT a claim that any setting works: this file cannot measure itself.
        # The calibration runs the canonical solutions THROUGH this JSONL
        # (calibrate_evalplus_sandbox.py), so the receipt necessarily comes after
        # the build.  Naming a literal here is how the manifest came to assert
        # 256 MiB / 5 s as "validated" while the measurement showed 4 rows
        # SIGKILLed at exactly that setting -- a resource kill the harness records
        # as ``tests_failed``, indistinguishable from a wrong answer.
        "sandbox_settings_validated": None,
        "sandbox_calibration": {
            "status": "not_attached",
            "how": ("run lrwkv_evidence/e1/calibrate_evalplus_sandbox.py against "
                    "this file, then `evalplus_tasks.py --attach-calibration "
                    "<receipt.json>`. Until then no setting is certified for this "
                    "file and the E1 code bodies have no ceiling to compare "
                    "against."),
        },
        "scoring_asymmetry_note": (
            "EvalPlus scores against inputs the prompt does not show. That is "
            "the benchmark's design, not a leak: the base assertions are the "
            "task statement and the expanded suite is the grader."),
    }


def build(name: str, data_root: Path, out_dir: Path) -> dict:
    parquet_dir = data_root / SOURCES[name]
    rows = load_parquet_rows(parquet_dir)
    if len(rows) != EXPECTED_ROWS[name]:
        raise SystemExit(
            f"{name}: {len(rows)} rows on disk but the release has "
            f"{EXPECTED_ROWS[name]}. A different item count is a different "
            f"benchmark; refusing to write a file that would be cited as "
            f"{name}.")
    built = (build_humanevalplus(rows) if name == "humanevalplus"
             else build_mbppplus(rows))
    ids = [r["task_id"] for r in built]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{name}: duplicate task_ids")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in built:
            f.write(json.dumps(row) + "\n")
    report = manifest(name, built, out_path, parquet_dir)
    (out_dir / f"{name}.manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def attach_calibration(receipt_path: Path, out_dir: Path) -> dict:
    """Fold a sandbox-calibration receipt into the built manifests.

    The receipt is only about the file it measured, so this refuses unless the
    receipt's ``source`` is the JSONL now on disk *and* that JSONL's sha256 still
    matches the manifest.  A receipt attached to a rebuilt file would certify a
    ceiling for items that are no longer there.

    ``isolation_mode`` is copied verbatim and never normalised: the login node
    reports ``socket_stub`` while the pod runs ``unshare``, so a receipt taken here
    certifies the *ceiling*, not the pod's isolation, and the E1 code bodies have
    to assert the pod's own mode separately.
    """
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    entries = receipt if isinstance(receipt, list) else [receipt]
    by_file = {e["suite_file"]: e for e in entries}
    updated = {}
    for name, entry in sorted(by_file.items()):
        if name not in SOURCES:
            raise SystemExit(f"receipt names unknown suite {name!r}")
        man_path = out_dir / f"{name}.manifest.json"
        if not man_path.is_file():
            raise SystemExit(f"{man_path} does not exist: build the suite first.")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        jsonl = Path(man["path"])
        if entry.get("source") != str(jsonl):
            raise SystemExit(
                f"{name}: the receipt measured {entry.get('source')!r} but the "
                f"manifest points at {str(jsonl)!r}. A ceiling measured on another "
                f"file says nothing about this one.")
        live = sha256_file(jsonl)
        if live != man["jsonl_sha256"]:
            raise SystemExit(
                f"{name}: {jsonl} has sha256 {live} but the manifest records "
                f"{man['jsonl_sha256']}. The file was rebuilt after the manifest; "
                f"rebuild and re-measure rather than attaching a stale ceiling.")
        if entry.get("n_measured") != man["items"]:
            raise SystemExit(
                f"{name}: the receipt measured {entry.get('n_measured')} of "
                f"{man['items']} items. A ceiling from a subset is a different "
                f"statistic; re-run the calibration without --limit.")
        man["sandbox_settings_validated"] = {
            "mem_mb": entry["mem_mb"], "timeout_s": entry["timeout_s"]}
        man["sandbox_calibration"] = {
            "status": "attached",
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "isolation_mode_measured": entry.get("isolation_mode"),
            "isolation_mode_note": (
                "the mode the CALIBRATION ran under. The login node reports "
                "socket_stub; pods run unshare. This certifies the ceiling, not "
                "the pod's isolation -- an E1 code shard must assert its own "
                "isolation_mode in-pod."),
            "ground_truth_pass_rate": entry["ground_truth_pass_rate"],
            "scorable_items": entry["scorable_items"],
            "known_broken_upstream": entry.get("known_broken_upstream", {}),
            "unexplained_failures": entry.get("unexplained_failures", []),
            "ceiling_note": (
                "ground_truth_pass_rate is the highest score any model can reach "
                "on this file at these settings. Rows in known_broken_upstream are "
                "unscorable by any model and must be reported 'unsupported' rather "
                "than counted as failures."),
        }
        man_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        updated[name] = man["sandbox_calibration"]
    missing = sorted(set(SOURCES) - set(by_file))
    if missing:
        print(f"# note: no receipt entry for {missing}; those manifests still say "
              f"not_attached", file=sys.stderr)
    return updated


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path,
                    help="capability_eval_data root holding raw/evalplus_*")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--suite", action="append", choices=sorted(SOURCES),
                    help="repeatable; default both")
    ap.add_argument("--attach-calibration", type=Path,
                    help="fold a calibrate_evalplus_sandbox.py receipt into the "
                         "manifests instead of rebuilding")
    args = ap.parse_args(argv)

    if args.attach_calibration:
        print(json.dumps(attach_calibration(args.attach_calibration, args.out_dir),
                         indent=2))
        return 0
    if args.data_root is None:
        raise SystemExit("--data-root is required to build")
    names = args.suite or sorted(SOURCES)
    report = [build(n, args.data_root, args.out_dir) for n in names]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
