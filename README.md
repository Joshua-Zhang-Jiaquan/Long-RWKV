# Long-RWKV

Code, contracts, and receipts for the paper **"Depth-Recycled Recurrent Diffusion for
Long-Context Agent Workloads"** — an independent study of whether a *non-latent*,
bidirectionally-scanned recurrent diffusion model gains from depth recycling at long
context, and whether any such gain survives equal-compute controls.

The repository is the code half of that paper. It is **self-contained**: a clone runs
its own test suite, verifies its own evidence chain, and needs nothing from the cluster
that produced it.

---

## What is and is not in here, and why

This is a **research repository under active execution**, not a release. Its most
important property is that it does not overstate what has been measured.

The paper's confirmatory claims — a depth-recycling quality gain, a long-context macro
improvement, a serving goodput gain — are **registered and not yet measured**. The
program is blocked at the point where those measurements would be taken, and the
blockers are recorded rather than worked around. What *is* established, with evidence
in `DAN/nonlatent_iclr/` and `.omo/evidence/`, is:

| Task | State | What it establishes |
| --- | --- | --- |
| 1 | complete | Historical-evidence audit; **some previously reported decode conclusions weakened** |
| 2 | complete | Corrected metric reanalysis; valid NLL observations retained separately |
| 3 | complete | Architecture contract — 102 tests, `whole_architecture_ready: false` preserved |
| 4 | complete | RULER / LongBench / 200 rights-cleared repository tasks; 3,600-unit token matrix |
| 6 | complete | Bounded throughput calibration (8/8 H100 ranks) and the allocation ledger |
| 5, 7–18, F1–F4 | **not run** | Preregistration, training, long-context, serving, theory, release |

No unavailable evidence is replaced with a fixture or an assumed result anywhere in this
repository. Where a cell is missing, it is missing and labelled.

---

## Layout

```
scale/experiments/nonlatent_iclr/   the experiment harness: contracts, tasks,
                                    qualifications, registries, CLI
scale/tests/nonlatent_iclr/         its test suite (756 tests)
scale/data/                         canonical record + split primitives it needs
scale/eval/capability/sandbox.py    process-level evaluator sandbox
DAN/v7_arch_round/                  the model and trainer (BiRWKV-7 diffusion),
                                    architecture spec, run spec, configs
DAN/nonlatent_iclr/                 canonical artifacts the paper cites:
                                    architecture_contract.json, task_registry.json,
                                    allocation_ledger.json, metric_corrections.json,
                                    task4_assets/ (the rights-cleared task panel)
.omo/evidence/…                     per-attempt receipts, hash-bound
paper/                              the manuscript
```

`parents[3]` from `scale/experiments/nonlatent_iclr/` resolves to this repo's root, and
several modules read `DAN/…` and `.omo/evidence/…` relative to it — **the layout above is
load-bearing, not cosmetic.**

---

## Running it

**Python 3.12 or newer is required.** The package uses PEP 695 `type` statements
(`type HexDigest = Annotated[...]`), which are a `SyntaxError` on 3.11. Note that the
sibling `scratch_1b` program in the parent tree pins 3.11.9 — a different stack for a
different experiment line. Getting this wrong produces a confusing collection error
rather than a version complaint, which is why it is called out here.

```bash
python3 -m pip install -r requirements.txt

# Portable run: no cluster artifacts needed.
# Expect 629 passing and 127 skipping with a stated reason.
PYTHONPATH=. python3 -m pytest scale/tests/nonlatent_iclr -q

# Complete run: point the three roots at prepared artifacts and 744 pass.
export NONLATENT_MODEL_DIR=/path/to/models/RWKV7-Goose-World3-2.9B-HF
export NONLATENT_WORKER_ROOT=/path/to/nonlatent_iclr_qualification
export NONLATENT_EXTERNAL_ROOT=/path/to/global_user
PYTHONPATH=. python3 -m pytest scale/tests/nonlatent_iclr -q
```

The three `NONLATENT_*` variables are the only host-specific configuration. The external
artifacts they point at — model weights, cluster worker outputs — are large, are not
redistributable from here, and are deliberately **not** committed. When they are absent
the affected tests **skip with a reason** rather than fail, so the suite never reports a
pass it did not earn.

---

## What a clone can and cannot verify

**Can:** the architecture contract and its rejections, the task registry and its
replay-derived fields, the tokenizer and length qualifications, the metric-correction
chain, the allocation ledger's arithmetic and its planted-violation refusals, the CLI's
happy and failure paths, and the source-drift checks that bind evidence to the exact
sources that produced it.

**Cannot, without the external roots:** the tests that replay against real model weights
or against recorded cluster worker output. Those are exactly the 115 skips.

## A note on modification

The evidence in this repository is **hash-bound**: receipts record the digests of the
sources and artifacts they were produced from, and the test suite re-checks those
bindings. Editing hashed source would break the provenance the receipts exist to prove,
so the sources are byte-identical to the ones that produced the evidence. Where a test
depended on a host-specific absolute path, the *test* was made portable (env override +
skip) rather than the code being rewritten — the fix belongs on the side that has no
provenance claim. `SELF_CONTAINMENT.md` records precisely what was changed.

## Citing

See `paper/` and `CITATION.cff`. Nothing here is a released result; cite the paper, not
the intermediate receipts.
