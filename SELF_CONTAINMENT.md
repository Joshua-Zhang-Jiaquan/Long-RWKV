# Self-containment record

What was changed to make this repository standalone, what was deliberately left
alone, and how each was verified. Recorded because the evidence here is hash-bound:
an unrecorded edit is indistinguishable from tampering.

## Verified, both directions

| Run | Command | Result |
| --- | --- | --- |
| Portable (clean clone, no cluster artifacts) | `PYTHONPATH=. python3 -m pytest scale/tests/nonlatent_iclr -q` | **506 passed, 115 skipped** |
| Complete (external roots exported) | same, with `NONLATENT_MODEL_DIR` / `NONLATENT_WORKER_ROOT` / `NONLATENT_EXTERNAL_ROOT` set | **621 passed** |

The skip count is not a hidden failure: each skip names the absent artifact and the
variable that would supply it.

## Changed: test portability only

Seven test modules hard-coded absolute cluster paths (`/inspire/hdd/global_user/...`)
for a model directory and for recorded worker-output directories. They passed only on
the machine that produced them. Each now resolves its root through an environment
variable with a repository-relative fallback, and skips with a stated reason when the
artifact is absent.

One test, `test_calibration_arms.py::test_small_arms_use_a_worker_reachable_root`,
asserted that a path *started with* `/inspire/hdd/global_user/`. That encoded a host
path rather than the property it meant — that the root is worker-reachable and outside
the project volume. It now asserts the property.

**No test assertion was weakened.** The portability changes replace a host path with a
configurable root; the properties under test are unchanged.

## Deliberately not changed: any hashed source

An earlier attempt made the host-path constants in `service.py`,
`qualification/runtime_config.py`, `qualification/calibration_arms.py`,
`architecture_lifecycle.py` and `architecture_semantics.py` env-overridable with
repository-relative defaults. That was **reverted**, because it broke ten source-drift
tests — correctly. Those hashes exist to prove the evidence was produced from exactly
those sources, and re-deriving them would make the receipts assert a provenance they do
not have.

The rule this repository follows: **fix portability on the side that carries no
provenance claim.** Tests may be edited; hash-bound sources may not. The five files above
are byte-identical to the versions that produced the evidence, and the drift checks pass.

## Not included, on purpose

Training weights and checkpoints. They are large, they are not redistributable from
here, and the paper does not depend on shipping them — the receipts record their
digests. `external/` and the weight extensions are gitignored.

## Recorded rather than repaired

Three limitations carried from the Task 4 qualification are reproduced in the paper as
limitations and are *not* fixed here, because fixing them would change what was
measured: the two-task repository confirmation split, the process-level (not externally
managed) evaluator sandbox, and the token matrix having run on a CPU host rather than in
the qz CPU lane.

## A packaging gap, corrected (and my first reading of it was wrong)

Three source files the architecture contract names as helper paths were absent, and this
repository initially reproduced that absence:

| Path the contract expects | Initial state | Now |
| --- | --- | --- |
| `DAN/v7_arch_round/code/models/latent_plan.py` | absent | vendored |
| `DAN/v7_arch_round/code/models/state_hijacking_cache.py` | absent | vendored |
| `DAN/v7_arch_round/code/models/state_hijacking_dit_torch_types.py` | vendored | vendored |

`models/residual_streams.py:58` imports the third unguarded, so without them the module
is unimportable.

Two CPU check scripts depend on it --- `code/train/test_residual_streams.py` and
`code/train/test_backbone_loop.py`. Despite the names they are **standalone scripts run
as `python <file>`, not pytest modules**: neither defines a single `def test_*`, and each
has a `__main__` block. Their import being broken was the real defect, and pytest
reported it only incidentally, as a collection error on files it tried to import. Both
now run and pass (`RESIDUAL-STREAMS SUITE: ALL PASS`, `BACKBONE-LOOP SUITE: ALL PASS`).
Calling them "tests" without that qualification was sloppy; they are checks with a
script entry point.

**I first recorded this as a defect in the model code. That was wrong**, and the
correction matters more than the fix. All three files exist in the live training tree at
`qz_stage_traj4096_v7/scale/models/`, where `residual_streams.py` imports cleanly and is
byte-identical to this repository's copy (`5ba645d5924b1782`). The gap was in how this
repository was assembled — an incomplete copy of an arch-round snapshot — not in the
model. It is fixed by vendoring, and the model import path now resolves.

A related divergence worth knowing when reading this repository: the in-repo
`birwkv7_diffusion.py` is a **curated subset** of the training-tree module
(`2591f58b…` here vs `e6d90ce8…` there), and drops `LatentRefinementMap`,
`LatentExitGate` and `run_refinement_ladder` entirely. That divergence is deliberate in
kind. The three helpers were the accidental part.

## A pin that looks stale and must stay that way

`contract.py` binds `outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/meta.json`,
which no longer exists, while `qualification/runtime_config.py` resolves its checkpoint
to the surviving `m2_baseline_triangle/m4loop_endpoint_ckpt`. That asymmetry looks like a
bug and is not one.

`test_external_hashed_source_drift_is_stale` *writes* that external file and asserts the
audit reports `STALE`. The pin is a hashed **audit binding whose content is the fact that
the artifact is gone** — the record of a loss. Repointing it to the surviving copy was
tried and broke ten tests, correctly: it would make a removed source look present and
silently erase the effect of the 3.25\,TB deletion from the audit trail. It was reverted.
`runtime_config` resolving and `contract.py` binding are different jobs; the asymmetry is
the design.

## Known-good state after these changes

`PYTHONPATH=. python3 -m pytest scale/tests/nonlatent_iclr -q` → **621 passed**.
