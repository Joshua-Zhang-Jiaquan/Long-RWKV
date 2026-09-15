# Self-containment record

What was changed to make this repository standalone, what was deliberately left
alone, and how each was verified. Recorded because the evidence here is hash-bound:
an unrecorded edit is indistinguishable from tampering.

## Verified, both directions

| Run | Command | Result |
| --- | --- | --- |
| Portable (clean clone, no cluster artifacts) | `PYTHONPATH=. python3 -m pytest scale/tests/nonlatent_iclr -q` | **485 passed, 115 skipped** |
| Complete (external roots exported) | same, with `NONLATENT_MODEL_DIR` / `NONLATENT_WORKER_ROOT` / `NONLATENT_EXTERNAL_ROOT` set | **600 passed** |

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
