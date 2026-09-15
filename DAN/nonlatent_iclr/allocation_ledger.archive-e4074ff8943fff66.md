# Task-6 allocation ledger

Calibration manifest: `147e6d7f007c09f261523069b2e34d2b6732bd724d500bbdc2ee133ca96dc311` (status PARTIAL).

| arm | source | basis | step s | tokens/s | GPU-h per seed @4B | GPU-h for 3 seeds |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | derived (A1_small_noloop) | canvas 4096 | 5.117 | 800.4 | 1388.17 | 4164.52 |
| A1 | measured (A1_small_noloop) | canvas 4096 | 5.117 | 800.4 | 1388.17 | 4164.52 |
| A2 | measured (A2_small_noloop) | canvas 4096 | 4.810 | 851.6 | 1304.78 | 3914.34 |
| A3 | measured (A3_small_loop) | unforecastable | - | - | - | - |
| A4 | derived (A3_small_loop) | unforecastable | - | - | - | - |
| A5 | measured (A5_small_loop_fwd) | unforecastable | - | - | - | - |

## Unforecastable

- **A3**: arm A3_small_loop produced no measurement in this job
- **A4**: arm A3_small_loop produced no measurement in this job
- **A5**: arm A5_small_loop_fwd produced no measurement in this job

## Storage

SAVE_ROOT is on the global_user volume, which was 98% full with ~240 GiB free when this ledger was written, and qz workers do not mount the project volume. Fix the retention policy before launching: a mid-run prune would destroy the failed-attempt evidence Task 7 must preserve, and it has happened before in this program.


## 2.9B confirmation

The 2.9B confirmation runs are priced separately and are NOT covered by this ledger: the measured large arms took only the 512-token rung, and both OOM'd at the 4096-token training canvas on a single 80 GiB GPU. Task 8 needs a sharded calibration, not an extrapolation.


## Assumptions

- per-step cost is the sum of a forward, a backward and an optimizer step measured at the training canvas
- a run's GPU-hours are (token budget / measured tokens per second) / 3600
- no multi-GPU scaling factor is applied; the ledger is per-GPU
- no communication, checkpointing pause, evaluation, or restart cost is included

## Not claimed

- sustainable goodput
- long-context capability
- quality
- physical edge-device performance
- measured training time at production sharding
