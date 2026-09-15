# Task-6 allocation ledger

Calibration manifest: `d22e139f4916dd06e0e19ea9e9d1f8d1c3318ce35e538c06f4520a76a61e4743` (status PASSED).

| arm | source | basis | step s | tokens/s | GPU-h per seed @4B | GPU-h for 3 seeds |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | derived (A1_small_noloop) | canvas 4096 | 5.126 | 799.1 | 1390.40 | 4171.20 |
| A1 | measured (A1_small_noloop) | canvas 4096 | 5.126 | 799.1 | 1390.40 | 4171.20 |
| A2 | measured (A2_small_noloop) | canvas 4096 | 4.620 | 886.7 | 1253.14 | 3759.43 |
| A3 | measured (A3_small_loop) | canvas 4096 | 4.274 | 958.4 | 1159.32 | 3477.95 |
| A4 | derived (A3_small_loop) | canvas 4096 | 4.274 | 958.4 | 1159.32 | 3477.95 |
| A5 | measured (A5_small_loop_fwd) | canvas 4096 | 5.006 | 818.2 | 1357.95 | 4073.85 |

## Unforecastable


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
