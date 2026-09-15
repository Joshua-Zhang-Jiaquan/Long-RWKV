# Metric corrections (2026-09-11)

## Corrected

- positive max_run_frac loop-minus-control is worse
- em is masked-token accuracy; whole-sequence exact match is separate
- tau-a is a commit-order diagnostic, not reference agreement
- r100 whole-window masking is unconditional, not prompt-conditioned

## Recomputed max_run_frac (loop minus control)

| Loop | Control | Pairs | Delta | Meaning |
|---|---|---:|---:|---|
| sampler_gate_m4_loop_s4750 | sampler_gate_m4_n2_s4000 | 2 | 0.00000000 | worse |
| sampler_gate_m4_loop_s4750 | sampler_gate_m4_n2_s6000 | 2 | 0.00000000 | worse |
| sampler_gate_m4_loop_s4750_reps2 | sampler_gate_m4_loop_s4750 | 2 | -0.00000000 | better |
| sampler_gate_m4_loop_s4750_reps3 | sampler_gate_m4_loop_s4750 | 2 | -0.00000000 | better |
| sampler_gate_m4_loop_s4750_reps4 | sampler_gate_m4_loop_s4750 | 2 | 0.00000000 | equal |

## Unmeasured

- No prompted generation quality claim; mask-helper tests do not generate samples.
- Original r100 did not preserve a prompt.
- No fresh trials; legacy normal CIs are descriptive only.
