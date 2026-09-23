# Full-model development inference qualification

This check precedes confirmation and uses no confirmation conditions. Its
checkpoint is the completed complementary-history development continuation at
400 additional updates (total step1900), selected to exercise the current
public-label serialization, not by confirmation performance.

Run explicit `TRITON_F32_DEFAULT=ieee`, FP32 tensors, PyTorch TF32 disabled,
and the existing deterministic-algorithm request. Eight independent GPU worker
processes in one job each evaluate the same12canvases: development seed51917,
indices0and1, dimensions2/4/8, fully masked and information-set-visible rounds.
Each production evaluation call is repeated three times. Compare its output
with a direct backbone call using explicit positions and gathers, bypassing
the SerialDenoiser/PackedDenoiser wrappers.

Thresholds are fixed before reading run output: maximum absolute binary
log-probability difference at most1e-5 for within-process repeats and wrapper
agreement, and at most1e-4 across the eight processes. All shards, finite
normalized outputs, identical checkpoint hashes, and the exact canvas panel
are required. Preserve failed checks; changing arithmetic or tolerance requires
a new documented development qualification, not silently accepting a failure.

This measures a fixed development checkpoint and canvas panel. It does not
establish equivalence to official RWKV, correctness of training gradients,
stability for all inputs, or reproducibility across independent scheduler
launches. The three repeated calls use the same process; the cross-process
comparison uses separate processes and GPUs within one launch. Qualification
does not establish dependency learning or replace endpoint evaluation.

Implementation: `lrwkv_evidence/train04dev/numerical_qualification.py`.
Submission: `qz/submit_train04dev_eval.py --numerical-qualification
--triton-precision ieee --training-plan <completed-plan> --submit`.
The submission receipt records immutable source hashes and the checkpoint.
Qualification-only receipts are excluded from endpoint-audit watcher matching.

## Executed result

The first eight-process job completed all shards and passed. Maximum absolute
log-probability differences were0for repeats,0for direct-backbone versus
serial-wrapper evaluation, and5.722045894884786e-6across processes. Results are
in results/train04paired/numerical_qualification_ieee.json; the immutable
submission receipt is results/train04dev/plan_eval_544c4bccd94fb340_rao_blackwell_s1900_9402a43a_ieee_numerical.json.
The checkpoint SHA256 is9402a43a3cdcf1867845ee968e47bfcc275404982263f24437852102c38c046a.
This passes the declared limited check; the scope limitations above still apply.
