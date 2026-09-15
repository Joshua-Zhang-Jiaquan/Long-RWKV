# Task 2 final bounded recheck

- **Verdict: REJECT on one exact acceptance gap; do not mark Task 2 yet.**
- Strict record-count repair passes: a valid 8×32 fixture yields 256 pairs, while per-shard `n_records=999` and `n_records=true` both reject.
- Markdown-write interruption leaves only valid JSON; an unpatched normal `prepare` retry publishes Markdown+receipt and verifies `METRICS_COMPLETE`.
- Receipt-write interruption leaves valid JSON+Markdown; normal retry publishes the receipt and verifies `METRICS_COMPLETE`.
- Malformed pre-existing JSON returns `TAMPERED`, remains byte-identical (`8c69fc…9776` before/after), and creates no Markdown or receipt.
- Real `CLI verify --task 2 --case failure` exits 0 and claims polarity, nonfinite, missing-pair, and prompt-leakage guards were rejected.
- Runtime call tracing proves that command executes one prompt probe and two paired probes—NaN and mismatched keys—but **zero finite matched-pair polarity probes**.
- Typed polarity itself is correct (`+0.8` max-run delta → `worse`) and `test_max_run_positive_delta_is_worse` passes inside the 71-test scope; the happy verifier also binds report interpretations.
- Nevertheless, the plan explicitly assigns reversed-polarity failure QA to `CLI verify --task 2 --case failure`; returning a success detail for an unexecuted control is a false-positive verifier result, so strict plan acceptance fails.
- Exact remaining repair: in `failure_probe_task_two`, execute a finite matched max-run pair with candidate > control and return `PROBE_FAILED` unless the result is positive and interpreted `worse`; add a CLI regression that fails if that call is removed.
- Scoped Task-1+2 suite passes: **71/71 in 1.36s**. All six quick default Task-1/2 verify/analyze commands exit 0; Task 1 still reports eight blocked scientific claims.
- One independent raw pass remains exact: 5 comparisons, 100 cells, 500 metric rows, 128,000 paired values, and 2,000/2,000 numeric fields equal with max error `0.0`.
- Current JSON SHA is `1c4568cc9ebbf3a656c803e34bc7d4e510ccaa2d9a133dd9602a3e19e31b9f08`; Markdown SHA is `a1d87e99eb1146ea8fd5824024d93a65c565994f63c7e6cdc2ae297b7bc98efd`; receipt attempt `7afe15dedf5944609effdb44659d2fb8` and analysis hash `8f57e4c5…26b4` bind exactly.
- Focused sources/tests/probes have zero LSP diagnostics; baseline hashes are unchanged, fixtures cleaned, and no product/plan/Boulder/Git/GPU/network/install/qz action occurred.
