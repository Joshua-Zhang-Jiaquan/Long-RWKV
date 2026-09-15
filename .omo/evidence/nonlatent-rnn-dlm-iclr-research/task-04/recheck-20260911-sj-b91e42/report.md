# Task 04 repaired CPU-registry recheck

- **Verdict: NEEDS FIX for the engineering partial; full Task 4 remains BLOCKED.** Most requested repairs are confirmed, but strict public HMM evidence and receipt source binding still have real gaps.
- Owned suite: 16/16 passed in 5.44 s, exit 0.
- Same-length public-evidence redaction and overwrite contradiction now fail validation; normal public evidence validates.
- **Remaining bug:** changing `HMM ...;query=state_A` to same-length `query=state_B` still validates, because `_analytic_hmm` ignores the query field.
- **Remaining bug:** replacing HMM binary observations `11111111` with `xxxxxxxx` still validates, because every non-`0` symbol is treated as `1`.
- **Remaining bug:** appending a second, contradictory `BEGIN/END EVIDENCE` block while preserving total length still validates; `public_evidence` accepts the first block without enforcing uniqueness.
- Seed 999, index 200, and lengths 0/1/16 reject. HMM load 1/128 yields 1/128 observations; dataflow yields 2/129 assignments; endpoint sources differ.
- Explicit placement works: a feasible 8192/50% task starts evidence at character 4096, while an unplaceable registered request raises `InfeasibleTaskError` rather than clamping.
- All 720 seed-101/index-0 cells were checked without corpus materialization: 689 valid, 31 explicitly infeasible, zero gold/length/position failures. Scan 0.295 s; full registry verification 1.105 s. The 144 length-8192 cells were 134/10 in 0.033 s.
- Structurally complete but arbitrary locally hash-bound manifests, including 200 fake records, cannot unlock readiness: five blockers, READY false, exit 2. No tokenizer/evaluator was executed.
- Real standalone temp-root results: prepare 2, happy verify 2, failure verify 0; status BLOCKED, blockers 5, counts 689/31. All four failure checks exercised production validators and returned true.
- Registry is current: saved and in-memory-regenerated SHA-256 both `75c487cb77ecb58b698b6fc9c032543528fd67dc9cb395784ce6a2f099d5531d`; previous bytes are preserved as archive hash `55fc4b55040d9136a142ca472305b477ce11e42a7f4e4b0e96f68ae024a97563`.
- Receipt status: three new receipts bind the current registry hash; four legacy receipts are unbound. **Remaining provenance gap:** receipts contain no implementation source hashes, so registry-hash equality alone cannot attest validator-source freshness.
- Known external gates remain exactly RULER, LongBench, 200 rights-cleared repository tasks, a real isolated evaluator, and RWKV tokenizer/token-length qualification. These character fixtures are not qualified RWKV-token benchmarks.
- Cleanup: all temporary trees removed; only this unique evidence directory was written. No product/plan/peer edits, full-corpus materialization, network, install, GPU/qz/container, git, or untrusted execution.
