# Task 04 bounded final partial recheck

- **Verdict: CONFIRMED engineering partial only.** All four named defects are repaired; full Task 4 remains BLOCKED on known external assets and is not complete.
- HMM state-A gold `11997661/40547627` and state-B complement `28549966/40547627` sum to 1; state B validates with its complement and rejects stale state-A gold.
- Same-length nonbinary observation mutation `11111111 -> xxxxxxxx` is rejected.
- A same-length second contradictory evidence block is rejected; exactly one evidence block is enforced.
- Named receipt `prepare-b815181ffa404314bce2075d08f7d8e2` is current and binds exact/provenance/assets/registry source hashes; isolated `exact_tasks.py` drift invalidates it, and a real legacy unbound receipt is ineligible.
- Source identity matches the requested `exact_tasks.py` SHA-256 `cd29f3ab5cc155357c37686ad43651083e9860631adea1868eaa0a33eca50ef9`; saved registry SHA-256 is `75c487cb77ecb58b698b6fc9c032543528fd67dc9cb395784ce6a2f099d5531d`.
- Owned tests: 19/19 passed in 5.50 s, exit 0.
- Default-assets temporary CLI: happy verify BLOCKED with five gates and 689/31 representative counts, exit 2; failure probe exercised four production rejections, exit 0.
- External gates remain RULER, LongBench, 200 rights-cleared repository tasks, real isolated evaluator, and RWKV tokenizer/token qualification; no qualified token benchmark is claimed.
- Cleanup: temporary CLI/source-copy trees removed; only this evidence directory was written, with no product/plan/peer writes or network/install/qz/GPU/container/git use.
