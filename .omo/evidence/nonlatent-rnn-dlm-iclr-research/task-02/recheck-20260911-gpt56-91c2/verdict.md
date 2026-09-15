# Task 2 repaired-state independent recheck

- **Verdict: REJECT. Do not mark Task 2 in the root plan.** The repaired report is truthful, but three executable acceptance defects remain.
- Current CPU collections pass: 98/98 full tests and 79/79 Task-1-file-plus-Task-2 tests. The supplied 95/71 counts are stale relative to the current on-disk collection.
- Prediction-derived max run is `0.75`, hand tau-a with one tie is `2/3`, zero-mask accuracy is `null`, and strict active-token partition/overlap controls all pass.
- This qualifies only the public CPU mask helper. No generated sequence, prompted-quality measurement, GPU run, or qz job was executed; sequence exact match remains correctly unavailable from historical records.
- Bad bool seed/metric/grid, failed rows, malformed schema, missing panel/grid, incomplete shards/grid, duplicate/missing pairs, and nonfinite values all fail closed.
- **Remaining defect 1:** `load_sampler_panel` accepts each shard declaring `n_records=999` while containing 32 rows; the complete 8-shard/256-pair fixture is accepted. Declared record counts are not validated.
- The repaired JSON is exact against six raw panels: 5 comparisons, 100 cells, 500 metric rows, 128,000 paired raw values, and 2,000/2,000 numeric fields bit-equal with maximum absolute error `0.0`.
- All six panels have 8 shards, 5,120 rows, 20 arms, 256 records/arm, zero failed rows, matching ordered pair keys, and report-matching panel hashes.
- JSON SHA-256 is `46ebd28d422a6a0f750cef483d170a07abc9fb8de804add3b820c3d7a69cdcfe`; Markdown SHA-256 is `a1d87e99eb1146ea8fd5824024d93a65c565994f63c7e6cdc2ae297b7bc98efd`; receipt and analysis-code hash bind exactly.
- Derived-delta, pair-count, empty-comparison, schema-version, analysis-hash, missing/tampered receipt, and missing/tampered Markdown controls all fail closed.
- **Remaining defect 2:** after interruption during Markdown publication, retry returns `PUBLICATION_EXISTS` and verify returns `TAMPERED`; after receipt interruption, retry returns `PUBLICATION_EXISTS` and verify returns `MALFORMED`. Only interruption before the first JSON write recovers.
- Default Task 1 and Task 2 happy/failure/analyze CLI commands all exit 0 with structured statuses; Task 1 retains eight blocked scientific claims.
- **Remaining defect 3:** Task-2 CLI failure QA executes only future-answer leakage. The required reversed-polarity, NaN, and missing-pair cases pass as independent controls but are not exercised by `CLI verify --task 2 --case failure`.
- Four owned sources, two tests, and four evidence probes have zero LSP diagnostics; all 98 current tests pass.
- Final hash check shows no source, test, report, Markdown, or receipt drift; temporary fixtures were removed and no product/plan/Git/network/GPU/qz/install action occurred.
