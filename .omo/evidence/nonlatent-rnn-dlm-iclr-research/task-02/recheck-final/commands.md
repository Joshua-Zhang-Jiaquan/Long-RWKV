# Final bounded recheck commands

| Command | Result |
|---|---|
| `mkdir .omo/evidence/nonlatent-rnn-dlm-iclr-research/task-02/recheck-final` | exit 0 |
| `sha256sum <focused sources/tests/report/receipt>` | exit 0; `baseline-hashes.txt` |
| Scoped five Task-1 files plus `test_metrics.py` and `test_prompt_masks.py` | exit 0; 71 passed in 1.36s |
| From `scale/`, default CLI verify happy/failure and analyze for Tasks 1 and 2 | six exits 0; `cli-outcomes.json` |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python <recheck-final>/final_probe.py` | exit 0; 8/8 strict/recovery/no-overwrite checks pass |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python <recheck-final>/failure_cli_trace.py` | exit 0; NaN, missing-pair, prompt calls observed; finite polarity call count 0 |
| One execution of prior independent `raw_recompute.py` with output redirected in-memory to `recheck-final` | exit 0; 5x20x5, 2,000/2,000 numeric fields exact, receipt/hash checks pass |
| LSP diagnostics over four focused sources, three tests, and two final probes | zero diagnostics |
| `sha256sum -c <recheck-final>/baseline-hashes.txt` | all ten entries `OK` |
