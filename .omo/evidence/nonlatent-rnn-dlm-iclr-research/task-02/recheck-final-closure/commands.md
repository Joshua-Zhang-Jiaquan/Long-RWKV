# Task 2 closure commands

| Command | Result |
|---|---|
| `mkdir .omo/evidence/nonlatent-rnn-dlm-iclr-research/task-02/recheck-final-closure` | exit 0 |
| `sha256sum <changed sources/test/current artifacts/prior proof>` | exit 0; `baseline-hashes.txt` |
| Scoped five Task-1 files plus Task-2 metric/mask files | exit 0; 72 passed in 1.35s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python <closure>/polarity_control.py` | exit 0; direct `0.8/worse`, normal success, reversed direction `PROBE_FAILED` |
| From `scale/`, default Task-2 CLI happy/failure/analyze | three exits 0; `cli-outcomes.json` |
| `PYTHONDONTWRITEBYTECODE=1 python <closure>/binding_check.py` | exit 0; 12/12 hash/receipt/5x20x5 reuse checks pass without raw reread |
| LSP diagnostics on changed source/test and two closure probes | zero diagnostics |
| `sha256sum -c <closure>/baseline-hashes.txt` | all ten entries `OK` |
