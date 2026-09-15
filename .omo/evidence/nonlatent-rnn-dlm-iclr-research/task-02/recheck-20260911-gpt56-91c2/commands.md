# Recheck commands

| Command | Exit/result |
|---|---|
| `mkdir <recheck>` | 0 |
| `sha256sum <owned sources/tests/report/receipt>` | 0; `baseline-hashes.txt` |
| `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider scale/tests/nonlatent_iclr` | 0; 98 passed in 16.07s |
| Same runner over six Task-1 files plus both Task-2 files | 0; 79 passed in 10.81s |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python <recheck>/numeric_probe.py` | 0; 7/7 controls pass |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python <recheck>/record_probe.py` | 0; 15/16 contract checks pass |
| `PYTHONDONTWRITEBYTECODE=1 python <recheck>/raw_recompute.py` | 0; 2,000/2,000 numeric fields exact |
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python <recheck>/publication_probe.py` | 0; tamper controls pass, 2/3 interruption recovery checks fail |
| From `scale/`, CLI `verify --task {1,2} --case {happy,failure}` and `analyze --task {1,2}` | all six exit 0; `cli-controls.json` |
| LSP diagnostics over four owned sources, two tests, and four probes | zero diagnostics |
| `sha256sum -c baseline-hashes.txt` | all nine entries `OK` |
