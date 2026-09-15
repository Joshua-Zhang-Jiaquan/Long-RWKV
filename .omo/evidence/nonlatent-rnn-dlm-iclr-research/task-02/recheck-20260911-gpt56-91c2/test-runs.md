# CPU test reruns

- Full current `scale/tests/nonlatent_iclr` collection: **98/98 passed** in 16.07 seconds.
- Task-1 file set plus both Task-2 test files: **79/79 passed** in 10.81 seconds.
- The supplied `71`/`95` counts are stale relative to the current on-disk collection; both current supersets pass.
- Both runs set `PYTHONDONTWRITEBYTECODE=1`, disabled pytest's cache plugin, and used no GPU or network.
