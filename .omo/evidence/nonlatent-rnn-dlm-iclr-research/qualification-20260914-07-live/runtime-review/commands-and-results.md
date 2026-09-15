# Commands and results

Run from repository root. Verification performs read-only evidence parsing/hashing and qz GetJob; it never loads weights or executes a model.

```sh
qz train GetJob --data '{"job_id":"job-5b99b0c6-bb92-44e2-924b-761de4ad276b"}' -o json
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -c 'import runpy; runpy.run_path(".omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260914-07-live/runtime-review/verify_lifecycle_v7.py", run_name="__main__")'
```

Observed scheduler status: job_succeeded. Successful exact verifier: exit 0; complete assertions and per-rank results in verdict.json. Validator source comes from the pinned immutable release. Raw scheduler response is preserved. input-sha256.txt binds every consumed input; evidence-sha256.txt binds review files.
