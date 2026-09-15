# Cleanup receipt

- Both adversarial CPU probes used `tempfile.TemporaryDirectory` under `/tmp/opencode`; command output records `cleanup=TemporaryDirectory.cleanup_called`.
- The CLI validate commands only parsed paths and did not create them. The broken `python -m ...probe` executed no `main` body and likewise created no output path.
- The failed first failure-producer inspection was a Python parse-time quoting error, before temporary-directory creation.
- Persistent writes are limited to this `preflight-review/` evidence tree. No qualification/product source was edited.
- No CUDA/model execution, model unpickling, qz CreateJob/dry-run/StopJob, qz config access, download, install, training, optimizer step, or git command occurred.
