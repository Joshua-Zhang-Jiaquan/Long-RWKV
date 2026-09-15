# Verification commands

From repository root:

```sh
qz train GetJob --data '{"job_id":"job-cd3c6850-91cc-4a1e-8f10-a3ce6c5baec9"}' -o json
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -c 'import runpy; runpy.run_path(".omo/evidence/nonlatent-rnn-dlm-iclr-research/qualification-20260914-08-live/runtime-review/verify_semantics_v8.py", run_name="__main__")'
```

Scheduler: job_succeeded. Verification: exit0, all assertions passed. The script only queries scheduler state, reads/hashes evidence and nonweight source, reparses typed JSON, and checks scalar/name ledgers. No model forward/backward, checkpoint load, trainer execution or GPU initialization occurs. It also captures the current canonical-contract gap; reruns after reconciliation may intentionally fail that historical-state assertion.
