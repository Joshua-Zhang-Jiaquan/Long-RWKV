# Qualification06 runtime-review validation commands

No scheduler command, GPU allocation, model import, checkpoint load, or source mutation was performed by this review.

The read-only JSON/AST validator used:

```text
PYTHONDONTWRITEBYTECODE=1
PYTHONPYCACHEPREFIX=/tmp/qualification06-runtime-review.qtqutl
PYTHONSAFEPATH=1
/usr/bin/python -P
```

It parsed existing JSON with the standard library, compared the established Torch spellings with `packaging.version.Version`, parsed manifested `model_checks.py` with `ast`, and read source bytes only for SHA-256 comparison. It imported no staged runtime module, Torch, FLA, model, or checkpoint. The external cache had zero entries and was removed.

Assertions covered exact job/request/run/nonce/manifest bindings; unique ranks 0–7; aggregate equality with rank files; checkpoint and parameter identities; finite `[1,48,65536]` logits; source-manifest hashes; `FullCanvasAdapter.open`, `forward`, and `reset` in `finally`; one controller-receipt occurrence of the run and nonce; scheduler/scientific status separation; unknown image digest; and preserved cache limitations.

Ledger verification:

```bash
sha256sum -c monitoring-v1/worker-output-hashes.sha256
sha256sum -c live-execution-hashes.sha256
```

Results: 14/14 worker-output entries and 25/25 live-execution entries passed.
