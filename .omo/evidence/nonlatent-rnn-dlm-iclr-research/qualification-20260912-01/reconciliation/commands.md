# Read-only command/evidence record

All activity was bounded to the exact qualification and exact retained controller session. No command below invoked `qz`, the controller, the payload, CUDA, network access, package installation, git, or an authentication/config reader.

1. Read the exact reserved shared-output path. Result: `File not found`.
2. Read the exact controller-receipt path named in `submission/controller-state.json`. Result: `File not found`.
3. Read the exact local submission directory and the requested controller/request/receipt files. Its 19 listed entries contain no CreateJob stdout, stderr, or raw-response artifact; `admitted-job.json` is absent.
4. Targeted searches inside this qualification found no `*.log`, stdout, or stderr artifact. Existing `submission-preflight/commands/commands.md` is pre-submission fake-runner evidence and explicitly says qz was unavailable; it is not a capture of the real attempt.
5. One exact-path metadata observation at `2026-09-12T08:43:42Z` used `stat` on only the reserved output, exact controller receipt, request-attempt record, and sanitized execution receipt. The first two were absent; local record metadata is preserved in `filesystem.md` and `filesystem.json`.
6. Read-only OpenCode session lookup was restricted to session `ses_f6b8329ffffeLy5ceZLHSJzRHN`, exact tool part `prt_094bd6ec6001DgOMXZ1yPbuf7i`. The complete, non-truncated outer-controller capture records:
   - start `2026-09-12T08:30:43.398Z`, end `2026-09-12T08:30:43.893Z`;
   - outer controller exit `2`;
   - sole captured output `createjob_response_missing\n`;
   - exact invocation of the reviewed `controller.py submit ... ROOT_PASS_CONFIRMED` path.
   The child qz stdout/stderr is not present in this command capture because `controller_runtime.run_qz` captured it internally and the error path did not emit it.
7. Read-only SHA-256 calculation for `controller.py`, `controller_contract.py`, and `controller_runtime.py` matched their frozen entries in `submission/artifact-hashes.sha256`; no source was changed.

The original CreateJob stdout/stderr bytes are therefore unavailable in the permitted retained evidence and are classified as lost, not reconstructed.
