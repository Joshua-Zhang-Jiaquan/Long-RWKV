# Independent submission-preflight cleanup record

- The 18-test run used an EXIT-trapped `/tmp/opencode/submission-preflight-tests.XXXXXX` directory, cache-disabled pytest, and `PYTHONDONTWRITEBYTECODE=1`.
- Root-PASS compatibility probes used EXIT-trapped `/tmp/opencode/submission-auth-pass.XXXXXX` directories. The first reviewer command failed only because of a wrong module export reference; its trap ran. The corrected probe passed and its trap also ran.
- The supplemental sequence used an EXIT-trapped `/tmp/opencode/submission-sequence-probe.XXXXXX` directory, fake runner, and fake receipt publisher. It made no GPFS write.
- Final glob checks found none of those three temporary path families.
- `request-attempt.json`, `admitted-job.json`, and the exact GPFS controller receipt destination remained absent, including broken symlinks, after all checks.
- No qz command was executed: no submit, dry-run, ListJobs, schema/read, login, or StopJob. No network, CUDA, git, install, or credential action occurred.
- The authorization, controller state, exact request, dry-run receipt, controller source, tests, payload source, and product state were read only and not modified.
- Only this requested `submission-preflight/` evidence tree was written. Authorization remains unconsumed with zero submitted jobs and zero GPU-hours.
