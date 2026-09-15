# Controller-response observations and hypotheses

## Observed controller behavior

- `controller_contract.qz_arguments` constructs `qz train CreateJob --data <exact-text> -o json`; its real branch omits `--dry-run` (`submission/controller_contract.py:149-155`).
- `record_and_call_once` writes the attempt before calling that branch, rejects a nonzero child return code as `createjob_failed`, and sends only child stdout to `extract_job_id` (`submission/controller_runtime.py:41-57`).
- `run_qz` captures both child streams, but no failure artifact persists either stream (`submission/controller_runtime.py:60-76`).
- `extract_job_id` requires nonempty syntactically valid JSON, then searches raw bytes for exactly one exact `"job_id"` string field anywhere in the document. It does not validate a success envelope, status, business code, or field location (`submission/controller_contract.py:158-175`). The ID must match the lowercase `job-<UUID>` pattern (`submission/controller_models.py:31-37`).
- The retained outer command capture is complete and contains only `createjob_response_missing`; the raw child qz response is not retained. The frozen source hashes match the files inspected.

Consequences of the parser, stated as code possibilities rather than claims about the lost response:

- Return code zero plus empty stdout becomes `createjob_response_missing`; stderr is ignored by ID extraction.
- Return code zero plus any valid JSON without the exact `job_id` spelling also becomes `createjob_response_missing`, including a semantic business-error envelope or a success envelope using a different ID field/location convention. Business error/code/message/status fields are not inspected.
- Any valid JSON containing exactly one correctly formatted exact `job_id` field anywhere would be accepted even if sibling or enclosing fields expressed a business error. This permissive acceptance path was not observed here because the actual result was `createjob_response_missing`.
- A nonempty malformed/truncated JSON response, an invalid ID, multiple exact ID fields, or a nonzero child exit would produce a different controller error. Those forms are inconsistent with the observed error under the frozen code.

## Three hypotheses

1. **Admitted but response unrecognized — POSSIBLE, NOT CONFIRMED.** Distinguishing evidence would be an actual scheduler ID bound to the reserved name/nonce, a raw success response using an unrecognized shape, or payload output carrying controller-bound identity. None exists in this filesystem/controller capture. Exact shared output and controller receipt were absent at `2026-09-12T08:43:42Z`; that weakens payload-start evidence but does not prove non-admission or non-queueing.
2. **API business error with child exit zero — POSSIBLE, NOT CONFIRMED.** The observed child return code was zero and the parser would collapse a valid no-ID business-error envelope, or stderr-only error with empty stdout, to the observed missing-response code. No raw stdout, stderr, error code, or message was retained, so code behavior alone cannot establish that a business error occurred.
3. **Empty or semantically incomplete client reply — POSSIBLE, NOT CONFIRMED.** Exact empty stdout or syntactically valid JSON lacking exact `job_id` matches the observed controller error. A malformed nonempty partial reply does not. The retained outer command log cannot distinguish empty stdout from valid no-ID JSON because the child bytes were captured and discarded on this error path.

The three hypotheses remain observationally ambiguous within this evidence scope. No actual scheduler job ID can be reported, no controller receipt may be published post hoc, and no runtime result is classified as passed.
