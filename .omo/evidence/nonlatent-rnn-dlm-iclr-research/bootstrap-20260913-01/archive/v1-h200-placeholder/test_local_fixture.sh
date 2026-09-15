#!/usr/bin/env bash
# test_local_fixture.sh — LOCAL FIXTURE TEST ONLY.
#
# Validates logging + failure-output behavior of bootstrap_diag.sh on CPU with
# isolated temp dirs and known-missing paths. This is NOT proof of worker/node
# behavior. No production GPFS paths are written. No network, no qz, no GPU.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d /tmp/bootstrap_diag_test.XXXXXX)"
trap 'rm -rf "$T"' EXIT

fail() { echo "TEST_FAIL: $*" >&2; exit 1; }

# ---------- PART 1: payload fixture run (overridden roots + missing paths) ----------
GOOD="$T/goodroot"; mkdir -p "$GOOD"
BADROOT="$T/not_a_dir"; : > "$BADROOT"      # a FILE: mkdir-under-file fails even for root
MISS="$T/definitely/missing/path"

RC=0
BOOT_RUN_ID=20260913T000000Z BOOT_NONCE=deadbeef \
BOOT_OUT_ROOTS="$GOOD $BADROOT/child" \
BOOT_PAYLOAD_PATH="$MISS" BOOT_STAGED_MODEL_PATH="$MISS" \
BOOT_META_PATH="$MISS" BOOT_RECEIPT_PATH="$MISS" \
bash "$HERE/bootstrap_diag.sh" > "$T/run.out" 2>&1 || RC=$?

[[ $RC -eq 0 ]] || fail "payload exited $RC (expected 0)"
grep -q "DIAG_SUMMARY run=20260913T000000Z nonce=deadbeef" "$T/run.out" || fail "summary line missing"
grep -q "PUBLISH_OK $GOOD/bootstrap-20260913T000000Z-deadbeef" "$T/run.out" || fail "writable root not published"
grep -q "PUBLISH_MKDIR_FAIL" "$T/run.out" || fail "unwritable root error not recorded"
grep -q "publish_ok=1/2" "$T/run.out" || fail "publish counter wrong"

DJSON="$GOOD/bootstrap-20260913T000000Z-deadbeef/diag.json"
[[ -s "$DJSON" ]] || fail "durable diag.json missing/empty"
python3 - "$DJSON" <<'PY' || fail "diag.json content assertions"
import json, os, sys
d = json.load(open(sys.argv[1]))
assert d["schema"].startswith("bootstrap-diag/v1"), d["schema"]
assert d["run_id"] == "20260913T000000Z" and d["nonce"] == "deadbeef"
assert d["request_hash"] and len(d["request_hash"]) == 16
p = d["paths"]["project_payload"]
assert p["exists"] is False, p
assert d["identity"]["uid"] is not None
# Presence must be recorded FAITHFULLY (not assumed absent), and NO env values
# may leak into the JSON — presence/match booleans only.
expected = ["NNODES", "NGPUS", "MICROBATCH", "GRAD_ACCUM", "STEPS", "TOKEN", "DAN_SCALE_DIR"]
assert set(d["qualification_env"]) == set(expected)
for k in expected:
    v = d["qualification_env"][k]
    assert v["present"] == (k in os.environ), k
    if v["present"] and k != "TOKEN":  # TOKEN/DAN_SCALE_DIR have no local expected value
        assert v["matches_expected"] in (True, False), k
raw = open(sys.argv[1]).read()
for k in expected:
    if k in os.environ and os.environ[k]:
        assert os.environ[k] not in raw, f"env VALUE leaked for {k}"
print("PART1_JSON_OK")
PY
[[ -s "$GOOD/bootstrap-20260913T000000Z-deadbeef/stdout.log" ]] || fail "durable stdout snapshot missing"
echo "PART1_PASS (local fixture; NOT a node proof) — artifacts under $T (removed on exit)"

# ---------- PART 2: spec round-trip QA (decoding must never execute payload) -------
SPEC="$HERE/job_spec_candidate.json"
[[ -s "$SPEC" ]] || { echo "PART2_SKIPPED (no job_spec_candidate.json)"; exit 0; }
python3 - "$SPEC" "$HERE/bootstrap_diag.sh" "$T/decoded_payload.sh" <<'PY' || fail "spec QA assertions"
import base64, json, sys
spec = json.load(open(sys.argv[1]))
payload = open(sys.argv[2]).read()
decoded_path = sys.argv[3]

cmd = spec["command"]
assert "\n" not in cmd, "command must be single-line"
assert cmd.startswith("echo ") and "base64 -d > /tmp/bootstrap_diag" in cmd
b64 = cmd.split(" ", 2)[1]
decoded = base64.b64decode(b64, validate=True).decode()
assert decoded == payload, "decoded inline payload differs from reviewed bootstrap_diag.sh"
open(decoded_path, "w").write(decoded)

fc = spec["framework_config"][0]
assert fc["instance_count"] == 1 and fc["shm_gi"] == 1800
assert fc["spec_id"] == "7166bd2e-6cbe-4bd9-be38-762d11003e7f"
assert fc["image"].endswith("relay2:v2")
assert spec["max_running_time_ms"] == "180000"
assert spec["task_priority"] == 4
assert spec["auto_fault_tolerance"] is False and spec["fault_tolerance_max_retry"] == 0
assert not any(k.startswith("reserve") for k in spec), "reserve fields must be omitted"
assert spec["project_id"].startswith("project-160ccb20") and spec["workspace_id"].startswith("ws-9dcc0e1f")
envs = {e["name"]: e["value"] for e in spec["envs"]}
assert set(envs) == {"BOOT_RUN_ID", "BOOT_NONCE", "BOOT_PAYLOAD_PATH", "BOOT_STAGED_MODEL_PATH", "BOOT_OUT_ROOTS"}
assert not any(k in envs for k in ("NNODES", "NGPUS", "MICROBATCH", "GRAD_ACCUM", "STEPS", "TOKEN", "DAN_SCALE_DIR")), \
    "diagnostic job must NOT inject qualification env (absence is the observation)"
assert "PENDING" in spec["logic_compute_group_id"], "placeholder pool id keeps candidate non-submit-able"
print("PART2_SPEC_JSON_OK")
PY
bash -n "$T/decoded_payload.sh" || fail "decoded payload has bash syntax errors"
cmp -s "$T/decoded_payload.sh" "$HERE/bootstrap_diag.sh" || fail "round-trip payload mismatch"
echo "PART2_PASS (round-trip: decode identical, syntax OK, payload NOT executed)"

echo "ALL_PASS"
