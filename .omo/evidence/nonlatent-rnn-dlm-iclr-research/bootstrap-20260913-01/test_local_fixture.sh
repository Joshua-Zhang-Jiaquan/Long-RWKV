#!/usr/bin/env bash
# test_local_fixture.sh v3 — LOCAL FIXTURE TEST ONLY.
# CPU validation of logging/failure behavior with isolated temp dirs and
# known-missing paths, plus frozen-spec round-trip QA including the explicit
# /usr/bin/bash -lc wrapper. NOT proof of worker/node behavior. No production
# GPFS writes, no network, no qz, no GPU. Frozen request is never regenerated.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d /tmp/bootstrap_diag_test.XXXXXX)"
trap 'rm -rf "$T"' EXIT

fail() { echo "TEST_FAIL: $*" >&2; exit 1; }
GOOD="$T/goodroot"; mkdir -p "$GOOD"
BADROOT="$T/not_a_dir"; : > "$BADROOT"          # file: mkdir-under fails even for root
MISS="$T/definitely/missing/path"

run_diag() { # $1=out-file, then env assignments as args
  local out="$1"; shift
  RC=0
  env "$@" BOOT_OUT_ROOTS="$GOOD:$BADROOT/child" \
    BOOT_PAYLOAD_PATH="$MISS" BOOT_LAUNCHER_PATH="$MISS" \
    BOOT_STAGED_MODEL_PATH="$MISS" BOOT_META_PATH="$MISS" BOOT_RECEIPT_PATH="$MISS" \
    bash "$HERE/bootstrap_diag.sh" > "$out" 2>&1 || RC=$?
}

jassert() { # $1=json-file
  python3 - "$1" <<'PY' || fail "json assertions failed for $1"
import json, os, sys
d = json.load(open(sys.argv[1]))
assert d["schema"].startswith("bootstrap-diag/v2"), d["schema"]
assert d["request_hash"] and len(d["request_hash"]) == 16
assert d["identity"]["uid"] is not None
p = d["paths"]["project_payload"]
assert p["exists"] is False, p
raw = open(sys.argv[1]).read()
for k in ("QUALIFICATION_RUN_ID", "NNODES", "TOKEN", "DAN_SCALE_DIR", "HF_HUB_OFFLINE"):
    v = d["qualification_env"].get(k)
    assert v is not None, k
    assert v["present"] == (k in os.environ), k
    if k in os.environ and os.environ[k]:
        assert os.environ[k] not in raw, f"env VALUE leaked for {k}"
print("JSON_OK")
PY
}

# ---------- PART 1a: baseline — API env present + matching ----------
run_diag "$T/a.out" \
  BOOT_RUN_ID=20260913T010101Z BOOT_NONCE=aaaa0001 \
  BOOT_API_SENTINEL_EXPECTED=boot-sentinel-aaaa0001 \
  BOOT_API_SENTINEL=boot-sentinel-aaaa0001
[[ $RC -eq 0 ]] || fail "1a exit $RC"
grep -q "DIAG_SUMMARY run=20260913T010101Z nonce=aaaa0001" "$T/a.out" || fail "1a summary"
grep -q "PUBLISH_OK $GOOD/bootstrap-20260913T010101Z-aaaa0001" "$T/a.out" || fail "1a publish"
grep -q "PUBLISH_MKDIR_FAIL" "$T/a.out" || fail "1a bad-root error not recorded"
grep -q "publish_ok=1/2" "$T/a.out" || fail "1a counter"
DJSON="$GOOD/bootstrap-20260913T010101Z-aaaa0001/diag.json"
[[ -s "$DJSON" ]] || fail "1a diag.json"
python3 - "$DJSON" <<'PY' || fail "1a sentinel asserts"
import json, sys
d = json.load(open(sys.argv[1]))
s = d["api_sentinel"]
assert s == {"configured": True, "present": True, "matches_expected": True}, s
assert d["paths"]["project_launcher"]["exists"] is False
ubp = d["python"]["usr_bin_python"]
assert ubp["path"] == "/usr/bin/python" and "exists" in ubp
print("1A_SENTINEL_OK")
PY
jassert "$DJSON"
[[ -s "$GOOD/bootstrap-20260913T010101Z-aaaa0001/stdout.log" ]] || fail "1a stdout snapshot"
echo "PART1a_PASS (baseline: envs channel delivers, match=true)"

# ---------- PART 1b: API env MISSING — inline controls still work ----------
RC=0
env -u BOOT_API_SENTINEL \
  BOOT_RUN_ID=20260913T010202Z BOOT_NONCE=bbbb0002 \
  BOOT_API_SENTINEL_EXPECTED=boot-sentinel-bbbb0002 \
  BOOT_OUT_ROOTS="$GOOD:$BADROOT/child" \
  BOOT_PAYLOAD_PATH="$MISS" BOOT_LAUNCHER_PATH="$MISS" \
  BOOT_STAGED_MODEL_PATH="$MISS" BOOT_META_PATH="$MISS" BOOT_RECEIPT_PATH="$MISS" \
  bash "$HERE/bootstrap_diag.sh" > "$T/b.out" 2>&1 || RC=$?
[[ $RC -eq 0 ]] || fail "1b exit $RC"
grep -q "sentinel_present=false" "$T/b.out" || fail "1b summary sentinel flag"
DJSONB="$GOOD/bootstrap-20260913T010202Z-bbbb0002/diag.json"
python3 - "$DJSONB" <<'PY' || fail "1b sentinel asserts"
import json, sys
s = json.load(open(sys.argv[1]))["api_sentinel"]
assert s == {"configured": True, "present": False, "matches_expected": None}, s
print("1B_SENTINEL_MISSING_RECORDED")
PY
jassert "$DJSONB"
echo "PART1b_PASS (API env missing -> recorded absent; inline controls still produced full diag)"

# ---------- PART 1c: unusable python -> honest fallback, publish still works -------
BIN="$T/bin"; mkdir -p "$BIN"
for b in bash date id hostname sha256sum cut tee mktemp mkdir cp tr sed sleep; do
  SRC="$(command -v "$b")" && ln -s "$SRC" "$BIN/$b"
done
RC=0
env -i PATH="$BIN" HOME="$T" \
  BOOT_RUN_ID=20260913T010303Z BOOT_NONCE=cccc0003 \
  BOOT_API_SENTINEL_EXPECTED=boot-sentinel-cccc0003 \
  BOOT_API_SENTINEL=boot-sentinel-cccc0003 \
  BOOT_PY_BIN=/nonexistent/python \
  BOOT_OUT_ROOTS="$GOOD:$BADROOT/child" \
  BOOT_PAYLOAD_PATH="$MISS" BOOT_LAUNCHER_PATH="$MISS" \
  BOOT_STAGED_MODEL_PATH="$MISS" BOOT_META_PATH="$MISS" BOOT_RECEIPT_PATH="$MISS" \
  bash "$HERE/bootstrap_diag.sh" > "$T/c.out" 2>&1 || RC=$?
[[ $RC -eq 0 ]] || fail "1c exit $RC"
grep -qE "py_status=python-(missing|error)" "$T/c.out" || fail "1c status"
grep -q "PUBLISH_OK" "$T/c.out" || fail "1c publish"
DJSONC="$GOOD/bootstrap-20260913T010303Z-cccc0003/diag.json"
python3 - "$DJSONC" <<'PY' || fail "1c fallback asserts"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["schema"] == "bootstrap-diag/v2-fallback", d["schema"]
assert d["api_sentinel_present"] is True
assert d["python_status"].startswith("python-"), d["python_status"]
print("1C_FALLBACK_HONEST")
PY
echo "PART1c_PASS (unusable python -> v2-fallback JSON + sentinel flag, no crash)"

# ---------- PART 2: wrapper + frozen spec round-trip QA (payload NOT executed) -----
python3 - "$HERE/job_spec_candidate.json" "$HERE/request_frozen.json" \
         "$HERE/bootstrap_diag.sh" "$T/decoded_payload.sh" <<'PY' || fail "spec QA asserts"
import base64, hashlib, json, re, shlex, subprocess, sys
spec = json.load(open(sys.argv[1]))
frozen = json.load(open(sys.argv[2]))
payload_path, decoded_path = sys.argv[3], sys.argv[4]
payload = open(payload_path).read()

cmd = spec["command"]
assert "\n" not in cmd, "command must stay single-line"

# FAILING-FIRST structural gate: explicit shell wrapper, standard quoting only.
argv = shlex.split(cmd)
assert argv[0] == "/usr/bin/bash" and argv[1] == "-lc", argv[:2]
inner = argv[2]
assert shlex.split(cmd) == ["/usr/bin/bash", "-lc", inner]

# Inner preserves decode + inline controls + cleanup + exit status.
assert " | base64 -d > " in inner and 'trap \'rm -f "$d"\' EXIT' in inner
assert "BOOTSTRAP_DIAG_EXIT=$rc" in inner and "exit $rc" in inner
assert "BOOT_API_SENTINEL=" not in inner, "sentinel actual must travel via envs only"
assert "BOOT_API_SENTINEL_EXPECTED=" in inner
for k in ("BOOT_RUN_ID=", "BOOT_NONCE=", "BOOT_OUT_ROOTS=", "BOOT_PAYLOAD_PATH=",
          "BOOT_LAUNCHER_PATH=", "BOOT_STAGED_MODEL_PATH=", "BOOT_RECEIPT_PATH="):
    assert k in inner, k
rms = re.findall(r'rm -f "[^"]*"', inner)
assert rms == ['rm -f "$d"'], rms  # the ONLY removal is the wrapper's own temp file
assert "curl " not in inner and "pip " not in inner and "chmod " not in inner

m = re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d", inner)
assert m, "payload b64 not found in inner"
decoded = base64.b64decode(m.group(1), validate=True).decode()
h = hashlib.sha256(decoded.encode()).hexdigest()
assert h == frozen["payload_sha256"] == hashlib.sha256(payload.encode()).hexdigest(), "decode hash mismatch"
open(decoded_path, "w").write(decoded)

# Inner must be syntactically valid for the exact interpreter it names.
r = subprocess.run(["/usr/bin/bash", "-n", "-c", inner], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[:500]

res = frozen["resources"]
assert res["logic_compute_group_id"] == "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
assert res["project_id"] == "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
assert res["workspace_id"] == "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"
assert res["spec_id"] == "7166bd2e-6cbe-4bd9-be38-762d11003e7f"
assert res["image"].endswith("relay2:v2") and "PENDING" not in res["logic_compute_group_id"]

c = frozen["caps"]
assert (c["gpu_type"], c["nodes"], c["gpus_per_node"]) == ("NVIDIA_H100_SXM_80G", 1, 8)
assert c["max_running_time_ms"] == 180000 and c["task_priority"] == 4
assert c["auto_fault_tolerance"] is False and c["fault_tolerance_max_retry"] == 0
assert c["internal_timeout_term_s"] == 120 and c["internal_kill_grace_s"] == 5

assert not any(k.startswith("reserve") for k in spec), "reserve keys must be ABSENT"
assert spec["max_running_time_ms"] == "180000" and spec["task_priority"] == 4
assert spec["auto_fault_tolerance"] is False and spec["fault_tolerance_max_retry"] == 0
fc = spec["framework_config"][0]
assert fc["instance_count"] == 1 and fc["shm_gi"] == 1800
assert spec["name"] == frozen["job_name"]
assert spec["name"] == "bootstrap-diag-p1-h100-20260913T073456Z-1fe0b3fe", "identity must be preserved"
assert spec["envs"] == [{"name": "BOOT_API_SENTINEL", "value": spec["envs"][0]["value"]}]
assert len(spec["envs"]) == 1 and spec["envs"][0]["name"] == "BOOT_API_SENTINEL"
assert hashlib.sha256(spec["envs"][0]["value"].encode()).hexdigest() == frozen["api_sentinel"]["value_sha256"]
assert hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() == frozen["request_sha256"]
print("PART2_SPEC_OK (bash -lc wrapper, decode-hash match, identity pinned, caps intact)")
PY
bash -n "$T/decoded_payload.sh" || fail "decoded payload bash syntax"
cmp -s "$T/decoded_payload.sh" "$HERE/bootstrap_diag.sh" || fail "round-trip mismatch"

# ---------- PART 3: wrapper behavior — cleanup + exit-status preservation ----------
FPAY="$T/fake_payload.sh"
MARKER="$T/wrapper_paths.txt"
{
  printf 'printf "%%s\\n" "$0" >> "%s"\n' "$MARKER"
  printf 'exit 7\n'
} > "$FPAY"
b64="$(base64 -w0 "$FPAY")"
inner2="$(cat <<INNER
d=/tmp/wraptest.\$\$.sh; trap 'rm -f "\$d"' EXIT; echo $b64 | base64 -d > "\$d" && bash "\$d"; rc=\$?; echo BOOTSTRAP_DIAG_EXIT=\$rc; exit \$rc
INNER
)"
WOUT="$T/wrapper.out"
WRC=0
/usr/bin/bash -lc "$inner2" > "$WOUT" 2>&1 || WRC=$?
[[ $WRC -eq 7 ]] || fail "wrapper rc=$WRC (expected 7 — status must be preserved)"
grep -q "BOOTSTRAP_DIAG_EXIT=7" "$WOUT" || fail "wrapper did not echo exit status"
[[ -s "$MARKER" ]] || fail "fake payload never ran"
read -r RANPATH < "$MARKER"
[[ -n "$RANPATH" && ! -e "$RANPATH" ]] || fail "temp script not trap-removed: $RANPATH"
echo "PART3_PASS (wrapper: pipe executed, rc=7 preserved, temp script trap-removed)"

# freeze guard: rebuild must refuse
GRC=0
bash "$HERE/build_spec.sh" > "$T/guard.out" 2>&1 || GRC=$?
[[ $GRC -eq 7 ]] || fail "freeze guard rc=$GRC (expected 7)"
grep -q "FROZEN" "$T/guard.out" || fail "freeze guard message"

echo "ALL_PASS (local fixture + wrapper + frozen-request QA; NOT a node proof)"
