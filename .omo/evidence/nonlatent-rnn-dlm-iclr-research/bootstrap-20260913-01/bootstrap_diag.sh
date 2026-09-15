#!/usr/bin/env bash
# bootstrap_diag.sh v2 — self-contained worker bootstrap probe (CPU + driver-query only).
#
# Contract (campaign nonlatent-rnn-dlm-iclr-research / bootstrap-20260913-01):
#   1. Starts WITHOUT the project source mount: job embeds this file inline (base64);
#      control values (run id, nonce, output roots, probe paths, sentinel-expected)
#      are passed INLINE on the decoded command line, NOT via the API env channel.
#   2. BOOT_API_SENTINEL arrives ONLY via job envs; expected value is inline. The
#      logger records presence + match booleans — a pod-level proof of whether the
#      scheduler env channel delivers envs at all (GetJob envs=[] is NOT pod proof).
#   3. Stdout captured immediately (tee) for GetJobLog; watchdog TERM@120s/KILL@+5s;
#      per-probe timeout <=15s. No model load, no torch import, no CUDA compute,
#      no installs, no env/config/token value dumps, no chmod.
#   4. Best-effort durable publish (unique children, no chmod) into KNOWN-GOOD GPFS
#      roots; every write error recorded verbatim. ROOTS are colon-separated.
set -u
export PYTHONUNBUFFERED=1
export LC_ALL=C

RUN_ID="${BOOT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
NONCE="${BOOT_NONCE:-$(printf '%04x%04x' "$((RANDOM * 32768 + RANDOM))" "$((RANDOM * 32768 + RANDOM))")}"
START_EPOCH="$(date +%s)"
REQUEST_TAG="nonlatent-rnn-dlm-iclr-research|bootstrap-20260913-01|v2"
REQUEST_HASH="$(printf '%s' "$REQUEST_TAG" | sha256sum | cut -c1-16)"

TMPD="$(mktemp -d /tmp/bootstrap_diag.XXXXXX)" || { echo "FATAL: mktemp failed"; exit 3; }
STDOUT_LOG="$TMPD/stdout.log"
: > "$STDOUT_LOG"

if command -v tee >/dev/null 2>&1; then
  exec > >(tee "$STDOUT_LOG") 2>&1
else
  exec > "$STDOUT_LOG" 2>&1
  echo "WARN: tee not found; stdout only via durable publish/GetJobLog"
fi

echo "== bootstrap-diag v2 run=$RUN_ID nonce=$NONCE start=$(date -u +%Y-%m-%dT%H:%M:%SZ) =="
echo "== request_tag=$REQUEST_TAG request_hash=$REQUEST_HASH =="
echo "== uid=$(id -u) gid=$(id -g) host=$(hostname) pid=$$ =="

( sleep 120; kill -TERM "$$" 2>/dev/null; sleep 5; kill -KILL "$$" 2>/dev/null ) &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT

# --- probe targets: INLINE defaults from build_spec (overridable by env) ---
PAYLOAD_PATH="${BOOT_PAYLOAD_PATH:-/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY}"
LAUNCHER_PATH="${BOOT_LAUNCHER_PATH:-/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/scale/experiments/nonlatent_iclr/qualification/run_qualification.sh}"
STAGED_MODEL_PATH="${BOOT_STAGED_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale}"
RECEIPT_PATH="${BOOT_RECEIPT_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification/controller-receipts/qualification-20260912-01-0ff9241e-460a-4647-95b3-af4c0fba16ce.json}"
META_PATH="${BOOT_META_PATH:-}"   # 43-byte checkpoint meta: NOT located yet -> not_configured (stat only if set)

DEFAULT_REF_ROOT="/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/logs"
DEFAULT_QUAL_ROOT="/inspire/hdd/global_user/zhangjiaquan-253108540222/nonlatent_iclr_qualification"
OUT_ROOTS="${BOOT_OUT_ROOTS:-$DEFAULT_REF_ROOT:$DEFAULT_QUAL_ROOT}"   # colon-separated

export RUN_ID NONCE REQUEST_HASH PAYLOAD_PATH LAUNCHER_PATH STAGED_MODEL_PATH RECEIPT_PATH META_PATH SENTINEL_EXPECTED="${BOOT_API_SENTINEL_EXPECTED:-}" START_EPOCH

PY=""
for cand in /usr/bin/python python3 python; do
  command -v "$cand" >/dev/null 2>&1 && { PY="$(command -v "$cand")"; break; }
done
PY="${BOOT_PY_BIN:-$PY}"   # optional pin (also the test seam for the fallback path)

DIAG_JSON="$TMPD/diag.json"
PY_STATUS="python-missing"
if [[ -n "$PY" ]]; then
  PY_STATUS="ok"
  "$PY" - "$DIAG_JSON" <<'PYDIAG'
import json, os, platform, socket, subprocess, shutil, sys, time
from datetime import datetime, timezone

out_path = sys.argv[1]
env = os.environ.get

def iso(epoch):
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def probe_path(p, with_version=False):
    rec = {"path": p}
    if not p:
        rec["status"] = "not_configured"
        return rec
    try:
        st = os.stat(p)
    except OSError as e:
        rec["exists"] = os.path.exists(p)
        rec["status"] = "stat_error"
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        return rec
    rec.update({
        "exists": True, "is_dir": os.path.isdir(p), "is_file": os.path.isfile(p),
        "readable": os.access(p, os.R_OK), "executable": os.access(p, os.X_OK),
        "size": st.st_size, "mode": oct(st.st_mode & 0o7777), "mtime_utc": iso(st.st_mtime),
    })
    if with_version and rec["executable"]:
        r = run([p, "-VV"], 10)
        rec["version"] = r.get("stdout", "")[:200]
    return rec

def which(name):
    p = shutil.which(name)
    return {"name": name, "path": p, "found": p is not None}

def run(cmd, t=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return {"cmd": cmd[0], "code": r.returncode,
                "stdout": r.stdout.strip()[-4000:],
                "error": r.stderr.strip()[-1000:] if r.returncode else ""}
    except Exception as e:
        return {"cmd": cmd[0], "code": None, "error": f"{type(e).__name__}: {e}"[:300]}

execs = [which(n) for n in ("bash", "python3", "python", "nvidia-smi", "tee", "mktemp", "stat", "sha256sum")]

pys = {}
for e in execs:
    if e["name"] in ("python3", "python") and e["found"]:
        r = run([e["path"], "-VV"], 10)
        pys[e["name"]] = {"path": e["path"], "version": r.get("stdout", "")[:200], "code": r.get("code")}
# explicit /usr/bin/python probe (exact path, not PATH discovery)
ubp = probe_path("/usr/bin/python", with_version=True)
ubp["path"] = "/usr/bin/python"
pys["usr_bin_python"] = ubp

# dist metadata only — packages are never imported (no CUDA init)
pkgs = {}
try:
    import importlib.metadata as im
    for name in ("torch", "fla", "pydantic"):
        try:
            pkgs[name] = {"version": im.version(name), "status": "ok"}
        except Exception as e:
            pkgs[name] = {"version": None, "status": type(e).__name__}
except Exception as e:
    pkgs = {"_error": f"{type(e).__name__}: {e}"[:300]}

paths = {
    "project_payload": probe_path(env("PAYLOAD_PATH")),
    "project_launcher": probe_path(env("LAUNCHER_PATH")),          # x-bit is the key hypothesis
    "staged_model_dir": probe_path(env("STAGED_MODEL_PATH")),
    "controller_receipt": probe_path(env("RECEIPT_PATH")),         # read-stat only
    "checkpoint_metadata_file": probe_path(env("META_PATH")),      # read-stat only, if configured
}

# scheduler env channel pod-proof: sentinel actual (envs-only) vs expected (inline)
sent = {"configured": bool(env("SENTINEL_EXPECTED")),
        "present": "BOOT_API_SENTINEL" in os.environ,
        "matches_expected": None}
if sent["configured"] and sent["present"]:
    sent["matches_expected"] = (os.environ["BOOT_API_SENTINEL"] == env("SENTINEL_EXPECTED"))

# qualification-family + training-family env keys: presence/match only, NEVER values
EXPECTED = {"QUALIFICATION_RUN_ID": None, "QUALIFICATION_NONCE": None,
            "QUALIFICATION_MANIFEST_SHA256": None, "QUALIFICATION_JOB_RECEIPT": None,
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "NNODES": "4", "NGPUS": "8", "MICROBATCH": "4", "GRAD_ACCUM": "6",
            "STEPS": "6000", "TOKEN": None, "DAN_SCALE_DIR": None}
quals = {}
for k, want in EXPECTED.items():
    present = k in os.environ
    matches = None if (not present or want is None) else (os.environ[k] == want)
    quals[k] = {"present": present, "matches_expected": matches}

gpu = {"attempted": False}
nv = next((e for e in execs if e["name"] == "nvidia-smi" and e["found"]), None)
if nv:
    gpu["attempted"] = True
    r = run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 15)
    gpu["code"] = r.get("code")
    if r.get("code") == 0 and r.get("stdout"):
        names = [l.strip() for l in r["stdout"].splitlines() if l.strip()]
        gpu["count"] = len(names)
        gpu["names"] = sorted(set(names))
    else:
        gpu["count"] = None
        gpu["error"] = (r.get("error") or "no output")[:500]

diag = {
    "schema": "bootstrap-diag/v2",
    "run_id": env("RUN_ID"), "nonce": env("NONCE"), "request_hash": env("REQUEST_HASH"),
    "purpose": "worker bootstrap environment probe; NOT a model qualification result",
    "time_utc": iso(time.time()), "start_epoch": env("START_EPOCH"),
    "identity": {"uid": os.getuid(), "gid": os.getgid(), "hostname": socket.gethostname()},
    "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    "executables": execs, "python": pys, "package_metadata": pkgs,
    "paths": paths, "api_sentinel": sent, "qualification_env": quals, "gpu_probe": gpu,
}
with open(out_path, "w") as f:
    json.dump(diag, f, indent=2, sort_keys=True)

def brief(p):
    if p.get("status") == "not_configured":
        return "not_configured"
    if not p.get("exists"):
        return "MISSING(" + p.get("error", "no such path")[:100] + ")"
    return (f"exists dir={p.get('is_dir')} r={p.get('readable')} x={p.get('executable')} "
            f"size={p.get('size')} mode={p.get('mode')}")

print("PATH project_payload:     ", brief(paths["project_payload"]))
print("PATH project_launcher:    ", brief(paths["project_launcher"]))
print("PATH staged_model_dir:    ", brief(paths["staged_model_dir"]))
print("PATH controller_receipt:  ", brief(paths["controller_receipt"]))
print("PATH checkpoint_metadata: ", brief(paths["checkpoint_metadata_file"]))
print("PY ", {k: v.get("version", "")[:60] for k, v in pys.items()})
print("PKGS", {k: v.get("version") for k, v in pkgs.items() if not k.startswith("_")})
print("SENTINEL", sent)
print("ENV", {k: ("present" if v["present"] else "absent") +
              ("" if v["matches_expected"] is None else f"/match={v['matches_expected']}")
              for k, v in quals.items()})
print("GPU", {k: gpu.get(k) for k in ("attempted", "count", "code", "error") if k in gpu})
print("PYDIAG_OK")
PYDIAG
  PY_RC=$?
  [[ $PY_RC -ne 0 || ! -s "$DIAG_JSON" ]] && PY_STATUS="python-error(rc=$PY_RC)"
fi

if [[ "$PY_STATUS" != "ok" ]]; then
  echo "WARN: python diagnostics unavailable (status=$PY_STATUS); writing fallback minimal JSON"
  esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
  {
    printf '{\n "schema": "bootstrap-diag/v2-fallback",\n'
    printf ' "run_id": "%s", "nonce": "%s", "request_hash": "%s",\n' "$(esc "$RUN_ID")" "$(esc "$NONCE")" "$(esc "$REQUEST_HASH")"
    printf ' "uid": %s, "gid": %s, "hostname": "%s",\n' "$(id -u)" "$(id -g)" "$(esc "$(hostname)")"
    printf ' "python_status": "%s",\n' "$(esc "$PY_STATUS")"
    printf ' "api_sentinel_present": %s,\n' "$([[ -n "${BOOT_API_SENTINEL:-}" ]] && echo true || echo false)"
    printf ' "launcher_path": "%s", "payload_path": "%s"\n' "$(esc "$LAUNCHER_PATH")" "$(esc "$PAYLOAD_PATH")"
    printf '}\n'
  } > "$DIAG_JSON"
fi

echo "== publish (best-effort; exact errors recorded, never fixed) =="
PUBLISH_LOG="$TMPD/publish.status"
: > "$PUBLISH_LOG"
IFS=' :' read -r -a ROOTS <<< "$OUT_ROOTS"   # colon- or space-separated
N_OK=0
N_TRY=0
for root in "${ROOTS[@]}"; do
  [[ -n "$root" ]] || continue
  N_TRY=$((N_TRY + 1))
  CHILD="$root/bootstrap-$RUN_ID-$NONCE"
  if mkdir -p "$CHILD" 2> "$TMPD/err.txt"; then
    if cp "$DIAG_JSON" "$STDOUT_LOG" "$CHILD/" 2> "$TMPD/err.txt"; then
      MSG="PUBLISH_OK $CHILD"
      N_OK=$((N_OK + 1))
    else
      MSG="PUBLISH_COPY_FAIL $CHILD err=$(tr '\n' ' ' < "$TMPD/err.txt")"
    fi
  else
    MSG="PUBLISH_MKDIR_FAIL $CHILD err=$(tr '\n' ' ' < "$TMPD/err.txt")"
  fi
  echo "$MSG"
  printf '%s\n' "$MSG" >> "$PUBLISH_LOG"
done

echo "DIAG_SUMMARY run=$RUN_ID nonce=$NONCE py_status=$PY_STATUS sentinel_present=$([[ -n "${BOOT_API_SENTINEL:-}" ]] && echo true || echo false) publish_ok=$N_OK/$N_TRY elapsed=$(( $(date +%s) - START_EPOCH ))s"
echo "== done =="
exit 0
