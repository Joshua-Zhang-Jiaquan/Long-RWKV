#!/usr/bin/env bash
# bootstrap_diag.sh — self-contained worker bootstrap probe (CPU + driver-query only).
#
# Contract (campaign nonlatent-rnn-dlm-iclr-research / bootstrap-20260913-01):
#   1. Starts WITHOUT the project source path being mounted: the job embeds this
#      file inline (base64 in `command`); nothing here reads the project mount to start.
#   2. Prints observations to stdout immediately (tee) so GetJobLog captures them
#      even if the job dies in seconds.
#   3. Best-effort durable publish of diag.json + stdout snapshot into a unique
#      child of KNOWN-GOOD GPFS output roots. Write errors are recorded verbatim,
#      never "fixed" (no chmod / permission changes anywhere).
#   4. Secrets policy: never prints env values, configs, or tokens — only key
#      presence and approved-value match booleans.
#   5. Bounded: per-exec timeout <=15s, watchdog TERM@120s + KILL grace@+5s.
#      No model load, no training, no package install, no CUDA compute, no corpus IO.
set -u
export PYTHONUNBUFFERED=1
export LC_ALL=C

RUN_ID="${BOOT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
NONCE="${BOOT_NONCE:-$(printf '%04x%04x' "$((RANDOM * 32768 + RANDOM))" "$((RANDOM * 32768 + RANDOM))")}"
START_EPOCH="$(date +%s)"
REQUEST_TAG="nonlatent-rnn-dlm-iclr-research|bootstrap-20260913-01"
REQUEST_HASH="$(printf '%s' "$REQUEST_TAG" | sha256sum | cut -c1-16)"

TMPD="$(mktemp -d /tmp/bootstrap_diag.XXXXXX)" || { echo "FATAL: mktemp failed"; exit 3; }
STDOUT_LOG="$TMPD/stdout.log"
: > "$STDOUT_LOG"

# Immediate stdout capture: tee keeps GetJobLog live even on early death.
if command -v tee >/dev/null 2>&1; then
  exec > >(tee "$STDOUT_LOG") 2>&1
else
  # tee missing is itself a finding; stdout goes to the snapshot file only.
  exec > "$STDOUT_LOG" 2>&1
  echo "WARN: tee not found; stdout only captured via durable publish/GetJobLog"
fi

echo "== bootstrap-diag v1 run=$RUN_ID nonce=$NONCE start=$(date -u +%Y-%m-%dT%H:%M:%SZ) =="
echo "== request_tag=$REQUEST_TAG request_hash=$REQUEST_HASH =="
echo "== uid=$(id -u) gid=$(id -g) host=$(hostname) pid=$$ =="

# Watchdog: TERM at 120s, KILL grace +5s. A leftover `sleep` after normal exit
# only fires kill on a dead pid and is reaped by job teardown.
( sleep 120; kill -TERM "$$" 2>/dev/null; sleep 5; kill -KILL "$$" 2>/dev/null ) &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT

# --- probe targets (defaults are campaign-known paths; override via BOOT_* env) ---
# Hypothesis under test: project source path unmounted/unreadable on worker.
PAYLOAD_PATH="${BOOT_PAYLOAD_PATH:-/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY}"
STAGED_MODEL_PATH="${BOOT_STAGED_MODEL_PATH:-/inspire/hdd/global_user/zhangjiaquan-253108540222/qz_stage_traj4096_v7/scale}"
META_PATH="${BOOT_META_PATH:-}"        # checkpoint metadata FILE (never weights); set once layout is confirmed
RECEIPT_PATH="${BOOT_RECEIPT_PATH:-}"  # controller receipt: intentionally unset (diagnostic is not model payload)

DEFAULT_REF_ROOT="/inspire/hdd/global_user/zhangjiaquan-253108540222/outputs_birwkv_diffusion/logs"
OUT_ROOTS="${BOOT_OUT_ROOTS:-$DEFAULT_REF_ROOT}"   # space-separated; every root attempted

export RUN_ID NONCE REQUEST_HASH PAYLOAD_PATH STAGED_MODEL_PATH META_PATH RECEIPT_PATH START_EPOCH

PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$(command -v "$cand")"; break; fi
done

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

def probe_path(p):
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

# Python interpreter versions (process metadata only; nothing is imported here).
pys = {}
for e in execs:
    if e["name"] in ("python3", "python") and e["found"]:
        r = run([e["path"], "-VV"], 10)
        pys[e["name"]] = {"path": e["path"], "version": r.get("stdout", "")[:300], "code": r.get("code")}

# Package dist metadata via importlib.metadata: reads installed-dist metadata,
# does NOT import the packages (so no CUDA/c-extension init).
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
    "staged_model_dir": probe_path(env("STAGED_MODEL_PATH")),
    "checkpoint_metadata_file": probe_path(env("META_PATH")),
    "controller_receipt": probe_path(env("RECEIPT_PATH")),
}

# Approved qualification env keys: presence + approved-value match. NEVER values.
EXPECTED = {"NNODES": "4", "NGPUS": "8", "MICROBATCH": "4", "GRAD_ACCUM": "6",
            "STEPS": "6000", "TOKEN": None, "DAN_SCALE_DIR": None}
quals = {}
for k, want in EXPECTED.items():
    present = k in os.environ
    matches = None if (not present or want is None) else (os.environ[k] == want)
    quals[k] = {"present": present, "matches_expected": matches}

# GPU inventory via nvidia-smi only if the binary exists (CPU-side driver query;
# no CUDA context, no compute).
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
    "schema": "bootstrap-diag/v1",
    "run_id": env("RUN_ID"), "nonce": env("NONCE"),
    "request_hash": env("REQUEST_HASH"),
    "purpose": "worker bootstrap environment probe; NOT a model qualification result",
    "time_utc": iso(time.time()), "start_epoch": env("START_EPOCH"),
    "identity": {"uid": os.getuid(), "gid": os.getgid(), "hostname": socket.gethostname()},
    "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    "executables": execs, "python": pys, "package_metadata": pkgs,
    "paths": paths, "qualification_env": quals, "gpu_probe": gpu,
}
with open(out_path, "w") as f:
    json.dump(diag, f, indent=2, sort_keys=True)

def brief(p):
    if p.get("status") == "not_configured":
        return "not_configured"
    if not p.get("exists"):
        return "MISSING(" + p.get("error", "no such path")[:100] + ")"
    return (f"ok dir={p.get('is_dir')} file={p.get('is_file')} r={p.get('readable')} "
            f"x={p.get('executable')} size={p.get('size')} mode={p.get('mode')}")

print("PATH project_payload:     ", brief(paths["project_payload"]))
print("PATH staged_model_dir:    ", brief(paths["staged_model_dir"]))
print("PATH checkpoint_metadata: ", brief(paths["checkpoint_metadata_file"]))
print("PATH controller_receipt:  ", brief(paths["controller_receipt"]))
print("PY ", {k: v.get("version", "")[:60] for k, v in pys.items()})
print("PKGS", {k: v.get("version") for k, v in pkgs.items() if not k.startswith("_")})
print("ENV", {k: ("present" if v["present"] else "absent") +
              ("" if v["matches_expected"] is None else f"/match={v['matches_expected']}")
              for k, v in quals.items()})
print("GPU", {k: gpu.get(k) for k in ("attempted", "count", "code", "error") if k in gpu})
print("PYDIAG_OK")
PYDIAG
  PY_RC=$?
  if [[ $PY_RC -ne 0 || ! -s "$DIAG_JSON" ]]; then
    PY_STATUS="python-error(rc=$PY_RC)"
  fi
fi

if [[ "$PY_STATUS" != "ok" ]]; then
  echo "WARN: python diagnostics unavailable (status=$PY_STATUS); writing fallback minimal JSON"
  esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
  {
    printf '{\n "schema": "bootstrap-diag/v1-fallback",\n'
    printf ' "run_id": "%s", "nonce": "%s", "request_hash": "%s",\n' "$(esc "$RUN_ID")" "$(esc "$NONCE")" "$(esc "$REQUEST_HASH")"
    printf ' "uid": %s, "gid": %s, "hostname": "%s",\n' "$(id -u)" "$(id -g)" "$(esc "$(hostname)")"
    printf ' "python_status": "%s",\n' "$(esc "$PY_STATUS")"
    printf ' "payload_path": "%s",\n' "$(esc "$PAYLOAD_PATH")"
    printf ' "staged_model_path": "%s"\n' "$(esc "$STAGED_MODEL_PATH")"
    printf '}\n'
  } > "$DIAG_JSON"
fi

echo "== publish (best-effort; exact errors recorded, never fixed) =="
PUBLISH_LOG="$TMPD/publish.status"
: > "$PUBLISH_LOG"
IFS=' ' read -r -a ROOTS <<< "$OUT_ROOTS"
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

echo "DIAG_SUMMARY run=$RUN_ID nonce=$NONCE py_status=$PY_STATUS publish_ok=$N_OK/$N_TRY elapsed=$(( $(date +%s) - START_EPOCH ))s"
echo "== done =="
exit 0
