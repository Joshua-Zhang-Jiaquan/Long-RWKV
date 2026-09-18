#!/usr/bin/env bash
# launch_efficiency_probe.sh -- the measured efficiency matrix (paper contribution 1).
#
# Runs scale/experiments/nonlatent_iclr/efficiency/probe_inference.py once per (model,
# context) cell on ONE GPU and writes one JSON per cell to $OUTDIR/<model_key>_<context>.json.
# This experiment is the evidence for the paper's efficiency claim, so a cell that does not
# fit is recorded as the probe's own "oom" row and the grid keeps going; only a failure to
# IMPORT the runner or to LOAD a checkpoint aborts the grid, because those are defects in the
# run rather than measurements that did not fit.
#
# The runner has no CLI, so it is driven through `python - ` importing probe_ladder and
# adapters -- see the payload below. A single-rung probe_ladder per cell (rather than one
# ladder for the whole model) is deliberate: the runner's ladder stops at the first OOM, and
# that stop would silently drop the larger contexts this grid is meant to attempt.
#
# Modes (MODE env):
#   probe  - run the grid (default)
# Any other MODE is refused: a typo must not quietly run nothing.
#
# Env:
#   OUTDIR      (required) per-cell JSON, boot log and run log are written here
#   MODEL_KEYS  comma-separated registry keys, or "all" (default: all)
#   CONTEXTS    comma-separated positive contexts, strictly ascending
#               (default 4096,16384,32768,65536)
#   NGPUS       must be 1 (default 1); a multi-GPU run cannot support a per-device claim
#   MODELS_ROOT root the hub checkpoints resolve under (default: the recorded root)
#   IMPORT_ROOT tree the scale package is imported from (default: this checkout)
#   DEVICE      torch device the probe measures on (default cuda:0)
#   DRY_RUN     1 - validate the grid, print the planned cells and exit without a device

set -uo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# The tree the runner is imported from.  The default is this checkout, but the repo stages a
# second copy of the scale tree for the cluster -- the same reason launch_birwkv_diffusion.sh
# reads DAN_SCALE_DIR -- so the import root has to be pointable at the tree under test.  The
# test suite points it at a fake scale package to exercise the grid without a checkpoint.
readonly IMPORT_ROOT="${IMPORT_ROOT:-$REPO_ROOT}"

# Roots the registry's relative checkpoint paths resolve under.  The default matches
# adapters.RECORDED_MODELS_ROOT, the root the baked weights_bytes constants were measured
# against; a different root must be given explicitly so a run cannot silently score the
# wrong checkpoint.
readonly DEFAULT_MODELS_ROOT="/inspire/hdd/global_user/zhangjiaquan-253108540222/models"

# One interpreter for the CUDA preflight, the registry preflight and the payload, so the
# three cannot silently run under different runtimes.
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"

# Put the import root first on sys.path.  The python steps below import the scale package
# from here, and prepending means an IMPORT_ROOT the caller set is the tree under test even
# when the ambient PYTHONPATH already names a different copy of it.
export PYTHONPATH="${IMPORT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# The probe is a single-device measurement.  A peak-memory reading describes the device it
# was taken on, so a run that spread cells over several GPUs could not support the table's
# per-device numbers at all -- this is why a mismatch is a hard failure, not a warning.
readonly REQUIRED_NGPUS=1

# The ladder probe_inference.CONTEXTS reports.  Named here so the launcher and the runner
# cannot drift apart without a reviewer seeing it.
readonly DEFAULT_CONTEXTS="4096,16384,32768,65536"

# The package the preflight and the payload both import; named once so a rename lands in
# one place.
readonly RUNNER_PACKAGE="scale.experiments.nonlatent_iclr.efficiency"

MODE="${MODE:-probe}"
case "$MODE" in
  probe) ;;
  *)
    echo "[launch] FATAL: unknown MODE=${MODE}; this launcher runs only MODE=probe" >&2
    exit 2
    ;;
esac

NGPUS="${NGPUS:-$REQUIRED_NGPUS}"
if [[ "$NGPUS" != "$REQUIRED_NGPUS" ]]; then
  echo "[launch] FATAL: NGPUS=${NGPUS} but the efficiency matrix is a single-device measurement;" >&2
  echo "[launch] a multi-GPU run cannot support a per-device memory claim. Set NGPUS=1." >&2
  exit 2
fi

OUTDIR="${OUTDIR:-}"
if [[ -z "$OUTDIR" ]]; then
  echo "[launch] FATAL: OUTDIR is required (the per-cell JSON, boot log and run log go there)" >&2
  exit 2
fi
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
# Run from OUTDIR, never from the caller.  cwd is first on sys.path for a `python -` step,
# so staying in a directory that happens to contain a `scale` package would shadow the
# IMPORT_ROOT this launcher means to import -- a silent switch of the code under test.
cd "$OUTDIR"

# Default only when UNSET: an explicitly empty MODEL_KEYS or CONTEXTS is a caller mistake
# and must reach the preflight, which refuses it, rather than silently becoming "all" or the
# default ladder -- a run that quietly measured a different grid than the caller asked for.
MODEL_KEYS="${MODEL_KEYS-all}"
CONTEXTS="${CONTEXTS-$DEFAULT_CONTEXTS}"
MODELS_ROOT="${MODELS_ROOT:-$DEFAULT_MODELS_ROOT}"
DEVICE="${DEVICE:-cuda:0}"
DRY_RUN="${DRY_RUN:-0}"

HOST="$(hostname)"
BOOTLOG="$OUTDIR/${HOST}.boot.txt"
RUNLOG="$OUTDIR/${HOST}.run.log"

{
  echo "==== BOOT(efficiency-probe:$MODE) $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  echo "host=$HOST NGPUS=$NGPUS DEVICE=$DEVICE DRY_RUN=$DRY_RUN"
  echo "OUTDIR=$OUTDIR MODEL_KEYS=$MODEL_KEYS CONTEXTS=$CONTEXTS"
  echo "MODELS_ROOT=$MODELS_ROOT IMPORT_ROOT=$IMPORT_ROOT"
  nvidia-smi -L 2>&1 || echo "<nvidia-smi unavailable>"

  echo "---- PREFLIGHT: torch + CUDA ----"
  # A dry run touches no device, so requiring a GPU to see the plan would defeat it -- the
  # same reason run_matrix_job --dry-run claims no node.  A real measurement still fails
  # closed here: without a CUDA device every peak reading would be meaningless.
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "PREFLIGHT OK: DRY_RUN=1 does not touch a device, so the CUDA check is skipped"
  else
    if torch_report="$("$PYTHON_BIN" -c \
      "import torch; print('torch', torch.__version__); raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>&1)"; then
      echo "PREFLIGHT OK: ${torch_report}; CUDA available"
    else
      echo "PREFLIGHT FAIL: torch cannot see a CUDA device (this is a GPU experiment)"
      printf '%s\n' "$torch_report"
    fi
  fi

  echo "---- PREFLIGHT: registry, requested model keys, weight paths ----"
  # The adapters module is torch-free at import, so a typo'd key or an un-downloaded
  # checkpoint -- the two failures a reader of the matrix must be able to tell apart -- is
  # reported even on a host where torch is unavailable.  build() stats the checkpoint path
  # and refuses by name (unknown key) or by path (weights absent), and the launcher only
  # relays that refusal, so the launcher and the registry cannot drift apart.
  "$PYTHON_BIN" - "$MODEL_KEYS" "$MODELS_ROOT" "$RUNNER_PACKAGE" <<'PYADAPTERS' 2>&1
import importlib
import sys

requested_arg, models_root, package = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    adapters = importlib.import_module(package + ".adapters")
except Exception as exc:  # noqa: BLE001 - reported as a named preflight failure
    print(f"PREFLIGHT FAIL: cannot import the adapters registry: {exc!r}")
    raise SystemExit(1)

declared = sorted(adapters.REGISTRY)
if not declared:
    print("PREFLIGHT FAIL: the adapters registry declares no model")
    raise SystemExit(1)
print(f"PREFLIGHT OK: adapters import; registry declares {len(declared)} keys: {', '.join(declared)}")

if requested_arg.strip() == "all":
    requested = declared
else:
    requested = [key.strip() for key in requested_arg.split(",") if key.strip()]
if not requested:
    print("PREFLIGHT FAIL: MODEL_KEYS named no model key")
    raise SystemExit(1)
print("RESOLVED_MODEL_KEYS=" + ",".join(requested))

for key in requested:
    try:
        adapter = adapters.build(key, models_root=models_root)
    except adapters.AdapterRefusal as exc:
        print(f"PREFLIGHT FAIL: {exc}")
        raise SystemExit(1)
    print(f"PREFLIGHT OK: model key {key!r} resolves to {adapter.resolved_path}")
PYADAPTERS

  echo "---- PREFLIGHT: CONTEXTS are positive, strictly ascending integers ----"
  # The runner refuses a non-monotone ladder, but it refuses it inside a cell's measurement;
  # catching it here means a typo costs no GPU time.  Strictly ascending (not merely sorted)
  # is required because the ladder stops at the first rung that does not fit, and a repeat
  # or a reversal would make that stop land on the wrong context.
  IFS=',' read -ra CONTEXT_LIST <<< "$CONTEXTS"
  context_prev=0
  context_ok=1
  if (( ${#CONTEXT_LIST[@]} == 0 )); then
    echo "PREFLIGHT FAIL: CONTEXTS named no context"
    context_ok=0
  fi
  for context in "${CONTEXT_LIST[@]}"; do
    if [[ ! "$context" =~ ^[0-9]+$ ]] || (( context <= 0 )); then
      echo "PREFLIGHT FAIL: CONTEXTS must be positive integers; got '${context}'"
      context_ok=0
      break
    fi
    if (( context <= context_prev )); then
      echo "PREFLIGHT FAIL: CONTEXTS must strictly ascend; ${context} follows ${context_prev}"
      context_ok=0
      break
    fi
    context_prev=$context
  done
  (( context_ok )) && echo "PREFLIGHT OK: ${#CONTEXT_LIST[@]} contexts parse and strictly ascend: $CONTEXTS"
  echo "==== END BOOT ===="
} > "$BOOTLOG" 2>&1
cat "$BOOTLOG"
if grep -q "PREFLIGHT FAIL" "$BOOTLOG"; then
  echo "[launch] FATAL: preflight failed; see $BOOTLOG"
  exit 2
fi

# The preflight resolved the requested keys (expanding "all"); the grid is built from that,
# not from the raw MODEL_KEYS, so the summary cannot count a key the registry never had.
MODEL_KEYS_RESOLVED="$(sed -n 's/^RESOLVED_MODEL_KEYS=//p' "$BOOTLOG" | tail -n 1)"
if [[ -z "$MODEL_KEYS_RESOLVED" ]]; then
  echo "[launch] FATAL: the preflight resolved no model key; see $BOOTLOG" >&2
  exit 2
fi
IFS=',' read -ra MODEL_KEY_LIST <<< "$MODEL_KEYS_RESOLVED"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[launch] DRY_RUN=1: ${#MODEL_KEY_LIST[@]} models x ${#CONTEXT_LIST[@]} contexts = $(( ${#MODEL_KEY_LIST[@]} * ${#CONTEXT_LIST[@]} )) cells planned"
  for key in "${MODEL_KEY_LIST[@]}"; do
    for context in "${CONTEXT_LIST[@]}"; do
      echo "  $OUTDIR/${key}_${context}.json"
    done
  done
  echo "SUMMARY: cells written=0 cells failed=0 path=$OUTDIR (dry run: no cell measured)"
  exit 0
fi

: > "$RUNLOG"
driver_status=0
for key in "${MODEL_KEY_LIST[@]}"; do
  # One driver per model: the checkpoint is loaded once and reused across contexts, because a
  # per-cell reload would multiply the multi-gigabyte load cost by the ladder length.  The
  # driver measures each cell with its own single-rung probe_ladder, so an OOM on one rung is
  # a row (the protocol) and never a reason to skip the larger contexts.  Only a nonzero exit
  # -- an import or load failure -- aborts the rest of the grid.
  "$PYTHON_BIN" - "$RUNNER_PACKAGE" "$key" "$OUTDIR" "$MODELS_ROOT" "$DEVICE" "${CONTEXT_LIST[@]}" <<'PYPROBE' 2>&1 | tee -a "$RUNLOG"
import importlib
import sys
from pathlib import Path

import torch

package, key, out_dir, models_root, device = sys.argv[1:6]
contexts = [int(value) for value in sys.argv[6:]]

try:
    adapters = importlib.import_module(package + ".adapters")
    probe = importlib.import_module(package + ".probe_inference")
except Exception as exc:  # noqa: BLE001 - an import failure is one of the two fatal cases
    print(f"IMPORT FAIL: {exc!r}")
    raise SystemExit(2)


class _ProbeArm:
    """Present a registry CheckpointAdapter as the probe's InferenceAdapter.

    probe_inference was written against an arm that carries ``analytic_weights_bytes`` and a
    context-INDEPENDENT ``analytic_state_bytes`` and exposes ``forward(context)`` as one
    full-context pass.  The registry's CheckpointAdapter carries ``spec.weights_bytes``, a
    context-TAKING ``analytic_state_bytes(context)`` method, and takes a token tensor.  The
    two describe the same measurement from opposite sides, so the difference is resolved here,
    once, instead of changing either proven interface.
    """

    def __init__(self, adapter, context):
        spec = adapter.spec
        self._adapter = adapter
        self._device = device
        self.model = spec.key
        self.family = spec.family
        self.objective = spec.objective
        self.nfe = spec.nfe
        self.analytic_weights_bytes = spec.weights_bytes
        self.analytic_state_bytes = adapter.analytic_state_bytes(context)

    def forward(self, context):
        # One full-context inference pass is what ``forward`` promises; the token content is
        # irrelevant to a memory/timing measurement, so a zeros batch of the right length is
        # what makes the pass the same shape for every arm.
        tokens = torch.zeros((1, context), dtype=torch.long, device=self._device)
        self._adapter.forward(tokens)


try:
    adapter = adapters.build(key, models_root=models_root)
    adapter.load(device)
except Exception as exc:  # noqa: BLE001 - build/load failure is the other fatal case
    print(f"LOAD FAIL: model key {key!r} could not be built and loaded on {device}: {exc!r}")
    raise SystemExit(3)

for context in contexts:
    arm = _ProbeArm(adapter, context)
    rows = probe.probe_ladder(arm, contexts=(context,), device=device, out_dir=out_dir)
    artifact = Path(out_dir) / probe.PROBE_FILENAME
    target = Path(out_dir) / f"{key}_{context}.json"
    artifact.replace(target)
    row = rows[0]
    print(f"CELL key={key} context={context} status={row['status']} file={target}")
PYPROBE
  driver_status="${PIPESTATUS[0]}"
  if (( driver_status != 0 )); then
    echo "[launch] FATAL: the probe driver for model key '${key}' exited ${driver_status}; aborting the grid"
    break
  fi
done

# An OOM row is a result, not an omission, so a cell whose row is "oom" is counted as failed
# and the run still ends with a summary -- the grid's scope is stated rather than inferred
# from which files happen to exist.
cells_written="$(grep -c 'status=ok' "$RUNLOG" 2>/dev/null || true)"
cells_failed="$(grep -c 'status=oom' "$RUNLOG" 2>/dev/null || true)"
echo "SUMMARY: cells written=${cells_written:-0} cells failed=${cells_failed:-0} path=$OUTDIR"
exit "$driver_status"
