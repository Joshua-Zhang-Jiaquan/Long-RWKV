"""Build and submit the Long-RWKV ICLR-2027 evaluation jobs on qz.

Every job here is **one H100** on the pinned 1-GPU shape.  That is not an
efficiency choice: the user's standing constraint for this campaign is fewer than
four H100s per job, and the cluster is CPU-bound (an 8-GPU ask needs 160 CPU and
queues on ``Insufficient cpu`` while the 1-GPU shape's 20 CPU clears in under two
minutes).  Sharding across many single-GPU jobs is therefore both the permitted
and the faster shape, and ``launch_capability_eval.sh`` already supports it: each
pod derives its worker index from ``NODE_RANK`` and strides through
``NUM_SHARDS``, so *k* independent 1-GPU jobs with ``NNODES=k`` and distinct
``NODE_RANK`` cover the benchmark exactly once.

Rules carried over from ``rwkv04b_v7/qz/emit_jobs.py``, each of which has cost
this project real time:

1. ``framework_config`` accepts exactly five keys; anything else is
   ``InvalidParameter`` and ``--dry-run`` does not catch it.
2. The command is validated with :mod:`shlex`, not ``bash -n`` -- an unbalanced
   quote inside ``bash -lc '...'`` passes ``bash -n`` and mangles in the pod.
3. shm, project and wall clock are derived from the spec, never passed in, so no
   body can pair a 1-GPU spec with the 8-GPU project.
4. ``$$`` is eaten by the pod launcher, so no body may contain it.
5. A submission is written to the ledger **before** ``CreateJob``, so a crash
   mid-call leaves a blocking ``attempted`` row rather than nothing.

Two additions specific to this campaign:

``NODE_RANK`` must be explicit, and must ride in the command
    The launcher derives ``NODE_RANK`` from *the trailing digits of the
    hostname* when ``NNODES>1`` (``launch_capability_eval.sh:235``).  That works
    for a multi-pod job whose pods are named ``...-worker-0-0``,
    ``...-worker-0-1``.  Here each shard is a *separate job*, so every pod is
    worker 0 of its own job and the hostname does not encode the shard -- every
    job would derive ``NODE_RANK=0`` and run shard 0, producing k copies of one
    shard, no coverage of the rest, and a merge that never fires because
    ``shard_count`` never reaches ``NUM_SHARDS``.  Nothing errors: the scores
    that come back are real scores of the wrong subset.

    So the rank is passed explicitly -- and in the **command's exports**, not in
    the payload's ``envs`` list.  Every job this project has ever submitted sent
    ``envs: []``, so that the scheduler delivers a non-empty ``envs`` to the pod
    is untested here, while the export channel is what the launcher's whole
    documented contract already travels through.  An undelivered ``envs`` would
    fail *silently and identically* to the hostname-derivation bug it was meant
    to fix, so the unverified channel is the wrong place for this value.
    :func:`check_shard_coverage` then reads the ranks back out of the submitted
    command strings -- the artifact that actually runs -- and proves the group
    tiles ``range(NUM_SHARDS)`` before anything is submitted.

The model must have a loader
    :mod:`lrwkv_evidence.tracks` marks ``sedd-medium`` and ``mamba2-2.7b``
    ``NO_LOADER`` (no ``model_type``, no dedicated loader, ``mamba_ssm`` absent
    from the image).  Building a body for either would burn queue time to reach a
    load failure, so :func:`body_for` refuses them locally.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lrwkv_evidence import tracks  # noqa: E402

IMAGE: Final = "docker.sii.shaipower.online/inspire-studio/relay2:v2"
IMAGE_TYPE: Final = "SOURCE_PRIVATE"
LCG: Final = "lcg-71b971a7-5bdd-4798-b5ba-08f1eabde49e"
WORKSPACE: Final = "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6"

#: The sanctioned pod shapes.  Each is a *quota id*: the scheduler derives shm_gi,
#: project and pricing from it, and a body that mixes one shape's spec_id with
#: another's shm is unrepresentable (hence :func:`make_body` reads all three from
#: this table rather than taking them as arguments).
#:
#: * 1 GPU (job-37a360d3): 20 CPU, 400 GiB shm, its own project, a 12 h wall.
#: * 2 GPU (job-dc02c9d6-era fallback): 2 cards on the shared project.
#: * 8 GPU (job-dc02c9d6): 160 CPU, 1800 GiB shm, the shared project, a **72 h**
#:   wall.  Measured by GetJob on the operator's pinned reference: ``quota_id``
#:   7166bd2e-6cbe-4bd9-be38-762d11003e7f, cpu_count 160, memory_size_gib 1800,
#:   gpu_info NVIDIA_H100_SXM_80G.
SPEC_1GPU: Final = "79fe954a-be92-4772-ac0b-94ad8a79b7bb"
SPEC_2GPU: Final = "26ef0d6e-330d-4650-a18a-7e1fbe8f3717"
SPEC_8GPU: Final = "7166bd2e-6cbe-4bd9-be38-762d11003e7f"
PROJECT_1GPU: Final = "project-632c8db8-4530-413a-ada5-df91774a7e09"
PROJECT_SHARED: Final = "project-160ccb20-98ab-4538-a847-01d1f83d5b0f"
TIMEOUT_1GPU_MS: Final = 43_200_000     # 12 h
TIMEOUT_8GPU_MS: Final = 259_200_000    # 72 h

#: ``spec_id -> (gpus_per_pod, shm_gi, project_id, wall_ms)``.  Every other module
#: asks this table; nothing re-states a shm or a project beside a spec id.
SHAPE: Final = {
    SPEC_1GPU: (1, 400, PROJECT_1GPU, TIMEOUT_1GPU_MS),
    SPEC_2GPU: (2, 800, PROJECT_SHARED, TIMEOUT_1GPU_MS),
    SPEC_8GPU: (8, 1800, PROJECT_SHARED, TIMEOUT_8GPU_MS),
}

#: The per-job GPU ceiling.  Two operator statements, in order:
#:
#: * 2026-09-20, the standing constraint this constant was written for: "fewer than
#:   4 H100 per job" -- hence a value of 3.
#: * **2026-09-21, superseding it: "use 8h100 for each benchmark / for each job".**
#:
#: Left at 4 the constant would refuse the very shape the operator just asked for,
#: and its error text would cite a constraint they have replaced.  That is the
#: [[a-stale-constant-can-disable-the-rule-it-guards]] failure mode: a guard whose
#: cited justification is stale is worse than no guard, because it reads as authority.
MAX_GPUS_PER_JOB: Final = 8

#: What the scheduler enforces underneath the allowance, read off a live CreateJob
#: refusal 2026-09-20: 16 GPU may RUN and 40 GPU may be SUBMITTED *per project*.  A
#: single 8-GPU job therefore consumes half of a project's running quota and a fifth
#: of its submitted quota, so the 8-GPU lane is two concurrent jobs deep per project
#: no matter how large the operator's allowance is.
PROJECT_RUNNING_GPU_QUOTA: Final = 16
PROJECT_SUBMITTED_GPU_QUOTA: Final = 40

#: Concurrent model loads inside one 8-GPU pod.
#:
#: ``eval/capability/load_gate.py`` serializes loads at 3 by default, and its own
#: comment says to raise that on a big-RAM pod -- naming the H200's 1.93 TB cgroup --
#: because "load is ~145 s and dwarfs the ~48 s of actual scoring, so serializing it is
#: the main throughput cost".  Its *justification* for the default is a **300 GB** pod.
#:
#: Measured in-pod 2026-09-21 (the ``---- host resources ----`` block the launcher now
#: emits): ``cgroup_memory_max = 1932735283200`` = **1.93 TB**, ``MemTotal =
#: 2113174472 kB``, ``nproc = 192``.  8 concurrent loads at the documented ~32.8 GB peak
#: is 262 GB -- **13.5 %** of the cgroup, the same headroom the 1.93 TB case is
#: described as having, on the very shape this campaign bills.
#:
#: What leaving it at 3 costs: workers logged ``load slot 0/3 acquired (waited 936s)``
#: through ``(waited 952s)`` -- ~15.7 min of *idle cards* per pod against a load of
#: ~145 s.  Across the 236 pods still queued that is roughly 250 GPU-hours spent waiting
#: on a gate whose stated precondition this pod does not meet.
#:
#: Deliberately does **not** also pin ``OMP_NUM_THREADS``.  The gate's own account names
#: thread oversubscription as the co-factor in the 2026-08-15 wedge (nproc is 192 here,
#: and 8 workers x the default would oversubscribe), but thread count can change
#: floating-point reduction order: a change whose numerics are not provably neutral, so
#: it waits until it can be measured instead of assumed.  Raising the *slot count* only
#: changes how many processes are inside the load at once; each worker's computation is
#: bit-identical either way.
WIDE_LOAD_SLOTS: Final = 8

#: The shapes ``WIDE_LOAD_SLOTS`` applies to.  Only the 8-GPU shape: the 2-GPU shape
#: bills the same project but is a different quota row with its own (unmeasured) cgroup,
#: so it keeps the gate's default.
WIDE_SHAPES: Final = frozenset({SPEC_8GPU})

#: ``BIRWKV_LOAD_SLOTS`` for a body at this shape, or an empty dict at the narrow shapes.
def load_slots_env(spec_id: str) -> dict[str, str]:
    return {"BIRWKV_LOAD_SLOTS": str(WIDE_LOAD_SLOTS)} if spec_id in WIDE_SHAPES else {}


ACCEPTED_FRAMEWORK_KEYS: Final = frozenset(
    {"image", "image_type", "instance_count", "shm_gi", "spec_id"})

G: Final = "/inspire/hdd/global_user/zhangjiaquan-253108540222"
SCALE_DIR: Final = f"{G}/qz_stage_traj4096_v7/scale"
LAUNCHER: Final = f"{SCALE_DIR}/qz/launch_capability_eval.sh"
OUTPUTS: Final = f"{G}/outputs_long_rwkv_iclr2027"
LEDGER: Final = Path(OUTPUTS) / "campaign_ledger.jsonl"

MMLU_TASKS: Final = f"{G}/capability_eval_data/tasks_mmlu_onetoken"

#: Basic-ability four-choice trees **beyond** MMLU, from the DiffRwkv preprocessing.
#: ``dataset -> (path relative to G, declared items, choices per item)``.
#:
#: These are RWKV-world tokenized (every id below 65536), which is why they are usable
#: only for a track whose tokenizer tree is ``rwkv_world`` -- see
#: :func:`check_aux_tree`.  They are a *different protocol* from the MMLU one-token
#: trees: the prompt template comes from ``preprocess_multichoice.py`` and the score is
#: the length-normalized NLL of the span from ``choice_start`` to the end, not the NLL
#: of a single label token.  Each dataset is therefore reported as its own table and
#: never pooled with MMLU.
#:
#: ``winogrande`` (1267) and ``arc_challenge``/``arc_easy`` are deliberately **absent**.
#: Measured on disk 2026-09-21: every winogrande row has ``choice_start == 0`` and about
#: a fifth of the ARC rows do.  ``_span_nll`` clamps ``choice_start <= 0`` to 1, so those
#: rows are scored over the whole sequence while their siblings are scored over the
#: choice span -- one dataset, two statistics, and the mean renders as a single cell.
#: They need a builder fix (a real ``choice_start``) before they can be scheduled.
AUX_TREES: Final = {
    "hellaswag": ("research/DiffRwkv/preprocessed_data/hellaswag/validation", 10042, 4),
    "race": ("research/DiffRwkv/preprocessed_data/race/validation", 4887, 4),
    "siqa": ("research/DiffRwkv/preprocessed_data/siqa/validation", 1954, 3),
    "story_cloze": ("research/DiffRwkv/preprocessed_data/story_cloze/validation",
                    1871, 2),
    "piqa": ("research/DiffRwkv/preprocessed_data/piqa/validation", 1838, 2),
    "commonsenseqa": ("research/DiffRwkv/preprocessed_data/commonsenseqa/validation",
                      1221, 5),
    "sciq": ("research/DiffRwkv/preprocessed_data/sciq/validation", 1000, 4),
    "openbookqa": ("research/DiffRwkv/preprocessed_data/openbookqa/validation", 500, 4),
    "gpqa_diamond": ("research/DiffRwkv/preprocessed_data/gpqa_diamond/test", 198, 4),
    "mmlu_redux": ("research/DiffRwkv/preprocessed_data/mmlu_redux/test", 5700, 4),
}

#: Dataset fill order, **paper-named panels first**.
#:
#: The protocol names MMLU, OpenBookQA and RACE explicitly with "exact split
#: definitions", and reading comprehension generally; the rest of the panel is
#: supporting breadth. Ordering by that claim rather than by size means the panels the
#: manuscript actually cites are scored early instead of last: OpenBookQA is 500 items
#: and was previously ninth, so moving it third buys a *named* panel for an eighth of
#: what RACE costs. Reordering cannot duplicate or lose work -- it only decides which
#: still-unsubmitted names are offered first, and every name is de-duplicated by the
#: ledger regardless of order.
AUX_ORDER: Final = ("hellaswag", "race", "openbookqa", "mmlu_redux", "siqa",
                    "story_cloze", "piqa", "commonsenseqa", "sciq", "gpqa_diamond")

#: The tracks the **drip** fills the auxiliary panels with.
#:
#: Operator direction 2026-09-21: "can we reduce the size of testing and get the results
#: sooner".  The lever taken is *model coverage*, not benchmark size, and that choice is
#: deliberate.  Shrinking a panel -- scoring 2,000 of HellaSwag's 10,042 items -- changes
#: the statistic and produces a number that renders exactly like the benchmark's; this
#: study already lost one reading to that (a 1,000-item `--max_samples` prefix read 0.573
#: where the full set reads 0.710).  Dropping *models* from a panel leaves every cell that
#: is reported measured over the same full item set.
#:
#: Why it is worth it here: measured 2026-09-21 on RACE, the same 4,887 items at eight
#: shards take **44 items/min** through the BiRWKV diffusion forward and **1,057
#: items/min** through a plain HF causal forward of the same parameter count -- 24x.  So
#: the wall clock of the remaining nine datasets is dominated by the diffusion arms, and
#: the released comparators are nearly free.  Keeping the comparators and the paper's own
#: rows, and dropping the redundant ladder steps and two of the three seed replicates,
#: cuts the remaining queue by roughly half without touching one item count.
#:
#: The full 25-track sweep is **not** lost: `--group e1-basic` with no `--track` still
#: builds it, and :func:`aux_track_ids` still returns it.  This tuple is the drip's
#: subset, and the two are separate so restoring breadth is editing one tuple.
AUX_TRACKS_DRIP: Final = (
    "a29_c6_f2_s14000",      # F2, the study's object
    "rel_c1_rwkv7_2p9b",     # R0, same-size causal
    "rel_c1_rwkv7_1p5b",     # smaller causal
    "rel_c1_rwkv7_0p4b",     # smallest causal, and the 0.4B matrix's own initialization
    "a29_c6loop_s9500",      # best full-MMLU looped arm
    "a29_c6loop_s20500",     # the arm earlier drafts lead with
    "a04_a1_s17",            # 0.4B factorial, seed 17: forward-only, no loop
    "a04_a2_s17",            #   bidirectional, no loop
    "a04_a3_s17",            #   bidirectional + tied loop
    "a04_a5_s17",            #   forward-only + tied loop
)

#: Tracks scored first within every dataset, in the order the manuscript needs them.
#:
#: The study's object is F2, and its comparison set is R0/R0-smaller/M0, so a dataset
#: that has those rows and nothing else is *usable* while a dataset with twenty other
#: arms and none of them is not. Dataset-major order alone would have put the released
#: comparators last in every dataset -- that is why R0's HellaSwag row only began after
#: twenty-two other tracks had finished. Everything not listed follows in registry
#: order, so this is a reordering and never a filter.
AUX_TRACK_PRIORITY: Final = (
    "a29_c6_f2_s14000",      # F2 itself, the evaluated checkpoint
    "rel_c1_rwkv7_2p9b",     # R0, the same-size causal comparator
    "rel_c7ar_mamba_2p8b",   # M0 as scored (Mamba-1 2.8B)
    "a29_c6loop_s20500",     # the looped arm the earlier drafts lead with
    "a29_c6loop_s9500",      # best full-MMLU arm in the ladder
    "rel_c1_rwkv7_1p5b",     # smaller causal, same family
    "rel_c1_rwkv7_0p4b",     # smallest causal, same family
    "a04_a1_s17", "a04_a2_s17", "a04_a3_s17", "a04_a5_s17",   # 0.4B factorial, seed 17
)


@lru_cache(maxsize=None)
def check_aux_tree(dataset: str) -> dict:
    """Refuse an auxiliary tree that is absent, short, or mixes two statistics.

    Three claims, each read off the ``.npz`` files rather than declared:

    * the item count is the one the benchmark is cited at -- a mean over a prefix is a
      different statistic that renders as the same cell
      ([[a-mean-over-a-subset-is-a-different-statistic]]);
    * every row's ``choice_start`` is > 0, because ``_span_nll`` clamps 0 to 1 and a
      clamped row is scored over the whole sequence instead of the choice span;
    * every ``label`` indexes a real choice row.

    ``--max_samples`` is never passed by this lane for the same reason as the first
    point: ``run_eval`` truncates with ``tasks[:max_samples]`` over the *sorted file
    list*, so a limit is a prefix of the file order, and this benchmark's files are
    task-grouped ([[a-benchmarks-file-order-is-part-of-its-contract]]).  The n=1,000
    HellaSwag reading that already exists is exactly such a prefix.
    """
    import numpy as np    # local: only this check needs it, and pods always have it

    if dataset not in AUX_TREES:
        raise SystemExit(f"unknown auxiliary dataset {dataset!r}; "
                         f"known: {sorted(AUX_TREES)}")
    rel, items, choices = AUX_TREES[dataset]
    root = Path(G) / rel
    files = sorted(root.glob("*.npz"))
    if not files:
        raise SystemExit(f"{dataset}: no .npz under {root}")
    if len(files) != items:
        raise SystemExit(
            f"{dataset}: {root} holds {len(files)} items but the benchmark is cited "
            f"at {items}. A different item count is a different statistic.")
    bad_start, bad_label, rows = 0, 0, 0
    for path in files:
        data = np.load(path)
        mask = data["attention_mask"]
        for j, start in enumerate(np.asarray(data["choice_start"]).ravel()):
            rows += 1
            if int(start) <= 0:
                bad_start += 1
            if not 0 <= int(np.asarray(data["label"]).ravel()[0]) < mask.shape[0]:
                bad_label += 1
    if bad_start or bad_label:
        raise SystemExit(
            f"{dataset}: {bad_start} of {rows} rows have choice_start <= 0 and "
            f"{bad_label} have an out-of-range label. A clamped row is scored over the "
            f"whole sequence while its siblings are scored over the choice span, so the "
            f"dataset would report one number over two statistics. Rebuild the tree "
            f"with a real choice_start.")
    return {"dataset": dataset, "root": str(root), "items": len(files),
            "choices": choices, "rows": rows}


def schedulable_tracks() -> list[str]:
    """Tracks ``run_eval.py`` can actually build, in registry order.

    Lives here rather than in the campaign because both callers need it and the
    campaign imports this module, so the reverse import would be circular.  Two
    predicates, both necessary: the model kind must have a loader in this harness
    (``UNSCHEDULABLE_KINDS`` covers NO_LOADER, FOREIGN_ARCH, LLADA and the missing
    runtime deps), *and* the tokenizer dir must have a registered one-token MMLU tree,
    because a track without one cannot be scored on MMLU at all.
    """
    return [t.track_id for t in tracks.ALL_TRACKS
            if t.model_kind not in tracks.UNSCHEDULABLE_KINDS
            and t.model_dir in TOKENIZER_TREE]


def aux_track_ids() -> list[str]:
    """The tracks an RWKV-world aux tree may be scored against, **paper order first**.

    A tree carries token *ids*, so it is only meaningful to a model whose tokenizer
    produced them.  MMLU has one tree per tokenizer for exactly this reason (the
    one-token trees differ in length as well as in the label id).  These trees are
    ``rwkv_world``, so the released Qwen/Llama/Mamba rows -- which the manifest lists
    with their own trees -- cannot be scored here without building theirs, and a
    silent cross-tokenizer score would look entirely normal.

    Sorted so the study's own rows lead (:data:`AUX_TRACK_PRIORITY`), because the fill
    order decides which panels become *usable* first.  A priority name that is not in
    the schedulable set is dropped rather than raising: the set can legitimately lose a
    track (a loader regression), and losing the *ordering preference* for it is not a
    reason to refuse the whole lane.
    """
    ids = [tid for tid in schedulable_tracks()
           if TOKENIZER_TREE.get(track_by_id(tid).model_dir) == "rwkv_world"]
    rank = {tid: i for i, tid in enumerate(AUX_TRACK_PRIORITY)}
    return sorted(ids, key=lambda t: (rank.get(t, len(rank)), ids.index(t)))

#: Which one-token MMLU task tree a track's tokenizer needs.  The label ids differ
#: per tokenizer (RWKV 300-303, Qwen 362/425/356/422, Llama 362/426/356/423, LLaDA
#: 355/413/348/435), so a track scored against the wrong tree would be scoring
#: four arbitrary tokens.  Keyed by the track's ``model_dir`` because that is what
#: supplies the tokenizer at eval time.
TOKENIZER_TREE: Final = {
    "models/RWKV7-Goose-World3-2.9B-HF": "rwkv_world",
    "models/RWKV7-Goose-World3-1.5B-HF": "rwkv_world",
    "models/rwkv7-0.4B": "rwkv_world",
    "models/rwkv7-0.4B-world": "rwkv_world",
    "models/Qwen2.5-3B": "qwen25",
    "models/Llama-3.2-3B": "llama32",
    "models/LLaDA-8B-Base": "llada",
    # State/linear-attention baselines (2026-09-21).  Each tree was built by
    # lrwkv_evidence.e1.mmlu_labels against that model's own tokenizer, because
    # the four label ids differ per vocab: mamba 329/378/330/399 (vocab 50280),
    # rwkv6 66/67/68/69 (vocab 65530).  Both are registered even though rwkv6 is
    # MISSING_RUNTIME_DEP: the tree is valid and the row becomes schedulable the
    # moment a loader exists, and an absent entry here would look like an
    # oversight rather than a decision.
    "models/mamba-2.8b": "mamba",
    "models/rwkv6-world-3b": "rwkv6",
    "hf_cache/hub/models--fla-hub--gla-1.3B-100B": "gla",
}


def validate_command(command: str) -> None:
    """Fail locally on a command the pod would mangle."""
    if not command.startswith("bash -lc '"):
        raise SystemExit(f"command must be a `bash -lc '...'` body, got {command[:40]!r}")
    inner = command[len("bash -lc '"):]
    if not inner.endswith("'"):
        raise SystemExit("the single-quoted body is not closed")
    inner = inner[:-1]
    if "'" in inner:
        raise SystemExit("the body contains a bare single quote, which closes it early")
    if "$$" in inner:
        raise SystemExit(
            "the body contains $$, which the pod launcher collapses to a bare $ "
            "before bash sees it; use $BASHPID")
    try:
        shlex.split(inner)
    except ValueError as exc:
        raise SystemExit(f"shlex cannot parse the body: {exc}")


def wrapped(exports: dict[str, str], launcher: str) -> str:
    parts = ["set -e"]
    for key, value in exports.items():
        if "'" in str(value):
            raise SystemExit(f"export {key} contains a single quote: {value!r}")
        parts.append(f'export {key}="{value}"')
    parts.append(f"bash {launcher}")
    return "bash -lc '" + "; ".join(parts) + "'"


def make_body(name: str, command: str, description: str, spec_id: str = SPEC_1GPU,
              instance_count: int = 1, envs: list[dict] | None = None,
              task_priority: int = 4) -> dict:
    shape = SHAPE.get(spec_id)
    if shape is None:
        raise SystemExit(
            f"spec {spec_id} is not one of the {len(SHAPE)} sanctioned shapes "
            f"{sorted(SHAPE)}; shm_gi, project_id and the wall clock are all derived "
            f"from the spec, so a spec outside this table cannot be priced")
    gpus_per_pod, shm_gi, project_id, wall_ms = shape
    total_gpus = gpus_per_pod * instance_count
    if total_gpus > MAX_GPUS_PER_JOB:
        raise SystemExit(
            f"{name}: {instance_count} pods x {gpus_per_pod} GPU = {total_gpus} "
            f"GPUs, over the standing limit of {MAX_GPUS_PER_JOB}")
    framework_config = [{
        "image": IMAGE, "image_type": IMAGE_TYPE,
        "instance_count": instance_count, "shm_gi": shm_gi, "spec_id": spec_id,
    }]
    extra = set(framework_config[0]) - ACCEPTED_FRAMEWORK_KEYS
    if extra:
        raise SystemExit(f"framework_config has keys the API rejects: {sorted(extra)}")
    validate_command(command)
    return {
        "name": name,
        "framework": "pytorch",
        "command": command,
        "framework_config": framework_config,
        "logic_compute_group_id": LCG,
        "project_id": project_id,
        "workspace_id": WORKSPACE,
        "task_priority": task_priority,
        "max_running_time_ms": str(wall_ms),
        "auto_fault_tolerance": True,
        "fault_tolerance_max_retry": 3,
        "fault_tolerance_retry_interval_sec": 300,
        "description": description,
        "envs": envs or [],
    }


def track_by_id(track_id: str) -> tracks.Track:
    for t in tracks.ALL_TRACKS:
        if t.track_id == track_id:
            return t
    raise SystemExit(f"unknown track {track_id!r}; see lrwkv_evidence/tracks.py")


def mmlu_outdir(track_id: str, split: str = "test", tag: str = "e1") -> str:
    return f"{OUTPUTS}/{tag}_mmlu_{split}_{track_id}"


#: The tag the wide MMLU groups are opened under (2026-09-21, "use 8h100 for each
#: benchmark").  A *new* tag rather than a re-shape of ``e1``: 27 e1 groups already
#: carry live rows at 1 or 2 GPU/pod, :func:`pod_topology`'s shape lock correctly
#: refuses to re-shape them, and the ledger's own dedup is keyed on the job name -- so
#: a re-tag is the only way to move a benchmark to the wider pod that does not either
#: overwrite a group's provenance or bill the same shard twice under two names.
TAG_8GPU: Final = "e1b"


def merged_present(track_id: str, split: str = "test", tag: str = "e1") -> bool:
    """Is there a merged MMLU aggregate for this track at this tag?

    A *merged* file, not a shard: the merge gate only fires once every one of
    ``NUM_SHARDS`` shard files exists, so its presence is the campaign's own receipt
    that the benchmark was scored end to end.  Counting shards instead would call a
    ten-of-twelve partial a finished reading -- and the two render as the same cell.

    This is what lets a re-tag (:data:`TAG_8GPU`) cover exactly the tracks that do not
    have a reading yet, instead of re-running 27 tracks to obtain 6.
    """
    merged = Path(mmlu_outdir(track_id, split, tag)) / "merged"
    return merged.is_dir() and any(merged.glob("*.json"))


def mmlu_task_dir(track: tracks.Track, split: str) -> str:
    tree = TOKENIZER_TREE.get(track.model_dir)
    if tree is None:
        raise SystemExit(
            f"{track.track_id}: no one-token MMLU tree is registered for tokenizer "
            f"dir {track.model_dir!r}. Build one with lrwkv_evidence.e1.mmlu_labels "
            f"and add it to TOKENIZER_TREE -- scoring against another tokenizer's "
            f"tree would score four arbitrary token ids.")
    return f"{MMLU_TASKS}/{tree}/{split}"


#: Where :mod:`lrwkv_evidence.convert_rwkv04b` writes eval-only step dirs for the
#: rwkv04b bundles.  Kept in sync with that module by :func:`eval_ckpt_dir`'s
#: meta.json check rather than by convention.
DERIVED_RWKV04B: Final = f"{G}/derived_ckpts_rwkv04b"


def eval_ckpt_dir(track: tracks.Track) -> str:
    """What ``--ckpt_dir`` must receive for this track.

    Usually the registry's own path.  For the five rwkv04b bundles it is the
    derived step dir, because the bundle is a *training* checkpoint (weights
    nested under ``model``, plus optimizer state and a ``TorchVersion`` object)
    and the eval loader reads a flat ``<dir>/model.pt``.  The registry keeps
    pointing at the bundle on purpose: the bundle's sha256 is the provenance a
    paper number should cite, and the derived dir records that sha in its
    ``meta.json``, so the citation survives the conversion.

    The derived dir is verified to actually descend from *this* track before it is
    used.  A dir named for one track but extracted from another would silently
    swap two checkpoints that differ only in seed -- the arms would still load,
    still score, and report the wrong seed's numbers.
    """
    if (track.path / "model.pt").is_file():
        return track.ckpt_arg
    if not (track.path.is_file() and track.path.suffix == ".pt"):
        return track.ckpt_arg
    derived = Path(DERIVED_RWKV04B) / track.track_id / f"step_{track.step:08d}"
    meta_path = derived / "meta.json"
    if not (derived / "model.pt").is_file() or not meta_path.is_file():
        return track.ckpt_arg    # check_loadable_layout raises with the fix named
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("derived_from") != str(track.path):
        raise SystemExit(
            f"{track.track_id}: {derived} was extracted from "
            f"{meta.get('derived_from')!r}, not from {track.path}. Refusing to "
            f"evaluate one checkpoint under another's identity -- these runs "
            f"differ only by seed, so the scores would look entirely normal.")
    if track.sha256 and meta.get("derived_from_sha256") != track.sha256:
        raise SystemExit(
            f"{track.track_id}: {derived} records source sha "
            f"{meta.get('derived_from_sha256')!r} but the registry pins "
            f"{track.sha256}. The bundle changed after conversion, or the dir is "
            f"stale; re-run convert_rwkv04b --force.")
    return str(derived)


def check_loadable_layout(track: tracks.Track) -> None:
    """Refuse a checkpoint whose on-disk layout the loader cannot consume.

    ``model_kind`` says which loader to call; it does not say the file is shaped
    the way that loader reads.  Two distinct layouts exist in this registry and
    only one is loadable as-is:

    * ``outputs_birwkv_diffusion``/``outputs_task7_matrix`` -- a ``step_*/``
      **directory** holding a flat ``model.pt``.  ``load_birwkv_diffusion`` does
      ``torch.load(Path(ckpt_dir) / "model.pt", weights_only=True)``, so this is
      the layout it wants.  22 tracks, all verified present.
    * ``outputs_rwkv04b/runs/*/step_*.pt`` -- a **file** whose payload is
      ``{"model", "optimizer", "scaler", "step", "record"}``, i.e. the weights are
      nested one level down and the file also carries a ``TorchVersion`` object
      that ``weights_only=True`` refuses outright.

    Pointing ``--ckpt_dir`` at the second kind asks the loader for
    ``<file>.pt/model.pt``, which cannot exist.  The job queues, pulls the image,
    builds 0.4B of geometry and only then dies -- about a card-hour per shard to
    learn something one ``stat`` answers.  So it is refused here, and the message
    names the conversion rather than implying the checkpoint is unusable: the
    weights are fine, the container is not.
    """
    if track.model_kind != "birwkv_diffusion":
        # hf_causal takes its weights from --model_dir and treats --ckpt_dir as
        # provenance only (run_eval.py:250), so no layout requirement applies.
        return
    if (track.path / "model.pt").is_file():
        return
    if track.path.is_file() and track.path.suffix == ".pt":
        derived = Path(DERIVED_RWKV04B) / track.track_id / f"step_{track.step:08d}"
        if (derived / "model.pt").is_file() and (derived / "meta.json").is_file():
            return    # eval_ckpt_dir routes here and verifies the lineage
        raise SystemExit(
            f"{track.track_id}: {track.rel_path} is a bundled checkpoint file "
            f"(keys model/optimizer/scaler/step/record), but "
            f"load_birwkv_diffusion reads <ckpt_dir>/model.pt as a FLAT state "
            f"dict. This body would ask for {track.path}/model.pt, which cannot "
            f"exist: the pod would spend its load time to reach a missing path. "
            f"Run `python3 -m lrwkv_evidence.convert_rwkv04b --track "
            f"{track.track_id}` to write a loadable step dir first -- the "
            f"weights are usable, only the container shape is wrong.")
    raise SystemExit(
        f"{track.track_id}: no model.pt under {track.path} and it is not a .pt "
        f"bundle either; the registry and the disk disagree about this entry.")


#: The causal span-scoring condition for each ``model_kind``.  These are two
#: *names* for one statistic, and which name works is a property of the model
#: class, not a preference:
#:
#: * ``fwdce`` calls ``model(ids, True)`` (multichoice.py:211), where the second
#:   positional argument is ``BiRWKV7ForMaskedDiffusion.forward``'s
#:   ``force_forward`` flag -- it selects the forward stream so a denoiser scores
#:   as the pretrained causal RWKV-7.
#: * ``raw`` calls ``backbone(input_ids=ids, attention_mask=mask.bool(), ...)``
#:   (multichoice.py:198) and explicitly falls back to the model itself when
#:   there is no ``.rwkv_model``, which is the ``hf_causal`` path.
#:
#: Both then go through the same ``_span_nll`` with the same shift, so the arms
#: are the same protocol under different entry points.  Handing ``fwdce`` to a
#: plain HF model binds ``True`` to ``attention_mask`` positionally; measured on
#: five released-model probes (2026-09-20), which died as
#: ``AttributeError: 'bool' object has no attribute 'shape'`` inside fla /
#: transformers after the load had already been paid for.
CAUSAL_SPAN_CONDITION: Final = {
    "birwkv_diffusion": "fwdce",
    "hf_causal": "raw",
    # A fla-registered causal LM is still a plain HF causal forward once its
    # architecture is registered, so it takes the same arm as hf_causal -- not
    # 'fwdce', which binds a BiRWKV denoiser's force_forward flag positionally and
    # would be handed to `attention_mask` here.
    "fla_causal": "raw",
}


def causal_span_condition(track: tracks.Track) -> str:
    """The ``CONDITIONS`` value that scores ``track`` with causal span NLL.

    Refuses rather than guessing: a wrong name here is not a crash at submit
    time, it is a crash after the model loads on a card, and for ``raw`` vs
    ``maskce`` it would not crash at all -- it would return a real number for a
    different statistic.
    """
    cond = CAUSAL_SPAN_CONDITION.get(track.model_kind)
    if cond is None:
        raise SystemExit(
            f"{track.track_id}: no causal span-scoring condition is known for "
            f"model_kind {track.model_kind!r}. CONDITIONS is not a free string: "
            f"'fwdce' reaches BiRWKV7ForMaskedDiffusion.forward's force_forward "
            f"flag and 'raw' reaches a plain HF forward, so the wrong one either "
            f"crashes on the card or silently scores a different statistic.")
    return cond


#: The two sanctioned pod shapes, keyed by GPUs per pod.  They route to *different
#: projects* (see :data:`PROJECT_1GPU` / :data:`PROJECT_SHARED`), which is the whole
#: reason the 2-GPU shape is worth having: the 16-running/40-submitted quota is
#: enforced per project, so the second shape is additional capacity rather than a
#: rearrangement of the first.  Operator order 2026-09-20: fill the 1-GPU project
#: first, then overflow into the 2-GPU one.
#: Which sanctioned shape serves a requested GPUs-per-pod.  There is deliberately no
#: 3, 5, 6 or 7: the scheduler grants a pod a whole quota shape (8 cards for the
#: 8-GPU quota), so asking for five cards and pinning ``CUDA_VISIBLE_DEVICES`` to
#: five books eight and computes on five.  ``4 -> SPEC_8GPU`` *is* such a case, and
#: it is here only because a suite with four shards (HumanEval+ at 164 items) has no
#: use for eight workers; the job still books eight cards and leaves four idle, which
#: is a cost the caller must mean rather than discover.
SPEC_BY_GPUS: Final = {1: SPEC_1GPU, 2: SPEC_2GPU, 4: SPEC_8GPU, 8: SPEC_8GPU}


def pod_topology(pod_rank: int, num_shards: int,
                 gpus_per_pod: int) -> tuple[dict, str, list[int]]:
    """The launcher-facing shape of one pod: its exports, name suffix and shards.

    The launcher gives worker ``w = NODE_RANK*NGPUS + g`` the shards
    ``w, w+W, ...`` with ``W = NNODES*NGPUS``
    (``launch_capability_eval.sh:229-256``), and it runs the ``NGPUS`` workers
    *concurrently*, each pinned with ``export CUDA_VISIBLE_DEVICES=$g``.  So at
    ``NNODES = num_shards/NGPUS`` the stride equals ``num_shards``, every worker
    draws exactly one shard, and a 2-GPU pod finishes two shards in one shard's
    wall clock.  That is what makes the 2-GPU shape real parallelism and not just
    a bigger pod.

    ``num_shards`` keeps meaning *shards*, never pods.  Conflating them is how a
    group silently halves its coverage: ``NUM_SHARDS`` is also the count the merge
    gate waits for, so a group that shipped ``num_shards/2`` would produce real
    numbers over half the benchmark and never merge.

    The 1-GPU name suffix is left exactly as it was (``s<k>of<n>``).  It is the
    ledger key for the 27 MMLU shards already in flight, and changing it would make
    every one of them look unsubmitted.
    """
    if gpus_per_pod not in SPEC_BY_GPUS:
        raise SystemExit(
            f"gpus_per_pod={gpus_per_pod} is not one of the sanctioned pod shapes "
            f"{sorted(SPEC_BY_GPUS)}; the standing constraint is at most "
            f"{MAX_GPUS_PER_JOB} H100 per job")
    if num_shards % gpus_per_pod:
        raise SystemExit(
            f"{num_shards} shards do not divide across {gpus_per_pod} GPUs per pod. "
            f"The launcher's stride is NNODES*NGPUS, so an uneven split either "
            f"leaves the tail shards unscored or scores some twice, and in both "
            f"cases the merge gate never sees NUM_SHARDS files.")
    num_pods = num_shards // gpus_per_pod
    if not 0 <= pod_rank < num_pods:
        raise SystemExit(
            f"pod rank {pod_rank} outside range(0,{num_pods}) for {num_shards} "
            f"shards at {gpus_per_pod} GPU/pod")
    shards = [pod_rank * gpus_per_pod + g for g in range(gpus_per_pod)]
    exports = {
        "NGPUS": str(gpus_per_pod),
        "NUM_SHARDS": str(num_shards),
        # NNODES>1 is what makes the launcher honour an inherited NODE_RANK at all
        # (at NNODES=1 it hard-assigns 0), and WORLD_GPUS=NNODES*NGPUS is the
        # stride each worker walks.
        "NNODES": str(num_pods),
        "NODE_RANK": str(pod_rank),
    }
    suffix = (f"s{pod_rank}of{num_shards}" if gpus_per_pod == 1
              else f"p{pod_rank}of{num_pods}x{gpus_per_pod}")
    return exports, suffix, shards


def mmlu_body(track_id: str, shard: int, num_shards: int, split: str = "test",
              conditions: str | None = None, tag: str = "e1",
              gpus_per_pod: int = 1) -> dict:
    """One MMLU pod for one checkpoint: ``gpus_per_pod`` shards, run concurrently.

    ``shard`` is the **pod rank**, which equals the shard index only at
    ``gpus_per_pod=1``; :func:`pod_topology` maps it to the shards the pod will
    actually draw.  ``num_shards`` always counts shards.

    ``conditions=None`` (the default) resolves per model kind via
    :func:`causal_span_condition`: ``fwdce`` for BiRWKV denoisers, ``raw`` for
    released HF causal LMs.  Both are the causal (shifted) span scoring that is
    protocol-faithful for the one-token label convention, so the arms remain
    comparable through the same ``_span_nll``.  ``maskce`` is the denoiser-native
    variant and a *different* statistic; it may be run as a declared secondary
    condition but never as a substitute.
    """
    track = track_by_id(track_id)
    if track.model_kind == tracks.NO_LOADER:
        raise SystemExit(
            f"{track_id} has no loader in this harness ({track.notes}); a job for "
            f"it would queue, load and fail. Report the baseline as absent.")
    if track.model_kind == tracks.FOREIGN_ARCH:
        raise SystemExit(
            f"{track_id} is a TiedBiRWKV7Block checkpoint: it stores one mixer per "
            f"block as layers.N.attn.*, while load_birwkv_diffusion builds untied "
            f"attn_fwd/attn_bwd and raises 'checkpoint/model mismatch' on the "
            f"missing keys. Measured 2026-09-20: five probes each paid a full 0.4B "
            f"load before dying this way. It is evaluable only through "
            f"rwkv04b/longrwkv/eval (LongRWKV.from_hf_pretrained with arm=), which "
            f"has no multichoice path -- so this lineage has no E1 lane at all and "
            f"joins the campaign at E3 through its own runner.")
    if track.model_kind == tracks.LLADA_MODEL_KIND:
        raise SystemExit(
            f"{track_id} runs through eval/capability/llada_batch_eval.py, whose "
            f"flags differ from run_eval.py; use the llada lane, not mmlu_body.")
    if track.model_kind == tracks.MISSING_RUNTIME_DEP:
        raise SystemExit(
            f"{track_id} cannot be built in relay2:v2: {track.notes}. This is not "
            f"a config problem -- check_hf_causal_is_loadable reads config.json "
            f"only and passes this row -- so the failure would be an in-pod "
            f"ImportError after the queue wait. Fix the environment or add the "
            f"loader the notes name, then change model_kind.")
    if not 0 <= shard < num_shards:
        raise SystemExit(f"shard {shard} outside range(0,{num_shards})")
    check_loadable_layout(track)
    topo, suffix, shards = pod_topology(shard, num_shards, gpus_per_pod)
    if conditions is None:
        conditions = causal_span_condition(track)
    out = mmlu_outdir(track_id, split, tag)
    exports = {
        "DAN_SCALE_DIR": SCALE_DIR,
        "CKPT_DIR": eval_ckpt_dir(track),
        "MODEL_KIND": track.model_kind,
        "MODEL_DIR": f"{G}/{track.model_dir}",
        "TASK": "multichoice",
        "TASK_DIR": mmlu_task_dir(track, split),
        "CONDITIONS": conditions,
        "OUTDIR": out,
        **load_slots_env(SPEC_BY_GPUS[gpus_per_pod]),
        **topo,
        "PROBE_ONLY": "0",
        "SEED": "42",
    }
    name = f"lrwkv-{tag}-mmlu-{split}-{track_id}-{suffix}".replace("_", "-")
    return make_body(
        name=name,
        command=wrapped(exports, LAUNCHER),
        spec_id=SPEC_BY_GPUS[gpus_per_pod],
        description=(
            f"Long-RWKV ICLR2027 E1: MMLU {split} ({conditions}, one-token label "
            f"convention) shard{'s' if len(shards) > 1 else ''} "
            f"{','.join(str(s) for s in shards)} of {num_shards} for {track_id} "
            f"[{track.paper_arm}, {track.local_label}]"))


def aux_outdir(dataset: str, track_id: str, tag: str = "e1") -> str:
    return f"{OUTPUTS}/{tag}_{dataset}_{track_id}"


def aux_body(dataset: str, track_id: str, shard: int, num_shards: int,
             gpus_per_pod: int = 8, tag: str = "e1") -> dict:
    """One pod scoring an auxiliary basic-ability tree for one checkpoint.

    No ``--max_samples``, deliberately: the launcher passes whatever this body exports,
    and ``run_eval`` truncates with ``tasks[:max_samples]`` over the sorted file list --
    a limit is a prefix of the file order, which for a task-grouped benchmark scores one
    task group and reports it as the dataset.

    The condition follows the same rule as MMLU: ``fwdce`` for a BiRWKV denoiser (the
    forward stream, which is the pretrained causal model) and ``raw`` for a released HF
    causal LM.  Both route through ``_span_nll`` with the same shift, so the arms stay
    comparable.
    """
    spec = check_aux_tree(dataset)
    track = track_by_id(track_id)
    if TOKENIZER_TREE.get(track.model_dir) != "rwkv_world":
        raise SystemExit(
            f"{track_id} tokenizes with "
            f"{TOKENIZER_TREE.get(track.model_dir)!r}, but the {dataset} tree is "
            f"RWKV-world. Scoring one tokenizer's ids with another's model returns a "
            f"real-looking number for text the model has never seen, so the row has to "
            f"be built against its own tree first.")
    check_loadable_layout(track)
    topo, suffix, shards = pod_topology(shard, num_shards, gpus_per_pod)
    out = aux_outdir(dataset, track_id, tag)
    exports = {
        "DAN_SCALE_DIR": SCALE_DIR,
        "CKPT_DIR": eval_ckpt_dir(track),
        "MODEL_KIND": track.model_kind,
        "MODEL_DIR": f"{G}/{track.model_dir}",
        "TASK": "multichoice",
        "TASK_DIR": spec["root"],
        "CONDITIONS": causal_span_condition(track),
        "OUTDIR": out,
        **load_slots_env(SPEC_BY_GPUS[gpus_per_pod]),
        **topo,
        "PROBE_ONLY": "0",
        "SEED": "42",
    }
    name = f"lrwkv-{tag}-{dataset}-{track_id}-{suffix}".replace("_", "-")
    return make_body(
        name=name,
        command=wrapped(exports, LAUNCHER),
        spec_id=SPEC_BY_GPUS[gpus_per_pod],
        description=(
            f"Long-RWKV ICLR2027 E1-basic: {dataset} ({spec['items']} items, "
            f"{exports['CONDITIONS']}, label-span NLL) shard"
            f"{'s' if len(shards) > 1 else ''} "
            f"{','.join(str(s) for s in shards)} of {num_shards} for {track_id} "
            f"[{track.paper_arm}, {track.local_label}]"))


def aux_group(dataset: str, track_id: str, num_shards: int = 8,
              gpus_per_pod: int = 8, tag: str = "e1") -> list[dict]:
    bodies = [aux_body(dataset, track_id, p, num_shards, gpus_per_pod, tag)
              for p in range(num_shards // gpus_per_pod)]
    check_shard_coverage(bodies, num_shards)
    return bodies


def exported(command: str, key: str) -> str | None:
    """Read back an ``export K="V"`` from a finished command string.

    Reads the submitted artifact rather than the dict it was built from, so a
    check cannot pass against an intention the command does not carry.
    """
    marker = f'export {key}="'
    start = command.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = command.find('"', start)
    return command[start:end] if end > start - 1 else None


def check_shard_coverage(bodies: list[dict], num_shards: int) -> None:
    """Prove the jobs tile ``range(num_shards)`` exactly once.

    Each *pod* is worker 0 of its own job, so the launcher's hostname-derived
    ``NODE_RANK`` would be 0 for all of them.  A missing or duplicated rank means
    silent miscoverage: duplicated work, an unscored remainder, and a merge that
    never fires because the shard count never reaches ``NUM_SHARDS``.  Nothing
    raises in the pod -- the numbers that come back are real numbers for the wrong
    subset -- so this is checked here, before anything is submitted.

    The unit that tiles is the **pod**, not the shard.  The launcher runs ``NGPUS``
    processes per pod and gives worker ``w = NODE_RANK*NGPUS + g`` the shards
    ``w, w+W, ...`` with ``W = NNODES*NGPUS``
    (``launch_capability_eval.sh:229-256``).  So a group of *P* pods at *n* GPUs
    covers ``num_shards`` iff the ranks tile ``range(P)`` and ``P*n == num_shards``.
    Asserting the ranks tile ``range(num_shards)`` instead would be right only at
    ``n == 1`` and would reject every correct 2-GPU group.

    ``NNODES`` is verified too, and not just for tidiness: at ``NNODES=1`` the
    launcher ignores an inherited ``NODE_RANK`` entirely and hard-assigns 0
    (``launch_capability_eval.sh:241-246``), so a group that forgot it would have
    every pod run shard 0 no matter how correct its ``NODE_RANK`` looked.
    """
    ranks = []
    for body in bodies:
        rank = exported(body["command"], "NODE_RANK")
        nnodes = exported(body["command"], "NNODES")
        shards = exported(body["command"], "NUM_SHARDS")
        ngpus = exported(body["command"], "NGPUS")
        if rank is None or nnodes is None:
            raise SystemExit(
                f"{body['name']}: the command exports no "
                f"{'NODE_RANK' if rank is None else 'NNODES'}; the pod would "
                f"derive rank 0 from its hostname and score shard 0")
        if int(nnodes) * int(ngpus or 1) != num_shards:
            raise SystemExit(
                f"{body['name']}: NNODES={nnodes} x NGPUS={ngpus} != "
                f"{num_shards} shards, so the launcher's stride "
                f"WORLD_GPUS={int(nnodes) * int(ngpus or 1)} would make pods "
                f"skip or repeat shards")
        if int(shards or -1) != num_shards:
            raise SystemExit(
                f"{body['name']}: NUM_SHARDS={shards} but the group has "
                f"{num_shards} members; the merge gate would never fire")
        if int(nnodes) != len(bodies):
            raise SystemExit(
                f"{body['name']}: NNODES={nnodes} but the group has {len(bodies)} "
                f"pods. The launcher's stride is NNODES*NGPUS, so a group smaller "
                f"than its own NNODES leaves the tail shards unscored and the "
                f"merge never fires.")
        ranks.append(int(rank))
    if sorted(ranks) != list(range(len(bodies))):
        raise SystemExit(
            f"pods do not tile range({len(bodies)}): got {sorted(ranks)}. "
            f"Duplicated ranks redo work and leave the remainder unscored, and "
            f"the result looks like a completed benchmark.")
    names = [b["name"] for b in bodies]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate job names in the group")


def mmlu_group(track_id: str, num_shards: int = 8, split: str = "test",
               conditions: str | None = None, tag: str = "e1",
               gpus_per_pod: int = 1) -> list[dict]:
    bodies = [mmlu_body(track_id, p, num_shards, split, conditions, tag,
                        gpus_per_pod)
              for p in range(num_shards // gpus_per_pod)]
    check_shard_coverage(bodies, num_shards)
    return bodies


def probe_body(track_id: str, split: str = "validation",
               extra_env: dict[str, str] | None = None) -> dict:
    """A 4-example smoke probe: does this checkpoint load and score at all?

    ``PROBE_ONLY=1`` makes the launcher run its own fail-closed 4-example probe
    and exit before the sweep.  One of these per checkpoint costs about a minute
    of queue plus a load, and turns a whole group's worth of wasted GPU time into
    one cheap refusal.  This is the Q1 receipt the plan requires before any E1/E3
    fan-out for that model.

    ``extra_env`` exists for **A/B'ing a change to how the model is built**, and the
    probe is the right instrument for that because its wall time is the quantity a
    construction change moves: measured 2026-09-21, a 2.9B probe's load dominates its
    runtime while the 4 scored examples are seconds.  It is not a general knob -- a
    lane-wide setting belongs in :func:`load_slots_env`, which is derived from the pod
    shape rather than passed per call.
    """
    body = mmlu_body(track_id, 0, 1, split, None, tag="q1probe")
    command = body["command"].replace(
        'export PROBE_ONLY="0"', 'export PROBE_ONLY="1"')
    for key, value in (extra_env or {}).items():
        command = command.replace(
            "export PROBE_ONLY=", f'export {key}="{value}"; export PROBE_ONLY=')
        if exported(command, key) != value:
            raise SystemExit(f"probe_body could not inject {key}={value!r}")
    body["command"] = command
    validate_command(body["command"])
    if exported(body["command"], "PROBE_ONLY") != "1":
        raise SystemExit("PROBE_ONLY did not take; the body would run a full sweep")
    name = f"lrwkv-q1probe-{track_id}".replace("_", "-")
    if extra_env:
        # A probe that runs under different environment is a DIFFERENT job, and it must
        # not answer to the qualification probe's name: `already_submitted` would refuse
        # to send it (the receipt name is taken), and if it did send it, the two runs'
        # receipts would be indistinguishable in one outdir. Suffixing keeps the
        # qualification receipt meaning exactly one thing.
        name += "-env" + "-".join(
            f"{k.lower().replace('_', '')}{v}" for k, v in sorted(extra_env.items()))
    body["name"] = name
    body["description"] = (
        f"Long-RWKV ICLR2027 Q1: 4-example load+score probe for {track_id}"
        + (f" [env {extra_env}]" if extra_env else ""))
    return body


def probe_outdir(track_id: str) -> str:
    return f"{OUTPUTS}/q1probe_mmlu_validation_{track_id}"


def probe_passed(track_id: str) -> tuple[bool, str]:
    """Did this track's Q1 probe actually produce a scored probe file?

    The launcher writes ``<host>.<task>.probe.json`` only after the probe scores
    all four examples, so the file's existence is the receipt.  Checking for a
    non-empty ``*.probe.json`` is deliberately stricter than checking that the
    job's state is ``succeeded``: a preempted-then-retried pod can exit 0 having
    produced nothing.
    """
    out = Path(probe_outdir(track_id))
    if not out.is_dir():
        return False, f"no probe output dir {out}"
    hits = [p for p in out.glob("*.probe.json") if p.stat().st_size > 0]
    if not hits:
        return False, f"no non-empty *.probe.json under {out}"
    return True, str(sorted(hits)[-1])


def require_probe(track_id: str) -> str:
    ok, detail = probe_passed(track_id)
    if not ok:
        raise SystemExit(
            f"{track_id} has no Q1 qualification receipt ({detail}). Run "
            f"`--group q1probe --track {track_id}` and let it finish first: a "
            f"fan-out launched before the load is proven spends the whole "
            f"group's GPU time to discover a load error one job would have found.")
    return detail


# --------------------------------------------------------------------------
# Generative suites (math / code)
# --------------------------------------------------------------------------

#: Vendored task files for the generative suites, with the item count each is
#: cited as.  The count is checked against the file so a truncated or re-derived
#: JSONL cannot be reported under the benchmark's name: "GSM8K" at 1,200 items is
#: a different statistic than GSM8K, and a mean over a subset renders as the same
#: cell as a mean over the whole thing.
GEN_SUITES: Final = {
    "gsm8k": {
        "task": "math",
        "tasks_file": f"{G}/capability_eval_data/tasks/gsm8k_tasks.jsonl",
        "expected_items": 1319,
        "max_new_tokens": 512,
    },
    "humanevalplus": {
        "task": "code",
        "tasks_file": f"{G}/capability_eval_data/tasks_evalplus/humanevalplus.jsonl",
        "expected_items": 164,
        "max_new_tokens": 512,
    },
    "mbppplus": {
        "task": "code",
        "tasks_file": f"{G}/capability_eval_data/tasks_evalplus/mbppplus.jsonl",
        "expected_items": 378,
        "max_new_tokens": 512,
    },
}

#: The sandbox budget the EvalPlus files were calibrated at.  Not a tuning knob:
#: ``eval_code``'s own defaults are 256 MiB / 5 s, and at those four of the 542
#: canonical solutions are SIGKILLed.  ``_run_single_test`` returns
#: ``(False, stderr, False)`` for any nonzero exit, so a resource kill is recorded
#: as ``tests_failed`` -- identical in the artifact to a wrong answer.  Measured
#: 2026-09-20: 163/164 and 378/378 at 4096/30.
CODE_MEM_MB: Final = 4096
CODE_TIMEOUT_S: Final = 30


def suite_manifest(suite: str) -> dict | None:
    path = Path(GEN_SUITES[suite]["tasks_file"])
    man = path.with_name(f"{suite}.manifest.json")
    if not man.is_file():
        return None
    return json.loads(man.read_text(encoding="utf-8"))


def check_gen_suite(suite: str) -> dict:
    """Refuse a generative suite whose file is absent, short, or uncalibrated.

    Three separate claims, all checked against disk rather than declared:

    * the file exists and has the item count the benchmark is cited at;
    * its sha256 still matches the manifest that describes it, so the shards are
      scoring the file the manifest's provenance belongs to;
    * for the EvalPlus suites, a sandbox calibration is *attached*.  Without one
      there is no ceiling, and a 0 % arm cannot be told apart from a sandbox that
      never ran the grader -- which is exactly what 0/40 looked like before the
      thread-cap fix.
    """
    if suite not in GEN_SUITES:
        raise SystemExit(f"unknown suite {suite!r}; known: {sorted(GEN_SUITES)}")
    spec = GEN_SUITES[suite]
    path = Path(spec["tasks_file"])
    if not path.is_file():
        raise SystemExit(
            f"{suite}: {path} does not exist. Build it "
            f"(lrwkv_evidence.e1.evalplus_tasks) before scheduling shards -- the "
            f"pod would load the model and then fail on a missing file.")
    n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if n != spec["expected_items"]:
        raise SystemExit(
            f"{suite}: {path} holds {n} items but the benchmark is cited at "
            f"{spec['expected_items']}. A different item count is a different "
            f"statistic and would render as the same cell.")
    man = suite_manifest(suite)
    if spec["task"] != "code":
        return {"suite": suite, "items": n, "manifest": man}
    if man is None:
        raise SystemExit(
            f"{suite}: no manifest beside {path}. The code lane needs the "
            f"provenance and the calibrated sandbox settings; run "
            f"lrwkv_evidence.e1.evalplus_tasks.")
    cal = man.get("sandbox_calibration") or {}
    if cal.get("status") != "attached":
        raise SystemExit(
            f"{suite}: the manifest has no attached sandbox calibration "
            f"(status {cal.get('status')!r}). Without a measured ground-truth "
            f"ceiling a 0 % arm is indistinguishable from a sandbox that never ran "
            f"the grader: canonical solutions scored 0/40 under RLIMIT_NPROC "
            f"before the thread-cap fix, with every row tagged 'tests_failed'. "
            f"Run calibrate_evalplus_sandbox.py and "
            f"`evalplus_tasks.py --attach-calibration`.")
    if cal.get("unexplained_failures"):
        raise SystemExit(
            f"{suite}: the calibration has {len(cal['unexplained_failures'])} "
            f"unexplained ground-truth failures "
            f"({cal['unexplained_failures'][:5]}). Those rows would be charged to "
            f"the model; classify them before scheduling shards.")
    settings = man.get("sandbox_settings_validated") or {}
    if (settings.get("mem_mb"), settings.get("timeout_s")) != (
            CODE_MEM_MB, float(CODE_TIMEOUT_S)):
        raise SystemExit(
            f"{suite}: the calibration certifies mem_mb={settings.get('mem_mb')} "
            f"timeout_s={settings.get('timeout_s')} but the bodies export "
            f"{CODE_MEM_MB}/{CODE_TIMEOUT_S}. A shard run at an uncalibrated "
            f"budget has no ceiling to compare against; re-measure at the budget "
            f"you intend to run.")
    live = sha256_head(path)
    if man.get("jsonl_sha256") != live:
        raise SystemExit(
            f"{suite}: {path} hashes to {live} but its manifest records "
            f"{man.get('jsonl_sha256')}. The file changed after the manifest and "
            f"the attached ceiling belongs to the old one; rebuild and re-measure.")
    return {"suite": suite, "items": n, "manifest": man,
            "ceiling": cal.get("ground_truth_pass_rate"),
            "scorable_items": cal.get("scorable_items"),
            "isolation_mode_measured": cal.get("isolation_mode_measured")}


def sha256_head(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: The generative decoder protocol every banked GSM8K/HumanEval number on disk was
#: produced under, read back off the pods that produced them (2026-09-21):
#: ``cap_eval_s9500_{math,code}``, ``cap_eval_b3_gsm8k_g4_*``, ``cap_eval_t2_*``,
#: ``cap_eval_wl1_*`` and the W-L0 battery all report
#: ``CONDITIONS=iter16 / COMMIT_GROUP=4 / COMMIT_ORDER=confidence`` (code also
#: ``TEMPERATURE=0.2``).  This is a *comparability* constraint, not a tuning choice:
#: the paper puts the new rows in the same table as those numbers, and g=0 is a
#: different decode path (single-forward multi-commit) from g=4.
GEN_STEPS: Final = 16
GEN_COMMIT_GROUP: Final = 4
GEN_COMMIT_ORDER: Final = "confidence"
GEN_TEMPERATURE: Final = "0.2"

#: ``CONDITIONS`` for a generative suite, per model kind.  For math/code the value
#: carries the **denoise step count** (``iter<N>`` -> steps=N), so the multichoice
#: ``fwdce``/``raw`` table does not apply -- but it must still be *set*.
#:
#: Measured 2026-09-21: leaving it unset does not defer the choice, it inherits the
#: launcher's *multichoice* default ``raw,ddpm100``, and that broke both kinds in
#: opposite directions.  ``birwkv_diffusion`` rejects any non-``iter`` method
#: (run_eval.py:338,481) and 55 pods died there *after* the probe passed -- the
#: math/code probe runs ``--arm math_probe``, which bypasses the condition parse
#: entirely, so a probe OK says nothing about the sweep's conditions.  ``hf_causal``
#: accepted it and was worse: ``raw`` and ``ddpm100`` both map to method ``raw`` and
#: the same ``HFCausalGenerator``, so every released baseline computed one statistic
#: twice under two arm labels (Qwen2.5-3B GSM8K em 0.8061 under both, 143/165
#: byte-identical completions) -- double the GPU-hours and an ``n`` that looks
#: doubled to anything counting records.
GEN_CONDITION: Final = {
    "birwkv_diffusion": f"iter{GEN_STEPS}",
    # Distinct label for the same greedy pass@1 decode, so the merged artifact does
    # not claim a diffusion step count for a model that has no denoise loop
    # (parse_generative_condition maps `greedy` -> method `raw`).
    "hf_causal": "greedy",
}


def generative_condition(track: tracks.Track) -> str:
    """The ``CONDITIONS`` value for a math/code shard of ``track``.

    Refuses rather than falling through to the launcher default, for the reason
    recorded on :data:`GEN_CONDITION`: the default is a multichoice default, it is
    wrong for both kinds, and neither failure is visible at submit time.
    """
    cond = GEN_CONDITION.get(track.model_kind)
    if cond is None:
        raise SystemExit(
            f"{track.track_id}: no generative CONDITIONS is known for model_kind "
            f"{track.model_kind!r}. Leaving it unset is not neutral -- the launcher "
            f"falls back to the multichoice default 'raw,ddpm100', which "
            f"birwkv_diffusion rejects on the card after the probe has passed and "
            f"hf_causal silently scores twice.")
    return cond


def gen_body(track_id: str, suite: str, shard: int, num_shards: int,
             tag: str = "e1", gpus_per_pod: int = 1,
             conditions: str | None = None) -> dict:
    """One pod of a generative suite (GSM8K / HumanEval+ / MBPP+).

    ``shard`` is the pod rank and ``gpus_per_pod`` shards run concurrently in it;
    see :func:`pod_topology`.

    ``conditions=None`` (the default) resolves per model kind via
    :func:`generative_condition`, and the decoder knobs are pinned to the banked
    protocol (:data:`GEN_STEPS`, :data:`GEN_COMMIT_GROUP`, :data:`GEN_COMMIT_ORDER`).
    These are exported explicitly rather than left to the launcher: every value it
    would default to differs from what the numbers this table joins were run under.
    """
    track = track_by_id(track_id)
    if track.model_kind == tracks.FOREIGN_ARCH:
        raise SystemExit(
            f"{track_id} is a TiedBiRWKV7Block checkpoint and has no "
            f"load_birwkv_diffusion path (see mmlu_body); it joins the campaign at "
            f"E3 through rwkv04b/longrwkv/eval.")
    if track.model_kind == tracks.NO_LOADER:
        raise SystemExit(
            f"{track_id} has no loader in this harness "
            f"(lrwkv_evidence/tracks.py marks it {tracks.NO_LOADER}).")
    if track.model_kind == tracks.LLADA_MODEL_KIND:
        raise SystemExit(
            f"{track_id} runs through eval/capability/llada_batch_eval.py, whose "
            f"flags differ from run_eval.py; use the llada lane.")
    if track.model_kind == tracks.MISSING_RUNTIME_DEP:
        raise SystemExit(
            f"{track_id} cannot be built in relay2:v2 ({tracks.MISSING_RUNTIME_DEP}): "
            f"{track.notes}")
    if not 0 <= shard < num_shards:
        raise SystemExit(f"shard {shard} outside range(0,{num_shards})")
    check_loadable_layout(track)
    spec = GEN_SUITES[suite]
    checked = check_gen_suite(suite)
    if conditions is None:
        conditions = generative_condition(track)
    topo, suffix, shards = pod_topology(shard, num_shards, gpus_per_pod)
    out = f"{OUTPUTS}/{tag}_{suite}_{track_id}"
    exports = {
        "DAN_SCALE_DIR": SCALE_DIR,
        "CKPT_DIR": eval_ckpt_dir(track),
        "MODEL_KIND": track.model_kind,
        "MODEL_DIR": f"{G}/{track.model_dir}",
        "TASK": spec["task"],
        "TASKS_FILE": spec["tasks_file"],
        "MAX_NEW_TOKENS": str(spec["max_new_tokens"]),
        "CONDITIONS": conditions,
        "COMMIT_GROUP": str(GEN_COMMIT_GROUP),
        "COMMIT_ORDER": GEN_COMMIT_ORDER,
        # TEMPERATURE is in the launcher's COMMON_ARGS, so it reaches math as well
        # as code -- it is NOT a code-only knob and must not be gated on the task.
        # Banked GSM8K and HumanEval both ran at 0.2. This equals the launcher's
        # current default, which is exactly why it is pinned: the launcher is a
        # copied artifact under $G and a default that moves would re-point every
        # generative row at a different decoder with nothing failing.
        "TEMPERATURE": GEN_TEMPERATURE,
        "OUTDIR": out,
        **topo,
        "PROBE_ONLY": "0",
        "SEED": "42",
    }
    if spec["task"] == "code":
        # SUITE selects the assembly rule in code.py (humanevalplus concatenates
        # prompt+completion, mbppplus is standalone). Getting it wrong does not
        # error: MBPP+ through the humaneval rule put prose into the program and
        # every row came back `syntax_error`.
        exports["SUITE"] = suite
        exports["CODE_MEM_MB"] = str(CODE_MEM_MB)
        exports["CODE_TIMEOUT_S"] = str(CODE_TIMEOUT_S)
    name = (f"lrwkv-{tag}-{suite}-{track_id}-{suffix}".replace("_", "-"))
    ceiling = checked.get("ceiling")
    return make_body(
        name=name,
        command=wrapped(exports, LAUNCHER),
        spec_id=SPEC_BY_GPUS[gpus_per_pod],
        description=(
            f"Long-RWKV ICLR2027 E1: {suite} shard"
            f"{'s' if len(shards) > 1 else ''} "
            f"{','.join(str(s) for s in shards)} of {num_shards} for "
            f"{track_id} [{track.paper_arm}, {track.local_label}]"
            + (f"; ground-truth ceiling {ceiling:.4f} at "
               f"{CODE_MEM_MB}MiB/{CODE_TIMEOUT_S}s" if ceiling is not None else "")))


def gen_group(track_id: str, suite: str, num_shards: int = 4,
              tag: str = "e1", gpus_per_pod: int = 1,
              conditions: str | None = None) -> list[dict]:
    bodies = [gen_body(track_id, suite, p, num_shards, tag, gpus_per_pod,
                       conditions)
              for p in range(num_shards // gpus_per_pod)]
    check_shard_coverage(bodies, num_shards)
    return bodies


#: What ``sandbox._active_isolation_mode()`` must report inside a pod.  The login
#: node reports ``socket_stub`` (no ``unshare`` privileges), so the calibration
#: receipt certifies the *ceiling* at 4096 MiB / 30 s and says nothing about the
#: isolation the scores were produced under.  Those are two different claims and
#: only the pod can settle the second one.
REQUIRED_POD_ISOLATION: Final = "unshare"


def check_code_isolation(out_dir: str | Path) -> dict:
    """Read a finished code shard's own records and verify how it was isolated.

    This is deliberately a *post-hoc reader*, not a preflight: the mode is a
    property of the pod the shard ran in, and no submit-time check can observe it.
    ``run_eval`` writes ``isolation_mode`` onto every code record
    (run_eval.py:531), so the artifact answers the question -- but only if someone
    asks.  An unasked shard's numbers look identical either way.

    Returns the observed modes per file.  Raises if any shard ran under something
    other than :data:`REQUIRED_POD_ISOLATION`, because a weaker sandbox is a
    different protocol than the one the paper would describe.
    """
    out = Path(out_dir)
    if not out.is_dir():
        raise SystemExit(f"{out} does not exist; no code shard has written there")
    shards = sorted(p for p in out.glob("code-*.json") if p.stat().st_size > 0)
    if not shards:
        raise SystemExit(
            f"no non-empty code-*.json under {out}. The isolation mode is recorded "
            f"per record, so an unfinished lane cannot be certified -- and must not "
            f"be reported as certified either.")
    observed: dict[str, dict[str, int]] = {}
    for path in shards:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records") or []
        modes: dict[str, int] = {}
        for rec in records:
            mode = str(rec.get("isolation_mode"))
            modes[mode] = modes.get(mode, 0) + 1
        if not modes:
            raise SystemExit(
                f"{path} has no records carrying isolation_mode; it cannot certify "
                f"how its tests were sandboxed.")
        observed[path.name] = modes
    wrong = {name: modes for name, modes in observed.items()
             if set(modes) != {REQUIRED_POD_ISOLATION}}
    if wrong:
        raise SystemExit(
            f"code shards under {out} did not all run under "
            f"{REQUIRED_POD_ISOLATION!r}: {json.dumps(wrong)}. The calibration "
            f"receipt was taken on the login node (socket_stub) and certifies the "
            f"ceiling only; a shard scored under a weaker sandbox is a different "
            f"protocol than the one the paper describes, so these scores are not "
            f"interchangeable with a calibrated run's.")
    return observed
