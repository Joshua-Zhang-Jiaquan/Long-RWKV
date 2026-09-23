"""Ledger and submission driver for the Long-RWKV ICLR-2027 evaluation campaign.

Three properties carried over from ``rwkv04b_v7/qz/campaign.py``, each of which
this project has already paid for once:

**Submissions are recorded before they happen.**  The ledger row is appended
*before* ``CreateJob``, so a crash between the two leaves an ``attempted`` row --
which blocks -- rather than nothing, which permits a duplicate.  A refused
submission still bills, and a duplicate 1-GPU 12-hour job is 12 wasted card-hours
that nothing reclaims.

**The ledger is created deliberately, once.**  ``--authorize`` writes the genesis
row and refuses if one exists; every other entry point refuses without it.  The
file is the operator's statement that this campaign may spend GPU quota, so it is
not something a script creates as a side effect of being run.

**Gates are read from artifacts, not from intent.**  A fan-out group is submitted
only when the model's Q1 probe file is on GPFS.  "The probe job was submitted" is
not the same claim as "the checkpoint loads", and the difference is a whole group's
GPU time spent to rediscover a load error.

Usage::

    python3 qz/campaign.py --status
    python3 qz/campaign.py --authorize                 # once
    python3 qz/campaign.py --group q1probe --dry-run
    python3 qz/campaign.py --group q1probe --submit
    python3 qz/campaign.py --group e1-mmlu --track a29_c6loop_s20500 --submit
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emit_jobs as E  # noqa: E402

LEDGER = E.LEDGER
QZ = "qz"

JOB_ID_PATTERN = re.compile(r"job_id[\"':\s]+(job-[0-9a-f-]+)")

#: A name in any of these states must not be submitted again.  ``attempted`` is
#: here because it means "a CreateJob call may have reached the scheduler": the
#: honest reading is that a live job may exist, so the name is burnt until someone
#: checks.  Treating it as retryable is how one ends up billing two jobs.
BLOCKING_STATES = frozenset({"attempted", "submitted", "succeeded"})

#: Scheduler status -> the ledger state it settles to.  Only terminal statuses are
#: listed: anything else (``job_running``, ``job_pending``, ``job_restarting``) is
#: still live and must keep blocking.  ``job_restarting`` in particular is how the
#: auto-fault-tolerance surfaces a crashed *round*, not a finished job -- it can
#: still end ``job_succeeded`` on a later round.
TERMINAL_STATUS: Final = {
    "job_succeeded": "succeeded",
    "job_failed": "failed",
    "job_stopped": "stopped",
    "job_cancelled": "stopped",
    "job_canceled": "stopped",
    "job_deleted": "stopped",
}

#: The Q1 order: every distinct checkpoint that E1/E3 will later evaluate needs a
#: load receipt, and the cheapest receipt is a 4-example probe.  Built from the
#: registry so a track added there cannot be silently skipped here.
#:
#: ``FOREIGN_ARCH`` is excluded for the reason the registry records: those
#: checkpoints store one tied mixer (``layers.N.attn.*``) where this harness
#: builds two untied ones, so a probe cannot pass -- it pays a full load and dies
#: on ``checkpoint/model mismatch``.  Excluding them here is not a claim they are
#: unusable; they are evaluated through ``rwkv04b/longrwkv/eval`` in the E3 lane,
#: which needs its own qualification order.
#:
#: The exclusion reads :data:`tracks.UNSCHEDULABLE_KINDS` rather than listing the
#: kinds, because this list and the emitter's refusals must agree: when they
#: disagree the Q1 order offers a track the emitter rejects, and that only shows
#: up at submit, one job at a time.
def q1_track_ids() -> list[str]:
    """Tracks the campaign may schedule.  One definition, in the emitter.

    Kept as a thin alias rather than a second predicate: the previous copy here tested
    ``model_kind`` and ``TOKENIZER_TREE`` itself, so the two could drift and a track
    could be schedulable for the emitter and unschedulable for the campaign.
    """
    return E.schedulable_tracks()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def already_submitted(name: str) -> dict | None:
    """The latest blocking row for this name, or None.

    Reads *all* rows and keeps the last state per name, so a ``failed`` row
    recorded after an ``attempted`` one correctly unblocks a legitimate resubmit.
    """
    latest: dict | None = None
    for row in read_ledger():
        if row.get("name") == name and row.get("state") in E_STATES:
            latest = row
    if latest is not None and latest.get("state") in BLOCKING_STATES:
        return latest
    return None


E_STATES = frozenset({"attempted", "submitted", "failed", "succeeded", "stopped"})

#: A group's job name ends in its topology: ``s<shard>of<n>`` for a 1-GPU pod,
#: ``p<pod>of<pods>x<gpus>`` for a wider one.  The prefix is the group.
TOPOLOGY_SUFFIX: Final = re.compile(
    r"^(?P<group>.*)-(?:s(?P<shard>\d+)of(?P<shards1>\d+)"
    r"|p(?P<pod>\d+)of(?P<pods>\d+)x(?P<gpus>\d+))$")


def shard_shape_in_ledger(group_prefix: str) -> int | None:
    """The GPUs-per-pod this group was already submitted at, or ``None``.

    A shard group must not change shape midway.  Measured in the live ledger
    2026-09-20: ``a29_c6_f2_s13000`` has shards 0,1,2 ``submitted`` as 1-GPU pods
    and 3-7 ``failed`` (the capacity refusal).  Re-emitting that group at 2 GPU/pod
    would ship pod 0 covering shards 0 *and* 1, so three live jobs would be
    duplicated -- and the duplicates carry *different names*, so
    :func:`already_submitted` cannot see the collision.  Name-level de-duplication
    is blind to this because the hazard is at the level of the shard, not the name.

    Only non-``failed`` rows count: a refused name holds no job, so a group whose
    every row failed is free to be re-shaped.
    """
    shapes: set[int] = set()
    for name, row in latest_by_name().items():
        if row.get("state") == "failed":
            continue
        m = TOPOLOGY_SUFFIX.match(name)
        if m is None or m.group("group") != group_prefix:
            continue
        shapes.add(1 if m.group("shard") is not None else int(m.group("gpus")))
    if len(shapes) > 1:
        raise SystemExit(
            f"{group_prefix} already has live rows at more than one pod shape "
            f"{sorted(shapes)}; the shard coverage cannot be reasoned about. "
            f"Reconcile and settle that group before submitting more.")
    return shapes.pop() if shapes else None


def latest_by_name() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in read_ledger():
        if row.get("name") and row.get("state") in E_STATES:
            latest[row["name"]] = row
    return latest


def check_group_shape(bodies: list[dict], gpus_per_pod: int) -> None:
    """Refuse a group whose shape disagrees with what its live rows already used."""
    if not bodies:
        return
    prefixes = set()
    for body in bodies:
        m = TOPOLOGY_SUFFIX.match(body["name"])
        if m is None:
            raise SystemExit(
                f"{body['name']} carries no topology suffix, so its group cannot "
                f"be identified and a shape change could not be detected")
        prefixes.add(m.group("group"))
    for prefix in sorted(prefixes):
        prior = shard_shape_in_ledger(prefix)
        if prior is not None and prior != gpus_per_pod:
            raise SystemExit(
                f"{prefix} has live rows submitted at {prior} GPU/pod but this "
                f"group renders at {gpus_per_pod} GPU/pod. The two shapes cover "
                f"the same shards under different names, so the duplicate is "
                f"invisible to name de-duplication and would bill twice. Finish "
                f"or reconcile that group at {prior} GPU/pod first.")


def parse_job_id(text: str) -> str | None:
    m = JOB_ID_PATTERN.search(text)
    return m.group(1) if m else None


def authorize() -> int:
    if LEDGER.exists():
        raise SystemExit(
            f"{LEDGER} already exists: genesis has been written and cannot be "
            f"redone. Authorizing twice would reset the de-duplication history "
            f"that every later submission is checked against.")
    append_ledger({"state": "genesis", "at": timestamp(),
                   "campaign": "long_rwkv_iclr2027",
                   "note": "operator authorization to spend GPU quota on the "
                           "Long-RWKV ICLR-2027 evaluation campaign",
                   "spec_1gpu": E.SPEC_1GPU,
                   "max_gpus_per_job": E.MAX_GPUS_PER_JOB})
    print(f"authorized: {LEDGER}")
    return 0


#: Refusals that mean "the cluster will not take another job right now".  Retrying
#: these is not just useless, it is harmful: looping on a quota refusal is a standing
#: operator rule, and a sweep that keeps calling CreateJob after one turns hundreds of
#: `failed` rows into noise a later reconcile has to read.
#:
#: Matched on the *common substring*, not on a whole message.  The first version of
#: this list held ``父项目配额不足`` ("parent project") because that is the variant in
#: the operator notes -- and the live refusal 2026-09-20 said ``总项目配额不足``
#: ("total project"), so the breaker did not fire and 196 names were attempted against
#: a scheduler that had already said no.  ``配额不足`` covers both and any third
#: prefix; the English tokens cover the pre-translation wording.
QUOTA_REFUSALS: Final = ("配额不足", "quota", "Insufficient", "insufficient",
                         "exceeds", "超出")

#: What the scheduler actually enforces, read off the refusal message 2026-09-20:
#: 16 GPU may RUN at once and 40 GPU may be SUBMITTED at once (running + queued).
#: These are per-project and unreadable through the CLI -- ``qz`` has no quota verb --
#: so they are recorded here and re-derived from any refusal that quotes them.
#: They are the reason this campaign drip-feeds: 392 E1 shards cannot be outstanding.
RUNNING_GPU_QUOTA: Final = 16
SUBMITTED_GPU_QUOTA: Final = 40

#: The operator's standing allowance, 2026-09-20: "you have up to 64 h100 to use",
#: confirmed the same day by an explicit choice of "64 GPUs live".  Enforced here,
#: *before* CreateJob, because it is not what the scheduler enforces.  Measured the
#: hard way the same day: a tick that used the scheduler's capacity refusal as its
#: only brake put 390 GPUs in flight, because ``PROJECT_SHARED`` turned out to have a
#: far larger quota than ``PROJECT_1GPU``'s 16/40 and simply accepted all 172 pods.  A
#: refusal is the *cluster's* limit; this is the *user's*, and only one of the two is
#: ours to respect.
#:
#: **This still says 64 after 2026-09-21's "use 8h100 for each job".**  That
#: direction changes the *shape* of a job, not the total: 64 live GPUs at 8 per pod is
#: **8 concurrent pods** where the same 64 bought 32 pods at 2 cards.  The gain is not
#: throughput (identical) but load amortisation -- one queue wait and one model load
#: per benchmark instead of eight -- and that is worth having on its own.  Raising the
#: total is the operator's call to state, not ours to infer from a request about job
#: size; the census prints the live figure on every tick so the decision stays visible.
#:
#: Note the real binding constraint is smaller than this: the scheduler admits 16 GPU
#: *running* per project (:data:`E.PROJECT_RUNNING_GPU_QUOTA`), so the shared project
#: runs two 8-GPU pods at a time whatever this allowance says.
OWN_LIVE_GPU_ALLOWANCE: Final = 32

#: Scheduler statuses that still hold GPUs.  ``job_creating``/``job_queuing`` count:
#: they consume submitted quota and will consume cards without any further action,
#: so treating them as free is how a census under-reports and the allowance is blown.
LIVE_STATUSES: Final = frozenset({
    "job_running", "job_queuing", "job_creating", "job_restarting", "job_pending"})


def list_all_jobs() -> list[dict]:
    """Every job in the workspace, across pages.

    ``--page-size`` above 100 is refused with a misleading "page_size too large",
    and the flag is ``--page-number`` (lower-hyphen), not ``--PageNumber``.
    """
    rows: list[dict] = []
    for page in range(1, 40):
        result = subprocess.run(
            [QZ, "train", "ListJobs", "--workspace-id", E.WORKSPACE,
             "--page-number", str(page), "--page-size", "100", "-o", "json"],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(
                f"ListJobs page {page} failed: "
                f"{((result.stdout or '') + (result.stderr or ''))[-300:]}")
        try:
            decoded = json.loads(result.stdout)
        except ValueError as exc:
            raise SystemExit(f"ListJobs page {page} returned unparsable JSON: {exc}")
        # Some qz versions exit zero for API errors. Never interpret an error
        # envelope or incomplete pagination as zero reserved campaign GPUs.
        if decoded.get('ResponseMetadata', {}).get('Error'):
            raise SystemExit(f"ListJobs page {page} API error: {decoded['ResponseMetadata']['Error']}")
        if not isinstance(decoded.get('Result'), dict):
            raise SystemExit(f"ListJobs page {page} has no valid Result")
        batch = decoded['Result'].get('jobs') or []
        if not isinstance(batch, list) or any(not isinstance(job, dict) for job in batch):
            raise SystemExit(f"ListJobs page {page} has invalid job records")
        if not batch:
            return rows
        rows.extend(batch)
    raise SystemExit('ListJobs pagination limit reached; capacity is unknown')


def live_census(jobs: list[dict] | None = None) -> dict:
    """How many GPUs this campaign currently holds, read from the scheduler.

    Counts ``gpu_count`` as the scheduler reports it rather than re-deriving it from
    the spec id: the spec is what we *asked* for and ``gpu_count`` is what was
    *granted*, and this project has already been bitten by the difference (a task
    declaring 1 GPU was granted 8).  Only names this campaign owns are counted --
    foreign jobs consume the same project quota but are not ours to stop.
    """
    jobs = list_all_jobs() if jobs is None else jobs
    mine, running, submitted, job_rows = 0, 0, 0, []
    for job in jobs:
        name = job.get("name") or ""
        if not name.startswith("lrwkv-") or job.get("status") not in LIVE_STATUSES:
            continue
        gpus = int(job.get("gpu_count") or 0)
        mine += 1
        submitted += gpus
        if job.get("status") == "job_running":
            running += gpus
        job_rows.append({"name": name, "job_id": job.get("job_id"),
                         "status": job.get("status"), "gpu_count": gpus,
                         "project_id": job.get("project_id")})
    return {"live_jobs": mine, "live_gpus_submitted": submitted,
            "live_gpus_running": running, "jobs": job_rows}


def allowance_headroom(census: dict | None = None) -> int:
    """GPUs this campaign may still put in flight under the operator's allowance."""
    census = live_census() if census is None else census
    return max(0, OWN_LIVE_GPU_ALLOWANCE - census["live_gpus_submitted"])


def is_quota_refusal(text: str) -> bool:
    return any(token in text for token in QUOTA_REFUSALS)


def submit_one(body: dict, dry_run: bool) -> dict:
    payload = json.dumps(body)
    name = body["name"]
    if dry_run:
        print(f"--- {name}")
        print(payload)
        return {"name": name, "state": "dry_run"}
    if not LEDGER.exists():
        raise SystemExit(
            f"no ledger at {LEDGER}: run `--authorize` once. A submission that "
            f"cannot be recorded cannot be de-duplicated later.")
    prior = already_submitted(name)
    if prior is not None:
        print(f"SKIP {name}: already {prior['state']} at {prior.get('at')} "
              f"(job_id {prior.get('job_id')})")
        return {"name": name, "state": "skipped", "prior": prior}
    append_ledger({"state": "attempted", "name": name, "at": timestamp(),
                   "description": body.get("description", "")})
    result = subprocess.run(
        [QZ, "train", "CreateJob", "--data", payload, "-o", "yaml"],
        capture_output=True, text=True)
    ok = result.returncode == 0 and "Error" not in (result.stdout or "")[:400]
    job_id = parse_job_id(result.stdout or "")
    row = {"state": "submitted" if ok else "failed", "name": name,
           "job_id": job_id, "at": timestamp(), "returncode": result.returncode,
           "tail": ((result.stdout or "") + (result.stderr or ""))[-400:]}
    if job_id is None:
        row["unparsed_response"] = ((result.stdout or "") + (result.stderr or ""))[-4000:]
    if not ok and is_quota_refusal(row["tail"]):
        # Distinguish "this body is wrong" from "the cluster is full": only the
        # first is worth fixing, and only the second should stop the sweep.
        row["refusal_kind"] = "capacity"
    append_ledger(row)
    print(f"{'OK  ' if ok else 'FAIL'} {name} job_id={job_id}")
    if not ok:
        print(((result.stdout or "") + (result.stderr or ""))[-1500:], file=sys.stderr)
    return row


def job_status(job_id: str) -> tuple[str | None, dict]:
    """The scheduler's current status for one job, or ``None`` if unreadable.

    Unreadable is returned rather than raised: a single ``AccessForbidden`` (which
    is also what a mistyped id looks like) must not stop the rest of the sweep, and
    it must not be mistaken for a terminal state.
    """
    result = subprocess.run(
        [QZ, "train", "GetJob", "--job-id", job_id, "-o", "json"],
        capture_output=True, text=True)
    if result.returncode != 0:
        return None, {"error": ((result.stdout or "") + (result.stderr or ""))[-300:]}
    try:
        payload = json.loads(result.stdout)["Result"]
    except (ValueError, KeyError) as exc:
        return None, {"error": f"unparsed GetJob response: {exc}"}
    return payload.get("status"), payload


def reconcile(dry_run: bool = True) -> int:
    """Settle ledger rows whose job has reached a terminal scheduler state.

    Nothing in the submit path writes a crashed job's outcome: ``submit_one``
    records ``submitted`` when ``CreateJob`` is accepted and never looks again.  So
    a job that failed in-pod keeps a ``submitted`` row forever, and
    ``already_submitted`` -- correctly, by its own rule -- refuses the resubmit that
    the fix calls for.  The repair is to record what the scheduler says, not to
    weaken the rule: an unreconciled ``submitted`` row and a genuinely running job
    are indistinguishable from the ledger alone, which is exactly why the block is
    right.

    Only terminal statuses are written.  ``job_restarting`` is skipped on purpose:
    it is a crashed *round* under auto-fault-tolerance, and the job may still
    succeed on a later one.
    """
    rows = read_ledger()
    job_ids: dict[str, str] = {}
    for row in rows:
        if row.get("name") and row.get("job_id"):
            job_ids[row["name"]] = row["job_id"]
    settled, live, unreadable = [], [], []
    for name, job_id in sorted(job_ids.items()):
        if already_submitted(name) is None:
            continue                      # already settled to a non-blocking state
        status, payload = job_status(job_id)
        if status is None:
            unreadable.append({"name": name, "job_id": job_id,
                               "detail": payload.get("error", "")})
            continue
        state = TERMINAL_STATUS.get(status)
        if state is None:
            live.append({"name": name, "job_id": job_id, "status": status,
                         "round": payload.get("current_running_round")})
            continue
        row = {"state": state, "name": name, "job_id": job_id, "at": timestamp(),
               "scheduler_status": status,
               "rounds_run": payload.get("current_running_round"),
               "finished_at": payload.get("finished_at"),
               "source": "reconcile"}
        settled.append(row)
        if not dry_run:
            append_ledger(row)
    print(json.dumps({
        "reconcile": "DRY RUN" if dry_run else "WRITTEN",
        "checked": len(job_ids),
        "settled": settled,
        "still_live": live,
        "unreadable": unreadable,
    }, indent=2))
    return 0


#: The ``--shards`` default.  A sentinel, not a value: when it is left alone the
#: generative groups use their own per-suite counts (:data:`GEN_SHARDS`), and an
#: explicit ``--shards`` overrides them.
DEFAULT_SHARDS: Final = 8

#: Groups whose bodies are generative-suite shards; the group name carries the suite.
GEN_GROUPS: Final = {
    "e1-gsm8k": "gsm8k",
    "e1-humanevalplus": "humanevalplus",
    "e1-mbppplus": "mbppplus",
}

#: The auxiliary basic-ability lane: the ten RWKV-world four-choice trees in
#: :data:`E.AUX_TREES`, scored by label-span NLL.  Its own group rather than a synonym
#: for ``e1-mmlu`` because the protocol differs (a span, not a one-token label) and
#: because it needs no probe receipt -- the MMLU Q1 probe validates the load, which
#: this lane inherits, but a dataset is not a model and the tree is validated instead
#: by :func:`E.check_aux_tree` on every build.
AUX_GROUP: Final = "e1-basic"

#: Suite shard counts.  These are not arbitrary: a shard's wall clock is
#: (items/num_shards) x max_new_tokens x NFE, and the 1-GPU wall is 12 h.  GSM8K's
#: 1319 items at 512 new tokens are the long pole, so it gets the most shards; the
#: 164 HumanEval+ items would waste a card split 8 ways.
#:
#: **All three are 8 as of 2026-09-21** ("use 8h100 for each benchmark"): a pod on the
#: 8-GPU shape has eight H100 and the launcher pins one shard per card, so a
#: four-shard split would run four workers and leave four cards idle for the length of
#: the job.  HumanEval+ at 164 items over 8 shards is 20.5 items per shard -- still a
#: load-bound shard, but the load is now shared by eight cards instead of one, which is
#: the whole point of the wider shape.
GEN_SHARDS: Final = {"gsm8k": 8, "humanevalplus": 8, "mbppplus": 8}


def aux_merged(dataset: str, track_id: str, tag: str = "e1") -> bool:
    """Has this (dataset, track) already produced a merged reading?

    The merge gate only fires once all ``NUM_SHARDS`` shard files exist, so the merged
    file is the campaign's own receipt that the dataset was scored end to end.  Same
    argument as :func:`E.merged_present`, and the same reason the selector is a receipt
    rather than a guess.
    """
    merged = Path(E.aux_outdir(dataset, track_id, tag)) / "merged"
    return merged.is_dir() and any(merged.glob("*.json"))


def aux_bodies(dataset: str, track: str | None, shards: int,
               gpus_per_pod: int = 8, tag: str = "e1") -> list[dict]:
    ids = [track] if track else E.aux_track_ids()
    bodies: list[dict] = []
    for tid in ids:
        try:
            bodies.extend(E.aux_group(dataset, tid, num_shards=shards,
                                      gpus_per_pod=gpus_per_pod, tag=tag))
        except SystemExit as exc:
            print(f"# skip {AUX_GROUP}/{dataset}/{tid} at {gpus_per_pod} GPU/pod: "
                  f"{str(exc)[:160]}", file=sys.stderr)
    check_group_shape(bodies, gpus_per_pod)
    return bodies


def group_bodies(group: str, track: str | None, shards: int,
                 split: str, gpus_per_pod: int = 1,
                 tag: str = "e1", dataset: str | None = None) -> list[dict]:
    if group == "q1probe":
        ids = [track] if track else q1_track_ids()
        return [E.probe_body(t, split="validation") for t in ids]
    if group == "e1-mmlu":
        # Multi-track is allowed here but each track is still its own shard group,
        # so a missing receipt refuses that track alone rather than the sweep.
        #
        # The refusal has to be *caught* for that to be true.  Measured 2026-09-21:
        # `require_probe` raised for rel_c7ar_mamba_2p8b (no Q1 receipt yet) and
        # `group_bodies("e1-mmlu", None, ...)` exited -- so a direct `--group e1-mmlu`
        # built nothing at all for the other 26 tracks, which is the opposite of what
        # this comment claimed.  `lane_bodies` already had the try/except; it belongs
        # here, where the sweep is actually assembled.
        ids = [track] if track else q1_track_ids()
        bodies: list[dict] = []
        for t in ids:
            try:
                E.require_probe(t)
            except SystemExit as exc:
                print(f"# skip e1-mmlu/{t}: {str(exc)[:160]}", file=sys.stderr)
                continue
            bodies.extend(E.mmlu_group(t, num_shards=shards, split=split,
                                       gpus_per_pod=gpus_per_pod, tag=tag))
        check_group_shape(bodies, gpus_per_pod)
        return bodies
    if group == AUX_GROUP:
        sets = [dataset] if dataset else list(E.AUX_ORDER)
        bodies = []
        for name in sets:
            # One dataset at a time, all requested tracks: a *complete table* is the
            # unit the paper uses, so finishing HellaSwag across the arms beats one
            # arm across ten datasets.
            bodies.extend(aux_bodies(name, track, shards, gpus_per_pod, tag))
        return bodies
    if group in GEN_GROUPS:
        suite = GEN_GROUPS[group]
        n = shards if shards != DEFAULT_SHARDS else GEN_SHARDS[suite]
        ids = [track] if track else gen_track_ids(suite)
        bodies = []
        for t in ids:
            E.require_probe(t)
            bodies.extend(E.gen_group(t, suite, num_shards=n,
                                      gpus_per_pod=gpus_per_pod))
        check_group_shape(bodies, gpus_per_pod)
        return bodies
    raise SystemExit(f"unknown group {group!r}")


def gen_track_ids(suite: str) -> list[str]:
    """The tracks the generative lane runs, in priority order.

    Deliberately *not* all 27 Q1 tracks.  GSM8K/HumanEval+ at these scales sit at
    the floor for every arm we have measured (<=2.25 % / <=1.2 %), so a 12-way
    sweep buys 12 readings of the same floor.  The paper needs the floor
    *established* on the headline arms plus one released reference that is known
    non-zero (the AR bases score 0/0, LLaDA-8B 61.5/4.9 -- that contrast is the
    point), and the 0.4B matrix contributes a single seed per arm as a sanity
    check rather than a 3-seed mean of zeros.
    """
    ids = ["a29_c6loop_s20500", "a29_c6loop_s9500", "a29_c6_f2_s14000",
           "rel_c1_rwkv7_2p9b", "rel_c0_qwen25_3b", "rel_c0_llama32_3b",
           "a04_a1_s17", "a04_a2_s17", "a04_a3_s17", "a04_a5_s17",
           "rel_c1_rwkv7_0p4b"]
    known = {t.track_id for t in E.tracks.ALL_TRACKS}
    missing = [t for t in ids if t not in known]
    if missing:
        raise SystemExit(f"gen lane names tracks not in the registry: {missing}")
    return ids


#: The order the campaign drains its groups in.  This is a *claim about the paper*,
#: not convenience ordering: MMLU is the E1 headline table and the only lane whose
#: result is already partly on disk (the ladder), so it goes first.
#:
#: **GSM8K is not in this tuple** (operator direction 2026-09-21: "skip gsm8k, focus
#: on more basic ability").  It is kept in :data:`GEN_GROUPS` so a direct
#: ``--group e1-gsm8k`` still works and so the already-banked GSM8K floor readings
#: keep a live code path that can reproduce them -- but the drip never offers it, and
#: therefore never spends the 1-GPU project's 40-submitted quota on it.  That quota
#: is the binding constraint on the basic-ability lane: 168 MMLU shard names were
#: refused by CreateJob (not by a job failure) while 32 GSM8K pods held it.
DRIP_ORDER: Final = ("e1-mmlu", AUX_GROUP)

#: The pod shapes a drip tick may use, in fill order.  **Lanes are separate ceilings,
#: not one pool**: 1 bills ``PROJECT_1GPU`` and 8 bills ``PROJECT_SHARED`` at the 72 h
#: wall.  The 1-GPU lane is not a conservative fallback here -- 27 MMLU groups already
#: carry live rows at that shape and :func:`check_group_shape` correctly refuses to
#: re-shape them, so their remaining shards must be finished at 1 GPU or not at all.
#: It is filled first because it is the dedicated project (operator order 2026-09-20),
#: and every group it cannot serve falls through to the 8-GPU lane.
#:
#: ``SPEC_2GPU`` is deliberately **not** a lane any more.  It bills the same project as
#: the 8-GPU shape, so it is not extra capacity, and offering it before the 8-GPU lane
#: would shape-lock every new group at two cards -- exactly the outcome "use 8h100 for
#: each benchmark" rules out.  A 2-GPU body remains constructible
#: (:data:`SPEC_BY_GPUS`) as the declared memory fallback for a 64K shard that OOMs.
LANE_SHAPES: Final = (1, 8)

#: Groups the drip must not offer, with the reason.  Separate from ``DRIP_ORDER``
#: minus-a-name so that a future edit that re-adds a group has to delete a stated
#: reason rather than silently extend a tuple.
DRIP_EXCLUDED: Final = {
    "e1-gsm8k": "operator direction 2026-09-21: skip gsm8k, focus on basic ability",
    "e1-humanevalplus":
        "operator direction 2026-09-21: ignore the math/code related for now; the "
        "basic-ability lane and the long-context benchmarks have the priority",
    "e1-mbppplus":
        "operator direction 2026-09-21: ignore the math/code related for now; the "
        "basic-ability lane and the long-context benchmarks have the priority",
}


def drip(dry_run: bool = True) -> int:
    """One tick: settle what finished, then refill both project lanes.

    Each project enforces :data:`RUNNING_GPU_QUOTA` running and
    :data:`SUBMITTED_GPU_QUOTA` submitted, so a 392-shard campaign cannot be
    outstanding no matter what the operator authorized (the grant is permission to
    *spend*, not a quota raise).  A tick therefore reconciles first -- which is what
    frees the submitted-quota slots -- and only then submits.

    **The lanes are separate ceilings, not one pool.**  ``SPEC_1GPU`` bills
    ``PROJECT_1GPU`` and ``SPEC_2GPU`` bills ``PROJECT_SHARED``, so a capacity
    refusal in one lane says nothing about the other.  Operator order 2026-09-20:
    fill ``project-632c8db8`` (1-GPU) first, then take ``project-160ccb20`` (2-GPU)
    "for more".  A refusal therefore stops *its own lane* and moves on, where a
    single-lane breaker would have stopped the whole tick and left the second
    project's quota unused.

    A group already in flight at one shape stays at that shape
    (:func:`check_group_shape`): the 2-GPU lane picks up groups that have no live
    rows yet, so the two lanes never cover the same shard under two names.

    The remaining names are never written to the ledger, so a tick that submits
    nothing is not a failed tick: it means both queues are still full, and the same
    command resumes unchanged later.
    """
    if not dry_run:
        reconcile(dry_run=False)
    census = live_census()
    headroom = allowance_headroom(census)
    report = []
    for gpus_per_pod in LANE_SHAPES:
        # Built HERE, after the previous lane has submitted -- not up front. The
        # 1-GPU lane's submissions are what shape-lock those groups out of the 2-GPU
        # lane, so a pending list computed before it ran would still offer group X as
        # 2-GPU pods after X's shards went out as 1-GPU jobs: the same shard billed
        # twice under two names that name de-duplication cannot relate.
        pending: list[dict] = []
        for group in DRIP_ORDER:
            for body in lane_bodies(group, gpus_per_pod):
                if already_submitted(body["name"]) is None:
                    pending.append(body)
        sent, refused, stopped_on_allowance = 0, None, False
        for body in pending:
            if dry_run:
                continue
            if headroom < gpus_per_pod:
                # The operator's own ceiling, checked BEFORE CreateJob. Waiting for
                # a scheduler refusal here is what put 390 GPUs in flight once: the
                # second project's quota is much larger than the first's, so the
                # cluster never said no.
                stopped_on_allowance = True
                break
            row = submit_one(body, dry_run=False)
            if row["state"] == "submitted":
                sent += 1
                headroom -= gpus_per_pod
            if row.get("refusal_kind") == "capacity":
                # Stop THIS lane only. The other lane is a different project with
                # its own 16/40, so inferring "full" from this refusal would leave
                # sanctioned capacity idle.
                refused = row["name"]
                break
        report.append({
            "gpus_per_pod": gpus_per_pod,
            "project": E.PROJECT_1GPU if gpus_per_pod == 1 else E.PROJECT_SHARED,
            "pending_before": len(pending),
            "submitted_this_tick": sent,
            "stopped_on_capacity_at": refused,
            "stopped_on_own_allowance": stopped_on_allowance,
            "pending_after": len(pending) - sent,
        })
    print(json.dumps({
        "live_gpus_submitted_before": census["live_gpus_submitted"],
        "own_allowance": OWN_LIVE_GPU_ALLOWANCE,
        "headroom_at_start": allowance_headroom(census),
        "headroom_left": headroom,
        "lanes": report, "dry_run": dry_run,
    }, indent=2))
    return 0


def lane_bodies(group: str, gpus_per_pod: int) -> list[dict]:
    """The bodies of ``group`` at one pod shape, skipping what the shape locks out.

    ``check_group_shape`` raises for a group already in flight at the other shape.
    That refusal is correct and load-bearing for a directly-named group, but inside
    a drip tick it must not abort the tick: the honest reading is "this group
    belongs to the other lane", so it is skipped per track rather than raised.

    MMLU is special-cased to the wide shape under :data:`E.TAG_8GPU`.  Operator
    direction 2026-09-21 ("use 8h100 for each benchmark") applies to the E1 headline
    table first, but 27 ``e1`` groups already hold live rows at 1 or 2 GPU/pod and the
    shape lock forbids moving them, so the wide lane opens a *new* group namespace and
    covers exactly the tracks that have no merged reading yet.  Re-running the 21
    tracks that already merged would cost ~170 GPU-h to re-derive identical numbers.
    """
    if group == "e1-mmlu":
        if gpus_per_pod != 8:
            # The narrow MMLU lane is retired, not merely deprioritised: every track
            # its 48 pending 1-GPU shards would cover is offered by the wide lane
            # instead, and that lane bills project-632c8db8 -- the one the excluded
            # GSM8K suite is still saturating (36 of its 40 submitted GPUs).
            return []
        ids = [t for t in q1_track_ids() if not E.merged_present(t)]
        bodies: list[dict] = []
        for track in ids:
            try:
                E.require_probe(track)
                bodies.extend(group_bodies(group, track, DEFAULT_SHARDS, "test",
                                           gpus_per_pod, tag=E.TAG_8GPU))
            except SystemExit as exc:
                print(f"# skip {group}/{track} at {gpus_per_pod} GPU/pod: "
                      f"{str(exc)[:160]}", file=sys.stderr)
        return bodies
    if group == AUX_GROUP:
        if gpus_per_pod != 8:
            # Same reasoning as the narrow MMLU lane: the trees and the wide shape are
            # what the operator asked for, and a 1-GPU pod would bill the project the
            # MMLU lane has already retired.
            return []
        bodies: list[dict] = []
        # `AUX_TRACKS_DRIP`, not `E.aux_track_ids()`: the drip fills the panels with the
        # manuscript's own rows so the remaining datasets finish in half the wall clock,
        # while a direct `--group e1-basic` still builds the full 25-track sweep. The
        # item counts are untouched either way -- see the tuple's own comment.
        drip_ids = E.aux_track_ids()
        keep = [t for t in E.AUX_TRACKS_DRIP if t in drip_ids]
        for dataset in E.AUX_ORDER:
            for track in keep:
                if aux_merged(dataset, track):
                    continue
                bodies.extend(aux_bodies(dataset, track, DEFAULT_SHARDS, 8))
        return bodies
    ids = gen_track_ids(GEN_GROUPS[group])
    bodies = []
    for track in ids:
        try:
            bodies.extend(group_bodies(group, track, DEFAULT_SHARDS, "test",
                                       gpus_per_pod))
        except SystemExit as exc:
            # A shape lock, a missing Q1 receipt or an unloadable layout all refuse
            # one track without saying anything about the rest of the sweep.
            print(f"# skip {group}/{track} at {gpus_per_pod} GPU/pod: "
                  f"{str(exc)[:160]}", file=sys.stderr)
    return bodies


#: The groups that survive a cut, most valuable first.  This is a claim about the
#: paper, not about cost: MMLU on the headline 2.9B arms plus the released C1
#: reference is the E1 table the manuscript is built around, and the ladder steps
#: and 0.4B seed matrix are supporting rows that can be re-submitted on a later
#: tick.  ``stop_excess`` keeps as many of these as the allowance holds and gives
#: everything else back.
KEEP_PRIORITY: Final = (
    "lrwkv-e1-mmlu-test-a29-c6loop-s20500",
    "lrwkv-e1-mmlu-test-a29-c6loop-s9500",
    "lrwkv-e1-mmlu-test-a29-c6-f2-s14000",
    "lrwkv-e1-mmlu-test-rel-c1-rwkv7-2p9b",
    "lrwkv-e1-mmlu-test-a29-c6-f2-s13000",
    "lrwkv-e1-mmlu-test-rel-c1-rwkv7-0p4b",
    "lrwkv-e1-mmlu-test-a04-a2-s17",
    "lrwkv-e1-mmlu-test-a04-a1-s17",
)


def keep_rank(group: str, running: int) -> tuple:
    """Sort key: lower is kept.  Paper priority first, then progress already spent.

    Within the unlisted groups a *running* group outranks a queuing one: a queuing
    job has spent nothing, so it is the cheapest thing to give back, while stopping a
    running group throws away GPU-hours already paid for.
    """
    try:
        return (0, KEEP_PRIORITY.index(group), group)
    except ValueError:
        return (1, 0 if running else 1, group)


def stop_job(job_id: str) -> tuple[bool, str]:
    """Stop one job, retrying only the transient throttle.

    Measured 2026-09-21: a 33-job sweep of ``--stop-excluded`` had *every* call return
    ``API error 429: ... 429 Too Many Requests`` from the gateway.  That is throttling,
    not a refusal -- the jobs were still live afterwards and the same calls succeed when
    spaced out.  Treating it as a failure is the safe default (no ``stopped`` ledger row
    is written, so the group stays resubmittable), but it also means a sweep silently
    frees nothing, which reads as "there was nothing to stop".

    A *quota* refusal is deliberately NOT retried -- looping on one is a standing
    operator rule -- and it does not arrive as 429.  The retry is bounded so a genuinely
    down gateway costs seconds, not minutes.
    """
    tail = ""
    for attempt in range(4):
        result = subprocess.run([QZ, "train", "StopJob", "--job-id", job_id, "-o", "json"],
                                capture_output=True, text=True)
        tail = ((result.stdout or "") + (result.stderr or ""))[-300:]
        if "429" not in tail:
            return result.returncode == 0 and "Error" not in tail, tail
        time.sleep(2.0 * (attempt + 1))
    return False, f"still throttled after 4 attempts: {tail}"


#: Seconds between consecutive StopJob calls.  The gateway's limit is on request *rate*,
#: so the backoff in :func:`stop_job` alone still trips it: 33 calls issued as fast as
#: the API answers hit 429 on the first one.
STOP_SPACING_S: Final = 1.5


def group_of(name: str) -> str:
    m = TOPOLOGY_SUFFIX.match(name)
    return m.group("group") if m else name


def stop_excess(dry_run: bool = True) -> int:
    """Bring live GPUs back under the operator's allowance by stopping whole groups.

    **Whole groups, never individual shards.**  A shard group only produces a result
    when all ``NUM_SHARDS`` files exist -- the merge gate waits for the count -- so
    stopping half a group converts it from "will finish" into "will spend its
    remaining shards' GPU-hours and still never merge".  Stopping the group instead
    leaves every one of its names re-submittable on a later tick.

    What survives is decided by :func:`keep_rank`: the paper's headline E1 groups
    first, then running work over queuing work.

    This exists because of a real overshoot on 2026-09-20: 390 GPUs in flight against
    a 64-GPU allowance, because the drip trusted a scheduler refusal as its brake and
    the second project's quota never refused.
    """
    census = live_census()
    over = census["live_gpus_submitted"] - OWN_LIVE_GPU_ALLOWANCE
    groups: dict[str, dict] = {}
    for job in census["jobs"]:
        g = groups.setdefault(group_of(job["name"]), {"gpus": 0, "jobs": [],
                                                      "running": 0})
        g["gpus"] += job["gpu_count"]
        g["jobs"].append(job)
        g["running"] += 1 if job["status"] == "job_running" else 0
    # Keep in priority order while the allowance holds; everything else is dropped.
    # Deciding by "keep" rather than by "drop" is what makes a whole group the unit:
    # a group is either entirely inside the budget or entirely given back.
    ranked = sorted(groups.items(), key=lambda kv: keep_rank(kv[0], kv[1]["running"]))
    kept_gpus, kept, dropped = 0, [], []
    for name, g in ranked:
        if kept_gpus + g["gpus"] <= OWN_LIVE_GPU_ALLOWANCE:
            kept_gpus += g["gpus"]
            kept.append({"group": name, "gpus": g["gpus"]})
        else:
            dropped.append({"group": name, "gpus": g["gpus"],
                            "jobs": len(g["jobs"]), "running": g["running"]})
    freed = sum(d["gpus"] for d in dropped)
    stopped, failed = 0, []
    for entry in dropped:
        if dry_run:
            continue
        for job in groups[entry["group"]]["jobs"]:
            time.sleep(STOP_SPACING_S)
            ok, tail = stop_job(job["job_id"])
            if ok:
                stopped += 1
                append_ledger({"state": "stopped", "name": job["name"],
                               "job_id": job["job_id"], "at": timestamp(),
                               "reason": "over own 64-GPU allowance; whole group "
                                         "stopped so every shard stays resubmittable",
                               "source": "stop_excess"})
            else:
                failed.append({"name": job["name"], "tail": tail})
    print(json.dumps({
        "live_gpus_submitted": census["live_gpus_submitted"],
        "live_gpus_running": census["live_gpus_running"],
        "own_allowance": OWN_LIVE_GPU_ALLOWANCE,
        "over_by": max(0, over),
        "groups_live": len(groups),
        "groups_kept": kept,
        "gpus_kept": kept_gpus,
        "groups_dropped": len(dropped),
        "gpus_freed": freed,
        "jobs_stopped": stopped,
        "stop_failures": failed,
        "dry_run": dry_run,
    }, indent=2))
    return 0


def stop_excluded(dry_run: bool = True) -> int:
    """Stop live groups whose suite :data:`DRIP_EXCLUDED` no longer wants.

    Separate from :func:`stop_excess`, which enforces the operator's *GPU ceiling*.
    This enforces the operator's *priority*: after 2026-09-21's "skip gsm8k, focus on
    more basic ability", 32 live GSM8K pods were holding the 1-GPU project's
    40-submitted quota while 168 MMLU shard names sat refused by CreateJob.  Leaving
    them to drain would have kept the basic-ability lane blocked for hours for a
    suite the operator dropped.

    The shard group is the unit for the same reason as in :func:`stop_excess`: a
    half-stopped group spends its remaining shards' GPU-hours and still never
    merges.  Here the group is being abandoned rather than deferred, so that
    argument is about the *cost* of the survivors, not about resubmittability -- an
    excluded group is not meant to come back, and the ``stopped`` rows say why.
    """
    census = live_census()
    groups: dict[str, dict] = {}
    for job in census["jobs"]:
        g = groups.setdefault(group_of(job["name"]), {"gpus": 0, "jobs": []})
        g["gpus"] += job["gpu_count"]
        g["jobs"].append(job)
    # A group name is "lrwkv-<tag>-<suite>-<track>..."; the excluded keys are
    # "e1-<suite>", so the match is on the name's own prefix rather than on a
    # re-derived suite string that could drift from the emitter's spelling.
    targets = {}
    for name, g in groups.items():
        for group_key, reason in DRIP_EXCLUDED.items():
            if name.startswith(f"lrwkv-{group_key}-"):
                targets[name] = (g, reason)
                break
    stopped, failed = 0, []
    freed = sum(g["gpus"] for g, _ in targets.values())
    for name, (g, reason) in sorted(targets.items()):
        if dry_run:
            continue
        for job in g["jobs"]:
            time.sleep(STOP_SPACING_S)
            ok, tail = stop_job(job["job_id"])
            if ok:
                stopped += 1
                append_ledger({"state": "stopped", "name": job["name"],
                               "job_id": job["job_id"], "at": timestamp(),
                               "reason": f"suite dropped from the drip: {reason}",
                               "source": "stop_excluded"})
            else:
                failed.append({"name": job["name"], "tail": tail})
    print(json.dumps({
        "live_gpus_submitted": census["live_gpus_submitted"],
        "excluded_groups": DRIP_EXCLUDED,
        "groups_matched": sorted(targets),
        "gpus_freed": freed,
        "jobs_stopped": stopped,
        "stop_failures": failed,
        "dry_run": dry_run,
    }, indent=2))
    return 0


def status() -> int:
    rows = read_ledger()
    by_name: dict[str, dict] = {}
    for row in rows:
        if row.get("name"):
            by_name[row["name"]] = row
    counts: dict[str, int] = {}
    for row in by_name.values():
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    probes = {t: E.probe_passed(t) for t in q1_track_ids()}
    report = {
        "ledger": str(LEDGER),
        "ledger_exists": LEDGER.exists(),
        "rows": len(rows),
        "distinct_names": len(by_name),
        "state_counts": counts,
        "q1_tracks": len(probes),
        "q1_receipts_present": sorted(t for t, (ok, _) in probes.items() if ok),
        "q1_receipts_missing": sorted(t for t, (ok, _) in probes.items() if not ok),
        "no_loader_reported_absent": sorted(
            t.track_id for t in E.tracks.ALL_TRACKS
            if t.model_kind == E.tracks.NO_LOADER),
        "llada_separate_lane": sorted(
            t.track_id for t in E.tracks.ALL_TRACKS
            if t.model_kind == E.tracks.LLADA_MODEL_KIND),
        "missing_runtime_dep": sorted(
            t.track_id for t in E.tracks.ALL_TRACKS
            if t.model_kind == E.tracks.MISSING_RUNTIME_DEP),
    }
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--authorize", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reconcile", action="store_true",
                    help="ask the scheduler for each blocking row's job and record "
                         "the terminal ones; needs --submit to write")
    ap.add_argument("--drip", action="store_true",
                    help="one campaign tick: reconcile, then refill the queue up to "
                         "the project quota; needs --submit to write")
    ap.add_argument("--stop-excess", action="store_true",
                    help="stop whole shard groups until live GPUs are back under "
                         f"the operator's {OWN_LIVE_GPU_ALLOWANCE}-GPU allowance; "
                         "needs --submit to write")
    ap.add_argument("--stop-excluded", action="store_true",
                    help="stop live groups of suites the drip no longer offers "
                         f"({', '.join(DRIP_EXCLUDED)}); they hold project quota "
                         "the remaining lanes are being refused for. Needs --submit")
    ap.add_argument("--census", action="store_true",
                    help="how many GPUs this campaign holds right now")
    ap.add_argument("--group", choices=("q1probe", "e1-mmlu", AUX_GROUP, *GEN_GROUPS))
    ap.add_argument("--track")
    ap.add_argument("--dataset",
                    help=f"one {AUX_GROUP} tree to build; default is all of "
                         f"{list(E.AUX_ORDER)}")
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    ap.add_argument("--tag", default="e1",
                    help="the job-name/outdir namespace. Changing it opens a NEW "
                         "shard group, which is the sanctioned way to re-run a "
                         "benchmark at a wider pod shape: `check_group_shape` "
                         "correctly refuses to re-shape a group with live rows, and "
                         "re-tagging leaves the old rows and their partial output "
                         "untouched as provenance instead of overwriting them")
    ap.add_argument("--gpus-per-pod", type=int, default=1, choices=(1, 2, 4, 8),
                    help="pod width. 1 bills project-632c8db8; 2, 4 and 8 all bill "
                         "project-160ccb20 (4 rides the 8-GPU quota and leaves four "
                         "cards idle). The quota is per project, so the wide shapes "
                         "are additional capacity, not just bigger pods")
    ap.add_argument("--split", default="test", choices=("test", "validation"))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--max-submit", type=int, default=0,
                    help="stop after this many successful CreateJob calls; the "
                         "unattempted names stay out of the ledger, so the same "
                         "command resumes where it stopped")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.authorize:
        return authorize()
    if args.census:
        c = live_census()
        c.pop("jobs")
        c["own_allowance"] = OWN_LIVE_GPU_ALLOWANCE
        c["headroom"] = max(0, OWN_LIVE_GPU_ALLOWANCE - c["live_gpus_submitted"])
        print(json.dumps(c, indent=2))
        return 0
    if args.stop_excess:
        return stop_excess(dry_run=not args.submit)
    if args.stop_excluded:
        return stop_excluded(dry_run=not args.submit)
    if args.reconcile:
        return reconcile(dry_run=not args.submit)
    if args.drip:
        return drip(dry_run=not args.submit)
    if args.status or not args.group:
        return status()

    bodies = group_bodies(args.group, args.track, args.shards, args.split,
                          args.gpus_per_pod, args.tag, args.dataset)
    if not args.submit:
        args.dry_run = True
    print(f"# {args.group}: {len(bodies)} job(s), "
          f"{'DRY RUN' if args.dry_run else 'SUBMITTING'}"
          + (f", capped at {args.max_submit}" if args.max_submit else ""))
    results, sent = [], 0
    for body in bodies:
        if args.max_submit and sent >= args.max_submit:
            print(f"# stopping at --max-submit {args.max_submit}; "
                  f"{len(bodies) - len(results)} name(s) left for the next tick "
                  f"(they are untouched in the ledger, so they are not lost)")
            break
        row = submit_one(body, args.dry_run)
        results.append(row)
        if row["state"] == "submitted":
            sent += 1
        if row.get("refusal_kind") == "capacity":
            # Stop the whole sweep on the first capacity refusal.  Continuing would
            # write hundreds of `failed` rows for bodies that were never wrong, and
            # looping on the parent-project quota message is a standing rule.
            print(f"# CAPACITY REFUSAL on {row['name']}; stopping the sweep with "
                  f"{len(bodies) - len(results)} name(s) unattempted. Re-run this "
                  f"same command later -- unattempted names are not in the ledger.",
                  file=sys.stderr)
            break
    ok = sum(1 for r in results if r["state"] in ("submitted", "dry_run"))
    skipped = sum(1 for r in results if r["state"] == "skipped")
    print(f"# {ok}/{len(results)} {'rendered' if args.dry_run else 'submitted'}"
          f"{f', {skipped} already done' if skipped else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
