"""The adaptation / released-checkpoint track registry for the Long-RWKV paper.

The manuscript's controlled track (App. A, 48 rows of ``training_run_matrix.csv``)
trains eight arms from scratch at 350M and 1.3B.  None of those runs exist, and
this registry does **not** claim them: every row there stays ``not_run``.  What
does exist is the two lineages App. A G3 explicitly sets aside --

    "The large pretrained lineage is a separate adaptation track because its
     initialization and actual parameter count differ from the controlled models"

-- plus the released checkpoints the paper's practical-comparison track names.
This module is the single place those checkpoints are described, so every number
the paper reports can be traced to a content hash rather than to a path that may
be pruned (``train.save_checkpoint`` keeps only the last three).

Two facts here are load-bearing and easy to get wrong, so they are recorded as
data rather than left to prose:

``params_stored`` vs ``params_executed``
    A ``BiRWKV7ForMaskedDiffusion`` stores **both** direction's mixers, so a
    *forward-only* arm (a1/a5, M6) still carries its backward tensors on disk and
    in the optimizer.  The paper's own counting rule ("directional copies count")
    makes ``params_stored`` the number to report, but an arm that never calls the
    backward mixer does not *execute* those parameters, and a reader comparing a
    forward-only arm against a bidirectional one at "equal parameters" is
    comparing two different computations.  Both numbers are therefore stored, and
    :func:`check_directional_accounting` refuses a forward-only entry that claims
    they are equal.

``tokens_seen`` is adaptation exposure, not total exposure
    Every entry here is initialized from a *released* RWKV-7 checkpoint whose own
    pretraining exposure is unknown to us.  ``tokens_seen`` counts only the tokens
    this project's run consumed.  A C6-vs-C1 contrast built from these rows is
    therefore "adaptation vs released", never an equal-exposure comparison, and
    :data:`EXPOSURE_DISCLOSURE` is what the tex must say beside any such number.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Final

G: Final = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222")
"""The pod-visible root.  ``/inspire/hdd/project`` and ``/inspire/qb-ilm2`` are
login-node only, so a checkpoint that is not under this root cannot be evaluated
by a qz job at all -- which is why ``f2 step 14000`` was copied here."""

EXPOSURE_DISCLOSURE: Final = (
    "adaptation track: initialized from a released RWKV-7 checkpoint whose "
    "pretraining exposure is not matched or known to this study; tokens_seen "
    "counts adaptation tokens only"
)

# --------------------------------------------------------------------------
# Arm mapping.
#
# The paper's arm ids describe (mixer, objective, direction).  The checkpoints
# were trained under this project's own labels.  The mapping below is by the
# three fields protocol.json["arms"] actually declares -- not by name similarity,
# because two of this project's labels are actively misleading:
#
#  * rwkv04b "A0" trained the *masked* objective (parameter_identity.json carries
#    arm_relabelled=True), so it is M6/C5, never C1.  The registry reads the
#    recorded relabel rather than the launch label.
#  * task7 "a1"/"a5" pass --force-forward, so they are C5 (direction=forward)
#    even though the stored tensors are bidirectional.
#
# The tied depth loop (a3/a5, m4loop) is NOT a paper arm: no protocol arm has a
# recycled-depth field.  Those entries carry loop_reps>0 and are reported as a
# declared deviation from C6/C5, which is what ``variant`` records.
# --------------------------------------------------------------------------

PAPER_ARMS: Final = {
    "C0": ("softmax", "autoregressive", "causal"),
    "C1": ("rwkv7", "autoregressive", "causal"),
    "C2": ("softmax", "absorbing_diffusion", "bidirectional"),
    # C3/C4 are declared in protocol.json["arms"] but were absent here, so any row
    # naming them was refused by validate_tracks.  They are the two mixers the
    # released state/linear-attention baselines actually instantiate, which is why
    # the gap only surfaced when those baselines were added (2026-09-21).
    "C3": ("additive_kernel", "absorbing_diffusion", "bidirectional"),
    "C4": ("scalar_delta", "absorbing_diffusion", "bidirectional"),
    "C5": ("rwkv7", "absorbing_diffusion", "forward"),
    "C6": ("rwkv7", "absorbing_diffusion", "bidirectional"),
    "C7": ("mamba2", "absorbing_diffusion", "bidirectional"),
    # The AR counterparts of C3/C4/C7.  A released Mamba/GLA/RWKV-6 checkpoint is
    # autoregressive, so filing it under the bidirectional-diffusion arm id would
    # assert an objective and a direction it does not have -- the same error the
    # module docstring records for rwkv04b "A0".  These ids carry
    # ``derived_from``/``deviation`` in :data:`BASELINE_ARMS` for protocol_patch.
    "C3_ar": ("additive_kernel", "autoregressive", "causal"),
    "C4_ar": ("scalar_delta", "autoregressive", "causal"),
    "C7_ar": ("mamba2", "autoregressive", "causal"),
    "C8_ar": ("rwkv6", "autoregressive", "causal"),
}

#: The arm ids above that ``protocol.json`` does not yet declare, with the fields
#: ``validate_protocol.py:50`` ("Unknown model arm") needs.  Mirrors the
#: ``NEW_ARMS``/``patch_protocol`` pattern in :mod:`lrwkv_evidence.protocol_patch`:
#: the paper's own arm set is never edited in place, it is *extended* with entries
#: that say what they deviate from.
BASELINE_ARMS: Final = {
    "C3_ar": {
        "mixer": "additive_kernel", "objective": "autoregressive",
        "direction": "causal", "derived_from": "C3",
        "deviation": "released autoregressive checkpoint, not trained under the "
                     "absorbing-diffusion objective C3 declares; enters as a "
                     "long-context reference for the gated-linear-attention mixer "
                     "only, with unmatched pretraining exposure",
    },
    "C4_ar": {
        "mixer": "scalar_delta", "objective": "autoregressive",
        "direction": "causal", "derived_from": "C4",
        "deviation": "released autoregressive checkpoint, not trained under the "
                     "absorbing-diffusion objective C4 declares",
    },
    "C7_ar": {
        "mixer": "mamba2", "objective": "autoregressive", "direction": "causal",
        "derived_from": "C7",
        "deviation": "released autoregressive checkpoint, not trained under the "
                     "absorbing-diffusion objective C7 declares. NOTE the mixer "
                     "field is the protocol's label for the selective-SSM family; "
                     "mamba-2.8b is Mamba-1 (S6), not Mamba-2/SSD",
    },
    "C8_ar": {
        "mixer": "rwkv6", "objective": "autoregressive", "direction": "causal",
        "derived_from": None,
        "deviation": "no protocol arm declares the RWKV-6 mixer (data-dependent "
                     "decay without the RWKV-7 delta-rule state update); it is the "
                     "immediate predecessor of the C1/C5/C6 mixer and enters as the "
                     "within-family scaling reference",
    },
}

UNMEASURED: Final = -1
"""Sentinel for a count this study has not measured.

It is deliberately not ``0`` and not ``None``: a zero reads as a measurement
("this checkpoint has no parameters"), and a ``None`` in a CSV cell is
indistinguishable from an empty field the writer forgot to fill.  Every consumer
must therefore branch on it explicitly, and no arithmetic here treats it as a
number."""

#: The ``--model_kind`` values ``eval/capability/run_eval.py`` actually accepts,
#: read from its argparse ``choices`` rather than from prose.  ``relay`` is the
#: frozen-renderer RELAY format and appears in no track here.
RUN_EVAL_MODEL_KINDS: Final = frozenset(
    {"relay", "birwkv_diffusion", "hf_causal", "fla_causal"})

#: LLaDA does not go through ``run_eval.py`` at all: it has its own entrypoint,
#: ``eval/capability/llada_batch_eval.py``, whose flags are ``--mode`` (cola|math|
#: code) / ``--model_dir`` / ``--tasks_file`` / ``--block_length`` / ``--steps``,
#: not run_eval's.  Recording it as ``hf_causal`` would name a flag set that does
#: not exist, so it gets its own kind and the emitter must branch on it.
LLADA_MODEL_KIND: Final = "llada_batch"

#: A checkpoint present on disk that nevertheless has no loader in this harness.
#: Both current members were checked, not assumed: ``models/sedd-medium/config.json``
#: and ``models/mamba2-2.7b/config.json`` carry **no** ``model_type`` and no
#: ``auto_map``, so ``AutoModelForCausalLM.from_pretrained`` cannot build either --
#: which is exactly the refusal ``efficiency/adapters.py:600-602`` already names
#: ("no model_type (mamba2) needs its own loader").  ``grep -rl sedd`` over the
#: whole eval tree returns nothing, and ``mamba_ssm`` is absent from the relay2:v2
#: image.  Writing ``hf_causal`` on these rows would have booked two baselines the
#: campaign cannot run; they are declared unevaluable so the paper reports them as
#: absent rather than pending.
NO_LOADER: Final = "no_loader"

#: The ``outputs_rwkv04b`` lineage is a **different architecture**, not a
#: different container.  Measured on five probe jobs (2026-09-20) that each paid
#: a full load before dying with
#: ``RuntimeError: checkpoint/model mismatch: ['layers.0.attn_fwd.x_r', ...]``.
#:
#: ``scale/models/birwkv7_diffusion.py:669-670`` builds **two untied** mixers per
#: block (``attn_fwd`` and ``attn_bwd``), so its state dict names both.  The r04
#: trainer's ``longrwkv/model.py:120`` ``TiedBiRWKV7Block`` builds **one** mixer
#: (``attn``) and calls it twice -- forward, then per-document reversed -- fusing
#: the two passes.  Verified on disk: ``r04_M6_s29`` has 867 tensors with
#: ``layers.0.attn.*`` and no ``attn_fwd``/``attn_bwd`` key at all, and the A1/A3
#: arms additionally carry ``input_cond.*`` / ``gate_mods.*`` modules the scale
#: model has no slot for.
#:
#: So ``convert_rwkv04b`` was necessary but not sufficient: it fixed the bundle
#: shape (weights nested under ``"model"``) and the derived ``step_*/model.pt``
#: dirs are correct, but the *key layout* still cannot map onto the scale model.
#: These checkpoints ARE evaluable -- through their own harness
#: (``rwkv04b/longrwkv/eval/run_shard.py:load_model``, which builds
#: ``LongRWKV.from_hf_pretrained(ckpt, config, arm=...)``) -- which is the E3
#: long-context lane, not the scale capability lane.  That harness has no
#: multichoice/MMLU path (``grep -rl mmlu rwkv04b/longrwkv`` is empty), so the
#: honest statement is: no E1 for this lineage, E3 via its own runner.
FOREIGN_ARCH: Final = "longrwkv_tied"

#: A checkpoint whose *config* dispatches fine but whose runtime dependency is
#: absent from the ``relay2:v2`` image.  :data:`NO_LOADER` was not enough: it is
#: defined by "no ``model_type`` and no ``auto_map``", and both members here have
#: one, so :func:`check_hf_causal_is_loadable` -- which only reads config.json --
#: would have passed them straight into the campaign as ``hf_causal``.
#:
#: Measured on the login node 2026-09-21 (same image versions as the pod:
#: python 3.12.3, torch 2.8.0a0+nv25.06, transformers 5.3.0, fla 0.5.0):
#:
#: * ``models/rwkv6-world-3b`` declares ``auto_map`` onto its bundled
#:   ``modeling_rwkv6.py``, and ``AutoModelForCausalLM.from_pretrained`` raises
#:   ``ImportError: This modeling file requires the following packages that were
#:   not found in your environment: bitsandbytes`` -- from
#:   ``dynamic_module_utils.check_imports``, which AST-scans the remote file
#:   *before* any weight is read, so no ``try/except`` in that file can soften it.
#:   Independently fatal: the file's line 41 imports
#:   ``fla.ops.rwkv6.recurrent_fuse``, a module that does not exist in fla 0.5.0
#:   (the symbol lives in ``fla.ops.rwkv6.fused_recurrent``), inside a bare
#:   ``try/except ImportError`` that only *prints* -- so ``fused_recurrent_rwkv6``
#:   would be an unbound name at the first forward, i.e. a crash after the load is
#:   paid for.  Two independent blockers, neither fixable from our side.
#: * ``hf_cache/hub/models--fla-hub--gla-1.3B-100B`` is a ``model_type: gla``
#:   checkpoint with fla-native keys (``model.layers.0.attn.g_proj.weight``).
#:   ``AutoConfig.from_pretrained`` raises ``ValueError: ... model type `gla` but
#:   Transformers does not recognize this architecture``, because ``gla`` is
#:   registered by ``import fla.models`` (``fla/models/gla/__init__.py:13``) and
#:   **no file in the eval harness imports fla at all** (``grep -rn 'import fla'``
#:   over ``eval/capability/`` is empty; ``fla/__init__.py`` is 4 lines and pulls
#:   in no submodule).  This one is *reachable* with a loader that imports
#:   ``fla.models`` first -- it is recorded as needing that code, not as absent.
#:
#: The distinction matters for the paper: :data:`NO_LOADER` rows are reported as
#: absent baselines, whereas these are reported as absent *for a stated
#: environment reason*, which is a different disclosure.
MISSING_RUNTIME_DEP: Final = "missing_runtime_dep"

MODEL_KINDS: Final = RUN_EVAL_MODEL_KINDS | {
    LLADA_MODEL_KIND, NO_LOADER, FOREIGN_ARCH, MISSING_RUNTIME_DEP}

#: The kinds ``run_eval.py``'s capability lane cannot schedule, as one name.
#:
#: This existed as a hardcoded ``(NO_LOADER, LLADA_MODEL_KIND, FOREIGN_ARCH)``
#: tuple in four places -- ``campaign.q1_track_ids`` and three tests -- and adding
#: :data:`MISSING_RUNTIME_DEP` on 2026-09-21 broke exactly one of them, which is
#: how the duplication showed itself: the Q1 order kept offering a track the
#: emitter refuses, so the disagreement would have surfaced one job at a time at
#: submit. Derived from :data:`RUN_EVAL_MODEL_KINDS` rather than listed, so a kind
#: added above is excluded here by construction and a *new schedulable* kind has
#: to be added to run_eval's own choices to become schedulable.
UNSCHEDULABLE_KINDS: Final = frozenset(MODEL_KINDS - RUN_EVAL_MODEL_KINDS)


@dataclasses.dataclass(frozen=True)
class Track:
    """One evaluable checkpoint and everything a record must cite about it."""

    track_id: str
    #: ``adaptation_2p9b`` | ``adaptation_0p4b`` | ``released``
    track: str
    #: The paper arm this checkpoint instantiates, by (mixer, objective, direction).
    paper_arm: str
    #: This project's own label, kept so a job log can be traced back.
    local_label: str
    #: Path to the weights, relative to :data:`G`.  A directory for the 2.9B/task7
    #: flat-``model.pt`` layout, a ``.pt`` file for the rwkv04b layout.
    rel_path: str
    #: The HF dir the geometry is built from (and the AR reference for C1).
    model_dir: str
    #: sha256 of the weight file.  Empty only for a released HF dir (many files;
    #: the dir's own manifest is the pin).
    sha256: str
    #: Parameters present in the checkpoint, both directions.  ``-1`` means "not
    #: measured": the rwkv04b payload bundles the optimizer, so its file size does
    #: not yield a parameter count, and its ``parameter_identity.json`` reports the
    #: executed count only.  A zero would read as a measurement of nothing.
    params_stored: int
    #: Parameters the arm's forward pass actually calls.  For a forward-only arm
    #: this is strictly less than ``params_stored``; see the module docstring.
    params_executed: int
    #: Adaptation tokens only, never total exposure.  ``-1`` for a released
    #: checkpoint, whose pretraining budget is not known to this study.
    tokens_seen: int
    training_seed: int | None
    step: int | None
    loop_reps: int
    #: A one-line note on how this entry departs from its paper arm, or "" if it
    #: is a faithful instance.  Never empty for a loop entry.
    variant: str
    #: One of :data:`MODEL_KINDS` -- the loader this checkpoint must be evaluated
    #: through.  A wrong value here is a silent loader swap, which is why it is
    #: data, checked against the harness's own argparse choices, and not inferred
    #: from the path.
    model_kind: str
    notes: str = ""

    @property
    def path(self) -> Path:
        return G / self.rel_path

    @property
    def ckpt_arg(self) -> str:
        """What ``--ckpt_dir`` must receive (a dir for flat layouts)."""
        return str(self.path)


def _t(**kw) -> Track:
    return Track(**kw)


# --------------------------------------------------------------------------
# 2.9B lineage (nominal "2.9B"; 4,091,581,440 stored bidirectional parameters).
# Single seed -- these runs predate the paper and were never replicated, so they
# cannot enter the validated run matrix and are a descriptive table instead.
# --------------------------------------------------------------------------

TRACKS_2P9B: Final = [
    _t(track_id="a29_c6loop_s20500", track="adaptation_2p9b", paper_arm="C6",
       local_label="m4-loop-2p9b step 20500",
       rel_path="outputs_birwkv_diffusion/m4-loop-2p9b/step_00020500",
       model_dir="models/RWKV7-Goose-World3-2.9B-HF",
       sha256="60b287e830f1f5697c9c8c3bde659be45138d47f535e31786784b14cf21cb139",
       params_stored=4_091_581_443, params_executed=4_091_581_443,
       tokens_seen=21_495_808_000, training_seed=None, step=20500, loop_reps=1,
       variant="tied depth loop over layers [16,32) with one extra rep; no "
               "protocol arm declares recycled depth, so this is C6 plus a "
               "declared deviation, and its executed block passes exceed C6's",
       model_kind="birwkv_diffusion",
       notes="the paper's 'Long-RWKV core' adaptation endpoint"),
    _t(track_id="a29_c6loop_s9500", track="adaptation_2p9b", paper_arm="C6",
       local_label="m4loop_ext_endpoint_ckpt step 9500",
       rel_path="m2_baseline_triangle/m4loop_ext_endpoint_ckpt",
       model_dir="models/RWKV7-Goose-World3-2.9B-HF",
       sha256="99a388e699c92cbcb654497aacc23ae43ae7d69286981cab0ce92f405d9b5612",
       params_stored=4_091_581_443, params_executed=4_091_581_443,
       tokens_seen=9_961_472_000, training_seed=None, step=9500, loop_reps=1,
       variant="tied depth loop over layers [16,32), one extra rep (see s20500)",
       model_kind="birwkv_diffusion",
       notes="the checkpoint the paper's App. 'historical, non-confirmatory' "
             "OBQA 56.60 / RACE 55.40 readings were taken on"),
    _t(track_id="a29_c6_f2_s14000", track="adaptation_2p9b", paper_arm="C6",
       local_label="f2-2p9b-ptcorpora step 14000",
       rel_path="research/lacesmm_assets/birwkv_f2_step14000",
       model_dir="models/RWKV7-Goose-World3-2.9B-HF",
       sha256="cb45226feb6c0dc831cb5b51aedc57067ea3f80e4a28b030d1a2485010f9bf19",
       params_stored=4_091_581_440, params_executed=4_091_581_440,
       tokens_seen=14_680_064_000, training_seed=None, step=14000, loop_reps=0,
       variant="", model_kind="birwkv_diffusion",
       notes="pre-loop bidirectional denoiser: the faithful C6 instance at 2.9B. "
             "Copied from /inspire/qb-ilm2 (login-node only) to G on 2026-09-20 "
             "so a pod can read it"),
    _t(track_id="a29_c6_f2_s13000", track="adaptation_2p9b", paper_arm="C6",
       local_label="f2-2p9b-ptcorpora step 13000",
       rel_path="research/lacesmm_assets/birwkv_f2_step13000",
       model_dir="models/RWKV7-Goose-World3-2.9B-HF",
       sha256="70f1e0860ba93c0a8279db884945a24ab716ff33d3dbfd8b8526f8d8af86eab7",
       params_stored=4_091_581_440, params_executed=4_091_581_440,
       tokens_seen=13_631_488_000, training_seed=None, step=13000, loop_reps=0,
       variant="", model_kind="birwkv_diffusion",
       notes="second f2 rung, for the exposure ladder only"),
]

#: The m4loop MMLU ladder rungs.  Same arm and seed as ``a29_c6loop_s20500``;
#: they exist so the E1 exposure curve has more than two points, and they are
#: never treated as replicates of each other.
TRACKS_2P9B_LADDER: Final = [
    _t(track_id=f"a29_c6loop_s{step}", track="adaptation_2p9b", paper_arm="C6",
       local_label=f"m4loop step {step}", rel_path=rel,
       model_dir="models/RWKV7-Goose-World3-2.9B-HF", sha256=sha,
       params_stored=4_091_581_443, params_executed=4_091_581_443,
       tokens_seen=tokens, training_seed=None, step=step, loop_reps=1,
       variant="tied depth loop over layers [16,32), one extra rep (see s20500)",
       model_kind="birwkv_diffusion", notes="exposure-ladder rung")
    for step, rel, sha, tokens in [
        (4750, "m2_baseline_triangle/m4loop_endpoint_ckpt",
         "ffaa464dabb3291c40749bbac4d6805e8a47082e3e2b082e0240365ffc525c07",
         4_980_736_000),
        (12500, "m2_baseline_triangle/m4loop_s12500_neutral_ckpt",
         "de2e8264e5952476e12678bd3f9c99f48a78123dab47fbe843e00fba050e56f5",
         13_107_200_000),
        (15000, "m2_baseline_triangle/m4loop_s15000_neutral_ckpt",
         "3cbae60fc2ff9a9350ede50ddca363fab9fbffeddb773cdd7a9e23aebe731ca9",
         15_728_640_000),
        (18500, "m4_ladder_ckpts/step_00018500",
         "",  # not hashed: superseded by s19500/s20500 and not evaluated further
         19_398_656_000),
        (19500, "outputs_birwkv_diffusion/m4-loop-2p9b/step_00019500",
         "51d9ed3bd549c695f8ac5b607a03905b22d84acea2314dfe6efa8ab8699c8822",
         20_447_232_000),
        (20000, "outputs_birwkv_diffusion/m4-loop-2p9b/step_00020000",
         "54ee5fd32c5c43706d07cf33d87281edb3fc15472ccdebb1b075b1055ee74dde",
         20_971_520_000),
    ]
]

# --------------------------------------------------------------------------
# 0.4B adaptation matrix: FOUR arms x THREE seeds, all finished, equal tokens.
# This is the only part of the evidence base with a seed dimension, so it is the
# only part that can enter the validator's run matrix as a phase.
#
# Stored parameters are identical across arms (591,054,848 / 851 with the loop's
# three scalars) because the checkpoint format keeps both directions.  Executed
# parameters differ: a1/a5 pass --force-forward and never call attn_bwd, whose
# tensors total 115,096,576 (measured from the state dicts).
# --------------------------------------------------------------------------

_BWD_SIDE_0P4B: Final = 115_096_576
_TASK7_SHA: Final = {
    ("a1", 17): "996352869fe24754264cea17186b6fe55242d7d14b9bcabdb6563454510558ae",
    ("a1", 29): "97af4617c31965b0b80969331418e4ea2bf233a6990f28069bd23bef87d87e9f",
    ("a1", 43): "a55f0121551978ba391cd32a8c9f6b899d61cbba801a0f9309e748399afd418a",
    ("a2", 17): "48ba5b5fbdee90df493b00464273c4cd4fd053e2b3343305b44df2d534ccd874",
    ("a2", 29): "bc57b65c15132e1fc08a57491f931554a3287bc81da31521ecceaac1c00712df",
    ("a2", 43): "7f82631b255bafd251d9828f7c7a5a6475e655ee1cb254cc0776783a7d92097e",
    ("a3", 17): "225e41c84877d8861c85b42af2cc6a25a18cce741b09abe160f313562d223a7f",
    ("a3", 29): "595a67f2da1b3bd11b6e6e740780d6d1135c530a3454f3535ea3ac0f80a943ab",
    ("a3", 43): "240129586a93a30bfaceb409aba450397513fef130483a0680bbca1165286fc5",
    ("a5", 17): "3ec08ab7b537c2387312b46e9e4be9bbaf3f1dd11585701fcd1a967c7d6997f9",
    ("a5", 29): "867a714e820cfa07d0b713ad65582a783fc420b245724ee864b1f89556c68440",
    ("a5", 43): "8252920f74bf48287e9713a358c0825374c3c7f01b6b7dc647b67162b0b2ea8f",
}

#: (paper arm, forward-only, loop reps, the deviation note).  Read from
#: ``scale/experiments/nonlatent_iclr/task7/arm_matrix.py:104-127``.
_TASK7_ARMS: Final = {
    "a1": ("C5", True, 0, ""),
    "a2": ("C6", False, 0, ""),
    "a3": ("C6", False, 1, "tied depth loop over layers [12,24), one extra rep; "
                           "no protocol arm declares recycled depth"),
    "a5": ("C5", True, 1, "tied depth loop over layers [12,24), one extra rep; "
                          "no protocol arm declares recycled depth"),
}

TRACKS_0P4B: Final = [
    _t(track_id=f"a04_{arm}_s{seed}", track="adaptation_0p4b",
       paper_arm=_TASK7_ARMS[arm][0], local_label=f"task7-{arm}-s{seed}",
       rel_path=f"outputs_task7_matrix/task7-{arm}-s{seed}/step_00001908",
       model_dir="models/rwkv7-0.4B", sha256=_TASK7_SHA[(arm, seed)],
       params_stored=591_054_848 + (3 if _TASK7_ARMS[arm][2] else 0),
       params_executed=(591_054_848 + (3 if _TASK7_ARMS[arm][2] else 0)
                        - (_BWD_SIDE_0P4B if _TASK7_ARMS[arm][1] else 0)),
       tokens_seen=2_000_683_008, training_seed=seed, step=1908,
       loop_reps=_TASK7_ARMS[arm][2], variant=_TASK7_ARMS[arm][3],
       model_kind="birwkv_diffusion",
       notes="8xH100 DCLM+ptcorpora adaptation, 1908 steps x 1,048,576 tok")
    for arm in ("a1", "a2", "a3", "a5") for seed in (17, 29, 43)
]

#: The second 0.4B lineage (``outputs_rwkv04b``).  Only four of the twelve runs
#: reached step 1908; the rest are partial or absent, which is why this lineage is
#: descriptive and the task7 matrix is the one that enters the validator.
#: ``M6`` here is the run launched as ``A0``: its own ``parameter_identity.json``
#: records ``arm_relabelled: true`` because it trained the masked objective, so it
#: maps to C5 and emphatically not to C1.
_RWKV04B_ARMS: Final = {
    "A1": ("C6", False, 0, ""),
    "A3": ("C6", False, 1, "tied depth loop plus a trained gate modulation; no "
                           "protocol arm declares either"),
    "M6": ("C5", True, 0, "launched under the label A0; parameter_identity.json "
                          "records arm_relabelled=true because the run trained "
                          "the masked-denoising objective, so it is a forward-only "
                          "denoiser (C5) and is NOT an autoregressive reference"),
}
_RWKV04B_FINISHED: Final = {
    ("A1", 43): ("1008a8c5a7be168c3c47718ac71606409757a576d5bbaf4ce59c4b2e1ab3e976",
                 455_010_304),
    ("A3", 17): ("acd2f6b4e29967797ef05ef643c3fd59989e6323ab48b9876d7367906842eab2",
                 None),
    ("M6", 17): ("c5507f88618feb3e6428700e3881abc11512d3609af74f2a699b3910bf4319b9",
                 453_938_176),
    ("M6", 29): ("bdfbfdb90ed9bce33e3fefe28c0e753b8191de610e88b88a79fd43e1400e6a3a",
                 453_938_176),
    ("M6", 43): ("434f87fb2c862e63180af9404e19dff3adb46a9c9ef3e622a3db61f1032370d7",
                 453_938_176),
}

TRACKS_RWKV04B: Final = [
    _t(track_id=f"r04_{arm}_s{seed}", track="adaptation_0p4b_second_lineage",
       paper_arm=_RWKV04B_ARMS[arm][0], local_label=f"{arm}_adapt_s{seed}",
       rel_path=f"outputs_rwkv04b/runs/{arm}_adapt_s{seed}/step_00001908.pt",
       model_dir="models/rwkv7-0.4B-world", sha256=sha,
       # The checkpoint bundles the optimizer, so ``params_stored`` cannot be read
       # off the file and is left unmeasured (-1) rather than guessed.  The
       # ``deployed_parameter_count`` the run's own parameter_identity.json records
       # counts the mixer copies actually built, i.e. the EXECUTED count, which is
       # why the forward-only M6 rows (453,938,176) sit below the bidirectional A1
       # row (455,010,304).
       params_stored=-1, params_executed=deployed if deployed else -1,
       tokens_seen=1_620_000_000, training_seed=seed, step=1908,
       loop_reps=_RWKV04B_ARMS[arm][2], variant=_RWKV04B_ARMS[arm][3],
       model_kind=FOREIGN_ARCH,
       notes="second 0.4B lineage; TiedBiRWKV7Block (one mixer called forward "
             "then per-doc-reversed) -- key layout layers.N.attn.* does NOT map "
             "onto the scale harness's untied attn_fwd/attn_bwd, measured "
             "2026-09-20. Evaluable through rwkv04b/longrwkv/eval only (E3 "
             "long-context lane); no E1 path exists for it. The checkpoint is "
             "also a torch.save payload (model, optimizer, scaler, step, record) "
             "rather than a flat state dict, which convert_rwkv04b already fixed")
    for (arm, seed), (sha, deployed) in sorted(_RWKV04B_FINISHED.items())
]

# --------------------------------------------------------------------------
# Released checkpoints.  No sha256: each is a multi-file HF directory, and the
# directory contents are the pin.  Their pretraining exposure is unknown to this
# study, which is exactly what makes them "released track" and not a control.
# --------------------------------------------------------------------------

TRACKS_RELEASED: Final = [
    _t(track_id="rel_c1_rwkv7_2p9b", track="released", paper_arm="C1",
       local_label="RWKV7-Goose-World3-2.9B-HF",
       rel_path="models/RWKV7-Goose-World3-2.9B-HF",
       model_dir="models/RWKV7-Goose-World3-2.9B-HF", sha256="",
       params_stored=2_949_000_000, params_executed=2_949_000_000,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="", model_kind="hf_causal",
       notes="the C1 reference for the 2.9B adaptation track. Also evaluable "
             "through the birwkv_diffusion path with --ckpt_dir base + fwdce, "
             "which is the comparability control: same harness, same scorer"),
    _t(track_id="rel_c1_rwkv7_0p4b", track="released", paper_arm="C1",
       local_label="rwkv7-0.4B-world", rel_path="models/rwkv7-0.4B-world",
       model_dir="models/rwkv7-0.4B-world", sha256="",
       params_stored=450_000_000, params_executed=450_000_000,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="", model_kind="hf_causal",
       notes="the C1 reference for the 0.4B adaptation track"),
    _t(track_id="rel_c1_rwkv7_1p5b", track="released", paper_arm="C1",
       local_label="RWKV7-Goose-World3-1.5B-HF",
       rel_path="models/RWKV7-Goose-World3-1.5B-HF",
       model_dir="models/RWKV7-Goose-World3-1.5B-HF", sha256="",
       params_stored=1_500_000_000, params_executed=1_500_000_000,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="", model_kind="hf_causal", notes="scale interpolation point"),
    _t(track_id="rel_c2_llada_8b", track="released", paper_arm="C2",
       local_label="LLaDA-8B-Base", rel_path="models/LLaDA-8B-Base",
       model_dir="models/LLaDA-8B-Base", sha256="",
       params_stored=8_000_000_000, params_executed=8_000_000_000,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="", model_kind=LLADA_MODEL_KIND,
       notes="the released softmax masked-diffusion reference (C2). 2x the "
             "parameters of the largest adaptation entry and an unknown, larger "
             "pretraining budget: it bounds what a diffusion LM can do, it does "
             "not isolate the mixer. Runs through llada_batch_eval.py, NOT "
             "run_eval.py: block_length/steps are its own flags"),
    _t(track_id="rel_c2_sedd_medium", track="released", paper_arm="C2",
       local_label="sedd-medium", rel_path="models/sedd-medium",
       model_dir="models/sedd-medium", sha256="",
       params_stored=320_000_000, params_executed=320_000_000,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="score-entropy parameterization, not the absorbing ELBO this "
               "paper's C2 declares",
       model_kind=NO_LOADER,
       notes="config.json is a hydra training config with no model_type and no "
             "auto_map, and 'sedd' appears nowhere in the eval tree: there is no "
             "loader, so this checkpoint is UNEVALUABLE here and is reported as "
             "absent, not pending"),
    _t(track_id="rel_c7_mamba2_2p7b", track="released", paper_arm="C7",
       local_label="mamba2-2.7b", rel_path="models/mamba2-2.7b",
       model_dir="models/mamba2-2.7b", sha256="",
       params_stored=2_700_000_000, params_executed=2_700_000_000,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="autoregressive, not the bidirectional diffusion C7 declares; "
               "stands as the released scalar-SSD recurrent reference only",
       model_kind=NO_LOADER,
       notes="config.json is the mamba_ssm native format (d_model/ssm_cfg, no "
             "model_type), mamba_ssm is absent from the relay2:v2 image, and "
             "efficiency/adapters.py:600 already names this as needing its own "
             "loader. UNEVALUABLE without new code; reported as absent"),
    _t(track_id="rel_c0_qwen25_3b", track="released", paper_arm="C0",
       local_label="Qwen2.5-3B", rel_path="models/Qwen2.5-3B",
       model_dir="models/Qwen2.5-3B", sha256="",
       params_stored=3_090_000_000, params_executed=3_090_000_000,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="", model_kind="hf_causal",
       notes="released softmax AR reference at comparable nominal size"),
    _t(track_id="rel_c0_llama32_3b", track="released", paper_arm="C0",
       local_label="Llama-3.2-3B", rel_path="models/Llama-3.2-3B",
       model_dir="models/Llama-3.2-3B", sha256="",
       params_stored=3_210_000_000, params_executed=3_210_000_000,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="", model_kind="hf_causal",
       notes="the KV-cache comparator in the E2 serving panel"),

    # ----------------------------------------------------------------------
    # State / linear-attention long-context baselines.  The paper's long-context
    # claim is about a *constant-state* mixer, and RWKV-7 is one member of that
    # family: without another family member the claim rests on a comparison to
    # softmax attention only, which cannot separate "recurrent state" from "our
    # checkpoint".  These three are the members present on this disk.  Each was
    # load-probed on the login node 2026-09-21 rather than assumed, and the
    # results differ per model -- see MISSING_RUNTIME_DEP.
    # ----------------------------------------------------------------------
    _t(track_id="rel_c7ar_mamba_2p8b", track="released", paper_arm="C7_ar",
       local_label="mamba-2.8b", rel_path="models/mamba-2.8b",
       model_dir="models/mamba-2.8b", sha256="",
       params_stored=2_768_345_600, params_executed=2_768_345_600,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="Mamba-1 (S6 selective scan), not Mamba-2/SSD: config.json is "
               "model_type=mamba with MambaForCausalLM. The protocol's C7 'mamba2' "
               "mixer label is the family name; this checkpoint is the earlier "
               "member and the row must not be read as an SSD result",
       model_kind="hf_causal",
       notes="LOADS: MambaForCausalLM built and forwarded on CPU 2026-09-21, "
             "2,768,345,600 params, 642 tensors, vocab 50280; MMLU labels ' A'/' "
             "B'/' C'/' D' are single tokens 329/378/330/399 and the one-token tree "
             "is built at capability_eval_data/tasks_mmlu_onetoken/mamba. "
             "PRICING CAVEAT: mamba_ssm and causal_conv1d are both absent from "
             "relay2:v2, so transformers takes MambaMixer.slow_forward, whose SSM "
             "recurrence is a Python `for i in range(seq_len)` loop (one matmul per "
             "timestep); mambapy is absent too, so the pscan path is unavailable "
             "and it is gated on self.training anyway. E1 at 4K is affordable; a "
             "32K/64K E3 cell is NOT priced by the RWKV arms' cost and must be "
             "measured at Q1 before any long-context fan-out"),
    _t(track_id="rel_c3ar_gla_1p3b", track="released", paper_arm="C3_ar",
       local_label="gla-1.3B-100B",
       rel_path="hf_cache/hub/models--fla-hub--gla-1.3B-100B",
       model_dir="hf_cache/hub/models--fla-hub--gla-1.3B-100B", sha256="",
       params_stored=-1, params_executed=-1,
       tokens_seen=100_000_000_000, training_seed=None, step=None, loop_reps=0,
       variant="gated linear attention, the paper's 'additive_kernel' mixer, "
               "autoregressive",
       model_kind="fla_causal",
       notes="Evaluable through the fla_causal loader, which imports fla.models "
             "before AutoConfig so the 'gla' architecture is registered "
             "(hf_causal_model.load_fla_causal; run_eval --model_kind fla_causal). "
             "The blocker this replaced was one missing import, not a missing or "
             "broken checkpoint: model_type=gla with fla-native keys "
             "(model.layers.0.attn.g_proj.weight, 339 tensors, 1,365,514,240 stored "
             "elements). Its Llama tokenizer gives single-token labels 330/365/334/384, "
             "so an E1 tree exists at tasks_mmlu_onetoken/gla. UNVERIFIED ON HARDWARE: "
             "`import fla.models` needs a GPU (it raises '0 active drivers' on a login "
             "node), so the loader cannot be exercised off-pod. The Q1 probe receipt "
             "is what settles it, and the emitter refuses an E1 fan-out without one. "
             "tokens_seen is from the checkpoint name (100B)"),
    _t(track_id="rel_c8ar_rwkv6_3b", track="released", paper_arm="C8_ar",
       local_label="rwkv6-world-3b", rel_path="models/rwkv6-world-3b",
       model_dir="models/rwkv6-world-3b", sha256="",
       params_stored=3_099_863_040, params_executed=3_099_863_040,
       tokens_seen=-1, training_seed=None, step=None, loop_reps=0,
       variant="RWKV-6 mixer (data-dependent decay, no delta-rule state update); "
               "the immediate predecessor of this paper's RWKV-7 mixer",
       model_kind=MISSING_RUNTIME_DEP,
       notes="UNEVALUABLE in relay2:v2, for two independent reasons measured "
             "2026-09-21. (1) Its bundled modeling_rwkv6.py imports bitsandbytes at "
             "module scope for a 4-bit rescale helper, and "
             "transformers.dynamic_module_utils.check_imports AST-scans the remote "
             "file before reading any weight -> ImportError, unreachable by any "
             "try/except in that file. (2) Line 41 imports "
             "fla.ops.rwkv6.recurrent_fuse, which does not exist in fla 0.5.0 (the "
             "symbol is in fla.ops.rwkv6.fused_recurrent), inside a try/except that "
             "only prints -- so fused_recurrent_rwkv6 would be unbound at the first "
             "forward, after a full 3B load. params from a meta-device load of "
             "pytorch_model.bin (902 tensors). The tree at "
             "tasks_mmlu_onetoken/rwkv6 (labels 66/67/68/69) is built and valid, so "
             "the row becomes schedulable the moment a working loader exists"),
]

ALL_TRACKS: Final = (TRACKS_2P9B + TRACKS_2P9B_LADDER + TRACKS_0P4B
                     + TRACKS_RWKV04B + TRACKS_RELEASED)

#: The phase name the 0.4B matrix enters ``protocol.json["phases"]`` under, with
#: the arm ids the validator will check.  ``validate_protocol`` demands
#: ``{(arm, seed)}`` be exactly the cross product of the phase's arms and the
#: protocol's three seeds, which the task7 matrix satisfies and no other lineage
#: here does.
ADAPTATION_PHASE: Final = {
    "id": "adaptation_0p4b",
    "required_for_core_claims": False,
    "arms": ["C5", "C6", "C5_loop", "C6_loop"],
    "parameter_target": 591_054_848,
    "tokens_per_run": 2_000_683_008,
    "runs": 12,
    "aggregate_token_exposures": 12 * 2_000_683_008,
    "source": "adaptation_from_released_rwkv7_0p4b",
}

#: task7 arm -> the phase arm id above.  The loop arms get their own ids rather
#: than being folded into C5/C6, because folding them would let the validator
#: certify a matrix in which two rows labelled C6 execute different depths.
PHASE_ARM_OF: Final = {"a1": "C5", "a2": "C6", "a3": "C6_loop", "a5": "C5_loop"}


def check_directional_accounting(tracks=ALL_TRACKS) -> None:
    """A forward-only arm must not claim it executes its stored backward half.

    This is the check that catches the specific confusion the module docstring
    warns about: copying ``params_stored`` into ``params_executed`` for an arm
    that passes ``--force-forward``.  That mistake is invisible in any single
    number and would make a forward-only arm look parameter-matched to a
    bidirectional one in the paper's tracks table.
    """
    for t in tracks:
        if t.paper_arm not in PAPER_ARMS:
            raise ValueError(
                f"{t.track_id}: paper_arm {t.paper_arm!r} is not one of "
                f"{sorted(PAPER_ARMS)}; the validator would reject the row")
        forward_only = PAPER_ARMS[t.paper_arm][2] == "forward"
        measured = UNMEASURED not in (t.params_stored, t.params_executed)
        if forward_only and measured and t.params_executed == t.params_stored:
            raise ValueError(
                f"{t.track_id}: paper arm {t.paper_arm} is direction=forward, so "
                f"the stored backward mixers are not executed, yet "
                f"params_executed == params_stored == {t.params_stored}. Either "
                f"the arm mapping or the count is wrong.")
        if t.loop_reps and not t.variant:
            raise ValueError(
                f"{t.track_id}: loop_reps={t.loop_reps} but no variant note. No "
                f"protocol arm declares recycled depth, so the deviation must be "
                f"stated wherever the row is reported.")
        if measured and t.params_executed > t.params_stored:
            raise ValueError(f"{t.track_id}: executes more than it stores")
        if t.track != "released" and t.tokens_seen <= 0:
            raise ValueError(
                f"{t.track_id}: an adaptation entry must state its own token "
                f"exposure; only a released checkpoint may leave it unknown")


def check_runtime_dep_rows_state_the_dependency(tracks=ALL_TRACKS) -> None:
    """A :data:`MISSING_RUNTIME_DEP` row must name what is missing.

    The whole value of this kind over :data:`NO_LOADER` is that it distinguishes
    "this baseline cannot be run here" from "this baseline is absent".  That
    distinction only survives into the paper if the row says *which* dependency,
    so a reader can tell a fixable gap (import ``fla.models`` first) from an
    unfixable one (a remote file that needs ``bitsandbytes`` to be importable at
    all).  A row with an empty note would collapse the two.
    """
    for t in tracks:
        if t.model_kind != MISSING_RUNTIME_DEP:
            continue
        if not any(dep in t.notes for dep in ("bitsandbytes", "fla", "mamba_ssm",
                                              "causal_conv1d", "triton")):
            raise ValueError(
                f"{t.track_id}: model_kind is {MISSING_RUNTIME_DEP} but its notes "
                f"name no missing dependency. Name the import that fails, or the "
                f"row is indistinguishable from an absent baseline.")


def check_ids_unique(tracks=ALL_TRACKS) -> None:
    ids = [t.track_id for t in tracks]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate track_id: {dupes}")


def check_model_kinds(tracks=ALL_TRACKS) -> None:
    """Every entry names a loader that exists, and none claims a wrong one."""
    for t in tracks:
        if t.model_kind not in MODEL_KINDS:
            raise ValueError(
                f"{t.track_id}: model_kind {t.model_kind!r} is not in "
                f"{sorted(MODEL_KINDS)}; run_eval.py would reject it at argparse")
        if t.model_kind == "birwkv_diffusion" and not t.model_dir:
            raise ValueError(
                f"{t.track_id}: birwkv_diffusion needs --model_dir for its "
                f"geometry and tokenizer (run_eval.py:207 raises without it)")


def check_hf_causal_is_loadable(tracks=ALL_TRACKS, models_root: Path | None = None) -> list[str]:
    """Refuse an ``hf_causal`` row whose config cannot reach a transformers class.

    ``AutoModelForCausalLM.from_pretrained`` dispatches on ``config.model_type``
    (or an ``auto_map`` for remote code).  A checkpoint with neither cannot be
    built, and the failure only appears in-pod, minutes into a job that has
    already been billed.  Two released checkpoints on this disk are exactly that
    case, which is why they carry :data:`NO_LOADER` -- and why this check exists
    to stop a future edit from quietly promoting them back to ``hf_causal``.

    **What this check cannot see** (measured 2026-09-21): it reads ``config.json``
    only, so it passes any checkpoint that *declares* a dispatch target even when
    building it raises.  ``rwkv6-world-3b`` has an ``auto_map`` and would sail
    through here while failing on a missing ``bitsandbytes``; ``gla-1.3B-100B``
    has ``model_type: gla`` and fails because nothing imports ``fla.models``.
    Those two are :data:`MISSING_RUNTIME_DEP`, and the guard for them is
    :func:`check_runtime_dep_rows_are_not_scheduled` plus the emitter refusal --
    a config-only check is structurally blind to an import-time failure.

    Returns the ids skipped because their config was unreadable, so a caller can
    tell "checked and fine" from "could not check".
    """
    root = models_root if models_root is not None else G
    skipped: list[str] = []
    for t in tracks:
        if t.model_kind != "hf_causal":
            continue
        cfg = root / t.model_dir / "config.json"
        if not cfg.is_file():
            skipped.append(t.track_id)
            continue
        try:
            conf = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped.append(t.track_id)
            continue
        if not conf.get("model_type") and not conf.get("auto_map"):
            raise ValueError(
                f"{t.track_id}: declared hf_causal but {cfg} has neither "
                f"model_type nor auto_map, so AutoModelForCausalLM cannot build "
                f"it. Use NO_LOADER (and report the baseline as absent) or add a "
                f"dedicated loader.")
    return skipped


def check_birwkv_key_layout(tracks=ALL_TRACKS, models_root: Path | None = None
                            ) -> list[dict]:
    """Refuse a ``birwkv_diffusion`` row whose state dict names the wrong mixers.

    This is the check that was missing on 2026-09-20.  ``check_loadable_layout``
    in the emitter verifies the *container* (a dir with a flat ``model.pt``), and
    ``check_model_kinds`` verifies the *name* is one run_eval accepts.  Neither
    looks inside the file, so five r04 probes passed both and then died in-pod on
    ``checkpoint/model mismatch: ['layers.0.attn_fwd.x_r', ...]`` -- a full
    0.4B load per job to learn one key name.

    ``load_birwkv_diffusion`` builds ``BiRWKV7Block`` with two untied mixers
    (``attn_fwd``/``attn_bwd``) and raises on any missing key that does not
    contain ``fuse_``, so a checkpoint carrying a single tied ``layers.N.attn.*``
    is unloadable no matter what the registry says.  Reading one key name off the
    file settles it for free.

    Returns rows recording what was actually checked, so a caller can distinguish
    "verified compatible" from "file absent, unchecked".  Raises only when a file
    is present and provably incompatible.
    """
    import torch

    root = models_root if models_root is not None else G
    rows: list[dict] = []
    for t in tracks:
        if t.model_kind != "birwkv_diffusion":
            continue
        target = t.path if t.path.suffix == ".pt" else t.path / "model.pt"
        if not target.is_file():
            rows.append({"track_id": t.track_id, "check": "absent",
                         "path": str(target)})
            continue
        state = torch.load(target, map_location="meta", weights_only=True,
                           mmap=True)
        keys = state if isinstance(state, dict) else {}
        untied = any("attn_fwd." in k for k in keys)
        tied = any(k.startswith("layers.") and ".attn." in k for k in keys)
        rows.append({"track_id": t.track_id, "check": "key_layout",
                     "n_tensors": len(keys), "has_attn_fwd": untied,
                     "has_tied_attn": tied})
        if not untied:
            raise ValueError(
                f"{t.track_id}: declared birwkv_diffusion but {target} has no "
                f"'attn_fwd.*' keys"
                + (" and does have tied 'layers.N.attn.*' keys, i.e. it is a "
                   "TiedBiRWKV7Block checkpoint" if tied else "")
                + f". load_birwkv_diffusion builds untied attn_fwd/attn_bwd and "
                  f"raises 'checkpoint/model mismatch' on the missing keys, in "
                  f"the pod, after paying the load. Use FOREIGN_ARCH and route it "
                  f"to its own harness.")
    return rows


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


#: Kinds whose weights live in a multi-file HF **directory**, so presence is
#: ``path.is_dir()`` and there is no single file to hash.  Written as a set of the
#: constants rather than inline string literals: this test previously compared
#: against the literal ``"llada"`` while :data:`LLADA_MODEL_KIND` is
#: ``"llada_batch"``, so the LLaDA row silently took the *file* branch and
#: reported ``present: False`` for a directory that is on disk.
DIR_SHAPED_KINDS: Final = frozenset({"hf_causal", LLADA_MODEL_KIND, NO_LOADER})


def verify_on_disk(tracks=ALL_TRACKS, rehash: bool = False) -> list[dict]:
    """Report, per track, whether the weights are present and the sha matches.

    ``rehash=False`` (the default) only checks presence: re-reading 16 GiB x 13
    files costs ~40 minutes, and the shas here were computed by this module's own
    ``--rehash`` pass.  The returned rows always say which check actually ran, so
    a caller cannot mistake "present" for "verified".
    """
    rows = []
    for t in tracks:
        row = {"track_id": t.track_id, "rel_path": t.rel_path,
               "declared_sha256": t.sha256, "check": "presence"}
        target = t.path if t.path.suffix == ".pt" else t.path / "model.pt"
        if t.model_kind in DIR_SHAPED_KINDS:
            row["present"] = t.path.is_dir()
            row["bytes"] = None
        else:
            row["present"] = target.is_file()
            row["bytes"] = target.stat().st_size if target.is_file() else None
        if rehash and row["present"] and t.sha256 and target.is_file():
            row["check"] = "sha256"
            row["measured_sha256"] = sha256_file(target)
            row["sha_match"] = row["measured_sha256"] == t.sha256
        rows.append(row)
    return rows


CSV_FIELDS: Final = [
    "track_id", "track", "paper_arm", "local_label", "rel_path", "model_dir",
    "sha256", "params_stored", "params_executed", "tokens_seen", "training_seed",
    "step", "loop_reps", "model_kind", "variant", "notes",
]


def write_csv(path: Path, tracks=ALL_TRACKS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for t in tracks:
            w.writerow({k: getattr(t, k) for k in CSV_FIELDS})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, help="write evidence/tracks.csv here")
    ap.add_argument("--verify", action="store_true", help="check presence on disk")
    ap.add_argument("--rehash", action="store_true",
                    help="with --verify, recompute every sha256 (slow: ~40 min)")
    ap.add_argument("--json", type=Path, help="write the verification report here")
    ap.add_argument("--check-key-layout", action="store_true",
                    help="open every birwkv_diffusion model.pt and confirm it "
                         "names attn_fwd (mmap'd to meta; no weights are read)")
    args = ap.parse_args(argv)

    check_ids_unique()
    check_directional_accounting()
    check_model_kinds()
    check_runtime_dep_rows_state_the_dependency()
    unchecked = check_hf_causal_is_loadable()
    report: dict = {
        "tracks": len(ALL_TRACKS),
        "by_track": {k: sum(1 for t in ALL_TRACKS if t.track == k)
                     for k in sorted({t.track for t in ALL_TRACKS})},
        "by_model_kind": {k: sum(1 for t in ALL_TRACKS if t.model_kind == k)
                          for k in sorted({t.model_kind for t in ALL_TRACKS})},
        "unevaluable_no_loader": [t.track_id for t in ALL_TRACKS
                                  if t.model_kind == NO_LOADER],
        "foreign_arch_own_harness_only": [t.track_id for t in ALL_TRACKS
                                          if t.model_kind == FOREIGN_ARCH],
        "missing_runtime_dep": [t.track_id for t in ALL_TRACKS
                                if t.model_kind == MISSING_RUNTIME_DEP],
        "hf_causal_loadability_unchecked": unchecked,
        "baseline_arms_to_patch": sorted(BASELINE_ARMS),
        "exposure_disclosure": EXPOSURE_DISCLOSURE,
        "adaptation_phase": ADAPTATION_PHASE,
        "controlled_matrix_status": "not_run (all 48 rows); this registry makes "
                                    "no claim about it",
    }
    if args.csv:
        write_csv(args.csv)
        report["csv"] = str(args.csv)
    if args.check_key_layout:
        report["key_layout"] = check_birwkv_key_layout()
    if args.verify:
        rows = verify_on_disk(rehash=args.rehash)
        report["verification"] = rows
        report["missing"] = [r["track_id"] for r in rows if not r["present"]]
        report["sha_mismatch"] = [r["track_id"] for r in rows
                                  if r.get("sha_match") is False]
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report.get("missing") or report.get("sha_mismatch"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
