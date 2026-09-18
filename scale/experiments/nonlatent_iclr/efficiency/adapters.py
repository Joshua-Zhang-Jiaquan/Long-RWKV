"""One-interface-per-model adapters for the measured efficiency matrix.

Contribution 1 of the paper is an *efficiency* claim, and until this module
existed no Transformer or LLaDA throughput or memory had ever been measured on
this cluster.  The adapters are the seam that makes the comparison honest: a
single ``Adapter`` interface over six heterogeneous checkpoints, each carrying
the one number the comparison turns on -- ``nfe``, the number of model forwards
needed for a single answer.

Three properties are load-bearing, and each is asserted by a test rather than
trusted:

* **``nfe`` is a required field, never a silent default.**  A 1-forward
  autoregressive decode and a 16-forward masked-diffusion answer are not the same
  unit of work, and a table that quietly scored the second as the first would be
  exactly the unmatched comparison the paper exists to avoid.
  ``ModelSpec.__post_init__`` refuses an autoregressive model whose ``nfe`` is
  not 1 and a masked-diffusion model whose ``nfe`` is 1.
* **Importing this module touches neither the disk nor CUDA.**  The matrix must
  be enumerable from a login node, so ``REGISTRY`` is built from measured
  *constants* rather than a scan of the filesystem, and every heavyweight import
  (``torch`` via ``theory_bounds``, ``transformers``) happens inside the method
  that needs it.  ``build`` is the only entry point that stats the disk, and it
  is never called at import time.
* **Paths are recorded, never invented.**  Each hub checkpoint is named by the
  subdirectory that actually exists under ``models_root``; the BiRWKV diffusion
  release ships inside this repository under ``release/``, so its path is
  derived from this module's location rather than from ``models_root``.  A key
  whose weights are absent refuses with the path, so a missing checkpoint cannot
  be scored as a present one.  The BiRWKV checkpoint is the one model that needs
  a SECOND path -- an HF geometry directory -- because it ships as a flat
  ``model.pt`` with no ``config.json``; that directory is resolved by its own
  env knob and its absence is a separate, named refusal, so "wrong geometry" and
  "weights not downloaded" cannot be confused.  The step that produced the
  endpoint is read from its ``meta.json`` and carried on the loaded record, so a
  table row is traceable to the checkpoint that produced it.

The ``params``/``weights_bytes`` numbers below are measurements of the bytes
that are actually on disk, taken 2026-09-17 from the files named by each
``path``; they are not nominal sizes read off a model card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Final, Iterable, Mapping, Protocol

#: The directory the hub checkpoints were measured under.  ``build`` takes its
#: own ``models_root``; this constant records where the baked numbers came from
#: so a reader can re-measure them without guessing.
RECORDED_MODELS_ROOT: Final = "/inspire/hdd/global_user/zhangjiaquan-253108540222/models"

#: Declared architecture families.  A transformer keeps a KV cache that is linear
#: in context; a linear-attention or state-space model keeps a constant per-layer
#: recurrent state.
TRANSFORMER: Final = "transformer"
LINEAR_ATTENTION: Final = "linear_attention"
STATE_SPACE: Final = "state_space"
FAMILIES: Final = frozenset({TRANSFORMER, LINEAR_ATTENTION, STATE_SPACE})

#: Declared training objectives.  The objective decides the answer's forward cost,
#: which is why ``nfe`` is not allowed to float free of it.
AUTOREGRESSIVE: Final = "autoregressive"
MASKED_DIFFUSION: Final = "masked_diffusion"
OBJECTIVES: Final = frozenset({AUTOREGRESSIVE, MASKED_DIFFUSION})

#: One forward produces one autoregressive token, so one answer costs one forward.
NFE_AUTOREGRESSIVE: Final = 1

#: LLaDA's full-sequence masked-diffusion decode.  The released weights ship no
#: sampler default, so the project's registered token-denoising schedule (16
#: steps) is recorded here; a run that uses a different schedule must override the
#: spec rather than inherit this number, because the two are not comparable units.
NFE_LLADA: Final = 16

#: The BiRWKV trajectory release's own sampling recipe: the release README's safe
#: default generates a 4096-token answer with ``--steps 100``.
NFE_BIRWKV: Final = 100

#: Storage width of the deployments below.  The KV cache and the recurrent state
#: are held in the same width as the weights (bf16 for the safetensors
#: checkpoints, fp16 for the mamba2 ``.bin``); 2 bytes either way.
BF16_BYTES: Final = 2
FP16_BYTES: Final = 2

#: The BiRWKV diffusion release lives inside this repository, not under the hub
#: models root.  Deriving its location from this file keeps the record correct if
#: the repository is mounted elsewhere; the path is never resolved (resolving
#: would stat the filesystem at import time, which the module forbids).
_REPO_ROOT: Final = Path(__file__).absolute().parents[4]
_BIRWKV_RELEASE: Final = str(_REPO_ROOT / "release" / "StateDiffRWKV-2.9B-trajectory4096-pretrained")

#: The registry key of the model under study, named so the override below can target it
#: without a string literal scattered through the module.
BIRWKV_KEY: Final = "birwkv-2.9b"

#: An override for the BiRWKV release directory.
#:
#: The default is the in-repo release path, which is the right DEFAULT -- the release is
#: meant to ship with the repository, and a path derived from the module's own location
#: cannot be wrong about where the repository is.  But the checkpoint this program
#: actually MEASURES is a training endpoint that lives on the cluster
#: (``m2_baseline_triangle/m4loop_*_endpoint_ckpt``), not in the tree, and the
#: manuscript labels every number by the STEP that produced it.  Without an override the
#: efficiency matrix refuses to run against the model it is about, or worse, runs
#: against whatever happens to sit at the default path and records it under a step label
#: it never had.
#:
#: Whatever this resolves to is RECORDED in the probe row, so a table entry can be read
#: back to the checkpoint that produced it.
BIRWKV_RELEASE_ENV: Final = "NONLATENT_BIRWKV_RELEASE"

#: The HF directory whose ``config.json`` and safetensors define the BiRWKV *geometry*.
#:
#: The measured checkpoint is a flat ``model.pt`` plus ``meta.json`` with NO ``config.json``, so
#: ``AutoModelForCausalLM.from_pretrained`` can never build it; the architecture has to come from
#: the HF RWKV-7 release the denoiser was warm-started from. The default is the recorded hub
#: directory for the same 2.9B backbone (verified present 2026-09-18), which is also the path the
#: ``rwkv7-goose-world3-2.9b`` registry entry resolves to under ``RECORDED_MODELS_ROOT``;
#: :data:`BIRWKV_GEOMETRY_ENV` redirects it for a run whose model tree lives elsewhere. The path is
#: never resolved here (resolving would stat the filesystem at import time, which the module
#: forbids).
BIRWKV_GEOMETRY_DIR: Final = str(Path(RECORDED_MODELS_ROOT) / "RWKV7-Goose-World3-2.9B-HF")

#: An override for the BiRWKV geometry directory (see :data:`BIRWKV_GEOMETRY_DIR`).
#:
#: Deliberately a SEPARATE knob from :data:`BIRWKV_RELEASE_ENV`: "which weights" and "which
#: architecture" are different faults, and one override for both would let a run point at the
#: right weights and the wrong geometry (or vice versa) with nothing to tell the two apart. The
#: absence of the directory named here is a NAMED refusal that prints the path, so "wrong
#: model_dir" and "model not downloaded" surface as the different faults they are.
BIRWKV_GEOMETRY_ENV: Final = "NONLATENT_BIRWKV_GEOMETRY"

#: An override for the BiRWKV training-time block size.
#:
#: Consulted only for a checkpoint that carries ``block_t_cond.*`` keys. The block size is a
#: training-time contract (the arm that produced the measured endpoint used 64), so it is never
#: defaulted: a wrong value silently misaligns every block's timestep and no downstream number
#: reveals it. Without this value such a checkpoint is refused, and the refusal names this knob.
BIRWKV_BLOCK_SIZE_ENV: Final = "NONLATENT_BIRWKV_BLOCK_SIZE"

#: The registry key of the mamba2 checkpoint, which needs its own loader for a reason that is
#: structural rather than cosmetic: its ``config.json`` is the mamba_ssm format (``d_model``,
#: ``n_layer``, ``ssm_cfg``) and carries no ``model_type``, so ``AutoModelForCausalLM`` cannot
#: resolve an architecture for it. Named so the dispatch can target it without a scattered literal.
MAMBA2_KEY: Final = "mamba2-2.7b"

#: The two files the mamba2 loader reads: the mamba_ssm-format config, and the state dict.
#:
#: A ``.bin`` rather than ``.safetensors`` is why the storage-width constant for this row is
#: :data:`FP16_BYTES` and not :data:`BF16_BYTES`; the model is built in that width so the probe's
#: peak reading is of the deployment the ``weights_bytes`` constant describes.
MAMBA2_CONFIG_FILE: Final = "config.json"
MAMBA2_WEIGHTS_FILE: Final = "pytorch_model.bin"

#: The parameter-name difference between the mamba_ssm checkpoint and the HF ``Mamba2ForCausalLM``
#: it is built into.
#:
#: ``transformers`` ships a tensor-for-tensor port of the same Mamba2 mixer, so the state dict maps
#: completely except for the token embedding: mamba_ssm calls it ``backbone.embedding`` where HF
#: calls it ``backbone.embeddings``. The in-proj packing (``z, x, B, C, dt``), the depthwise
#: ``conv1d`` layout, and every per-head tensor (``A_log``/``dt_bias``/``D``) are identical, which
#: is why a SINGLE rename suffices; a checkpoint whose names do not reduce to this map is caught by
#: the load's key-by-key check rather than silently half-applied.
MAMBA2_HF_KEY_RENAME: Final = {
    "backbone.embedding.weight": "backbone.embeddings.weight",
}


class AdapterRefusal(ValueError):
    """A model cannot be adapted to the measured matrix as requested.

    Raised rather than returning a partially-populated adapter.  Every caller in
    this module needs to be able to tell "this key is unknown" from "this key is
    known but its weights are absent" from "this model's loader is missing": the
    three produce very different claims about the matrix, and a single generic
    error would let a missing checkpoint be scored as a present one.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything the matrix must know about a model before it is loaded.

    ``nfe`` is deliberately the fourth field and has no default.  Comparing a
    1-forward autoregressive decode to a 16-NFE diffusion answer as if they were
    the same unit of work is exactly the unmatched comparison the paper exists to
    avoid, so the cost of an answer is a required, per-model declaration and
    ``__post_init__`` refuses a declaration that contradicts its objective.
    """

    key: str
    family: str
    objective: str
    nfe: int
    #: Total parameter count, measured from the tensor headers of the files below.
    params: int
    #: Total bytes of the weight files on disk, measured with the files present.
    weights_bytes: int
    #: The model directory: a subdirectory of ``models_root`` for a hub
    #: checkpoint, or an absolute path for a checkpoint released in this repo.
    path: str

    def __post_init__(self) -> None:
        if not self.key:
            raise AdapterRefusal("a model spec must carry a key")
        if self.family not in FAMILIES:
            raise AdapterRefusal(
                f"{self.key}: undeclared family {self.family!r}; declared {sorted(FAMILIES)}")
        if self.objective not in OBJECTIVES:
            raise AdapterRefusal(
                f"{self.key}: undeclared objective {self.objective!r}; declared {sorted(OBJECTIVES)}")
        if self.nfe < 1:
            raise AdapterRefusal(f"{self.key}: nfe must be at least 1, got {self.nfe}")
        if self.objective == AUTOREGRESSIVE and self.nfe != NFE_AUTOREGRESSIVE:
            raise AdapterRefusal(
                f"{self.key}: an autoregressive answer is one forward, but nfe={self.nfe}; "
                f"a multi-forward autoregressive cost means the unit of work is wrong")
        if self.objective == MASKED_DIFFUSION and self.nfe == NFE_AUTOREGRESSIVE:
            raise AdapterRefusal(
                f"{self.key}: a masked-diffusion answer needs more than one forward, but "
                f"nfe=1 would score it as an autoregressive decode")
        if self.params <= 0 or self.weights_bytes <= 0:
            raise AdapterRefusal(
                f"{self.key}: params and weights_bytes must be measured and positive, "
                f"got params={self.params} weights_bytes={self.weights_bytes}")
        if not self.path:
            raise AdapterRefusal(f"{self.key}: a model spec must carry a path")


@dataclass(frozen=True, slots=True)
class AttentionGeometry:
    """The shape a Transformer KV cache is computed from.

    ``kv_heads`` is the grouped-query width.  It defaults to full multi-head
    attention, which is the comparison most favourable to the Transformer and
    therefore the honest one to quote provided the reduction from ``n_heads`` is
    stated (GQA/MQA shrink the transformer side).
    """

    n_layers: int
    n_heads: int
    hidden_size: int
    dtype_bytes: int
    kv_heads: int | None = None

    def __post_init__(self) -> None:
        if min(self.n_layers, self.n_heads, self.hidden_size, self.dtype_bytes) <= 0:
            raise AdapterRefusal("attention geometry must be positive")
        if self.kv_heads is not None and not 0 < self.kv_heads <= self.n_heads:
            raise AdapterRefusal(
                f"kv_heads must lie in (0, n_heads], got {self.kv_heads}/{self.n_heads}")


@dataclass(frozen=True, slots=True)
class RecurrentGeometry:
    """The shape a constant recurrent state is computed from.

    The protocol's closed form is square -- ``head_dim`` on both axes -- because
    the RWKV WKV state is ``head_dim x head_dim`` per head.  ``state_dim`` keeps
    the value axis explicit so an SSM whose state is rectangular is not silently
    squared: it equals ``head_dim`` for the RWKV family and is the SSM state size
    (128) for mamba2 whose SSD state is ``head_dim x d_state``.
    """

    n_layers: int
    n_heads: int
    head_dim: int
    state_dim: int
    directions: int
    dtype_bytes: int

    def __post_init__(self) -> None:
        if min(self.n_layers, self.n_heads, self.head_dim, self.state_dim,
               self.directions, self.dtype_bytes) <= 0:
            raise AdapterRefusal("recurrent geometry must be positive")


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A materialized checkpoint plus the provenance needed to trust the row.

    ``step`` is the checkpoint step that produced the weights, read from ``meta.json`` for a model
    whose release carries one. It is ``None`` for a hub checkpoint, whose provenance is its
    directory name rather than a training step; for BiRWKV it is always an int, because the
    manuscript labels every efficiency number by the step that produced it and a row that cannot be
    traced to a step is not usable.
    """

    spec: ModelSpec
    path: str
    device: str
    model: object
    step: int | None = None


class Adapter(Protocol):
    """The interface the measured matrix consumes, one instance per model key."""

    spec: ModelSpec

    def load(self, device: str) -> LoadedModel:
        """Materialize the weights on ``device``; never called at import time."""
        ...

    def analytic_state_bytes(self, context: int) -> int:
        """Bytes of running per-example state at ``context`` tokens, analytically."""
        ...

    def forward(self, tokens: object) -> object:
        """One model forward over ``tokens`` -- the unit ``nfe`` counts."""
        ...


def _specs() -> tuple[ModelSpec, ...]:
    """The registry table: measured constants, so import never stats a file.

    ``params``/``weights_bytes`` were measured 2026-09-17 by summing the tensor
    headers and the on-disk weight files listed under ``RECORDED_MODELS_ROOT``
    (and, for BiRWKV, the release under this repository).  ``safetensors``
    checkpoints store every tensor once, so the parameter sums are exact;
    ``mamba2`` ties its embedding to the head, so its count is de-duplicated by
    storage rather than taken as the naive state-dict sum.
    """
    return (
        ModelSpec(
            key="llama-3.2-3b", family=TRANSFORMER, objective=AUTOREGRESSIVE,
            nfe=NFE_AUTOREGRESSIVE, params=3_212_749_824, weights_bytes=6_425_529_048,
            path="Llama-3.2-3B"),
        ModelSpec(
            key="qwen2.5-3b", family=TRANSFORMER, objective=AUTOREGRESSIVE,
            nfe=NFE_AUTOREGRESSIVE, params=3_085_938_688, weights_bytes=6_171_926_992,
            path="Qwen2.5-3B"),
        ModelSpec(
            key="llada-8b-base", family=TRANSFORMER, objective=MASKED_DIFFUSION,
            nfe=NFE_LLADA, params=8_015_581_184, weights_bytes=16_031_197_112,
            path="LLaDA-8B-Base"),
        ModelSpec(
            key="mamba2-2.7b", family=STATE_SPACE, objective=AUTOREGRESSIVE,
            nfe=NFE_AUTOREGRESSIVE, params=2_702_599_680, weights_bytes=5_405_424_282,
            path="mamba2-2.7b"),
        ModelSpec(
            key="rwkv7-goose-world3-2.9b", family=LINEAR_ATTENTION, objective=AUTOREGRESSIVE,
            nfe=NFE_AUTOREGRESSIVE, params=2_947_735_040, weights_bytes=5_895_584_608,
            path="RWKV7-Goose-World3-2.9B-HF"),
        ModelSpec(
            key="birwkv-2.9b", family=LINEAR_ATTENTION, objective=MASKED_DIFFUSION,
            nfe=NFE_BIRWKV, params=3_609_662_849, weights_bytes=7_219_660_451,
            path=_BIRWKV_RELEASE),
    )


#: The registry keyed by model key, importable without touching the disk.
REGISTRY: Final[dict[str, ModelSpec]] = {spec.key: spec for spec in _specs()}


#: Geometry per key.  Kept beside the registry rather than inside ``ModelSpec``
#: because the required spec fields are the protocol's, and a model's head
#: geometry is a property of the checkpoint, not of the comparison.
_GEOMETRY: Final[dict[str, AttentionGeometry | RecurrentGeometry]] = {
    "llama-3.2-3b": AttentionGeometry(
        n_layers=28, n_heads=24, hidden_size=3072, dtype_bytes=BF16_BYTES, kv_heads=8),
    "qwen2.5-3b": AttentionGeometry(
        n_layers=36, n_heads=16, hidden_size=2048, dtype_bytes=BF16_BYTES, kv_heads=2),
    "llada-8b-base": AttentionGeometry(
        n_layers=32, n_heads=32, hidden_size=4096, dtype_bytes=BF16_BYTES, kv_heads=32),
    "mamba2-2.7b": RecurrentGeometry(
        n_layers=64, n_heads=80, head_dim=64, state_dim=128, directions=1,
        dtype_bytes=FP16_BYTES),
    "rwkv7-goose-world3-2.9b": RecurrentGeometry(
        n_layers=32, n_heads=40, head_dim=64, state_dim=64, directions=1,
        dtype_bytes=BF16_BYTES),
    # The BiRWKV denoiser runs the RWKV backbone in both directions, so its
    # running state is carried twice -- the factor the protocol's closed form
    # names as "2 directions".
    "birwkv-2.9b": RecurrentGeometry(
        n_layers=32, n_heads=40, head_dim=64, state_dim=64, directions=2,
        dtype_bytes=BF16_BYTES),
}


def _absolute_path(spec: ModelSpec, models_root: str) -> Path:
    """A hub path is relative to ``models_root``; a recorded absolute path stands.

    The BiRWKV release is the one entry an environment variable may redirect, for the
    reason :data:`BIRWKV_RELEASE_ENV` gives: the measured checkpoint is a cluster
    endpoint rather than a shipped release.  The override applies to that key ONLY --
    an override that could redirect any key would make the registry's paths advisory,
    and a table built from advisory paths cannot be checked against the models it names.
    """
    if spec.key == BIRWKV_KEY:
        override = os.environ.get(BIRWKV_RELEASE_ENV)
        if override:
            return Path(override)
    recorded = Path(spec.path)
    return recorded if recorded.is_absolute() else Path(models_root) / recorded


def _birwkv_geometry_dir() -> Path:
    """The HF geometry directory for the BiRWKV denoiser, override first.

    Never resolved (stat'd) here: presence is the caller's check, so a missing directory surfaces
    as a refusal in :meth:`CheckpointAdapter._load_birwkv` that names the path rather than as an
    import-time filesystem probe.
    """
    override = os.environ.get(BIRWKV_GEOMETRY_ENV)
    return Path(override) if override else Path(BIRWKV_GEOMETRY_DIR)


def _birwkv_block_size() -> int:
    """The BiRWKV training-time block size, or 0 when the knob is unset.

    0 means "not supplied"; it is not a default block size. A non-integer or non-positive value is
    a caller defect and is refused rather than coerced -- a coerced block size would misalign every
    block's timestep, which no downstream number reveals.
    """
    raw = os.environ.get(BIRWKV_BLOCK_SIZE_ENV, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise AdapterRefusal(f"{BIRWKV_BLOCK_SIZE_ENV} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise AdapterRefusal(f"{BIRWKV_BLOCK_SIZE_ENV} must be positive, got {value}")
    return value


def _refuse_block_t_cond_without_block_size(keys: Iterable[object], block_size: int) -> None:
    """Refuse a timestep-conditioned checkpoint whose block size was not supplied.

    A checkpoint carrying ``block_t_cond.*`` was trained with a per-block noise level. Loading it
    without first attaching the arm makes those keys *unexpected* and the load dies before its first
    cell (gate (ii) attempt 1, job-74707291). Attaching it with a GUESSED block size is worse: the
    block size is a training-time contract, so a wrong value silently misaligns every block's ``t``
    with its positions and no downstream metric would reveal it. Hence a refusal that names the
    route out rather than a default.

    No-op either when no ``block_t_cond.*`` key is present or when ``block_size > 0``.
    """
    if block_size > 0:
        return
    for key in keys:
        if str(key).startswith("block_t_cond."):
            raise AdapterRefusal(
                "the checkpoint carries block_t_cond.* keys but no block size was supplied; the "
                "block size is a training-time contract and guessing it would misalign every "
                f"block's timestep. Set {BIRWKV_BLOCK_SIZE_ENV}=<training block size> to load it.")


def _read_birwkv_step(ckpt_dir: Path) -> int:
    """Read the checkpoint step from ``meta.json`` beside ``model.pt``.

    The measured endpoint's ``meta.json`` is ``{"step": 9500, "tokens_seen": 9961472000.0}``, and
    the manuscript labels every efficiency number by that step. An absent file, or a payload without
    ``step``, therefore refuses with the path rather than recording a fabricated ``0`` that would
    silently relabel the row -- the same fail-closed choice the block-size guard makes.
    """
    meta = ckpt_dir / "meta.json"
    if not meta.is_file():
        raise AdapterRefusal(
            f"the BiRWKV checkpoint at {ckpt_dir} carries no meta.json, so the step that produced "
            f"it cannot be recorded; every table row must be traceable to a checkpoint step")
    payload = json.loads(meta.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "step" not in payload:
        raise AdapterRefusal(
            f"the BiRWKV meta.json at {meta} carries no 'step'; every table row must be traceable "
            f"to the checkpoint step that produced it")
    return int(payload["step"])


def _load_birwkv_model(*, ckpt_dir: str, model_dir: str, device: str, block_size: int) -> object:
    """Build the BiRWKV geometry from the HF dir, then apply the flat training state dict.

    This reproduces the contract of the eval loader ``load_birwkv_diffusion`` in
    ``DAN/v7_arch_round/code/eval/birwkv_diffusion_model.py`` rather than importing it: that module
    pulls ``eval.capability.load_gate`` -- the shard-serializing host-RAM gate that exists only
    inside a multi-process GPU eval pod -- and loads a tokenizer this single-process memory probe
    never calls, so it is not on this tree's import surface. Every guard it documents is kept here,
    because each one cost a job:

    * a ``block_t_cond`` checkpoint without a block size is REFUSED (see
      :func:`_refuse_block_t_cond_without_block_size`), not guessed;
    * ``residual_streams.*`` / ``loop.*`` arms are attached BEFORE the state dict is applied, so
      their trained tensors map onto live modules instead of landing in ``unexpected`` -- the shape
      of the stored tensors recovers the config (``m_res_raw [L,n,n]`` gives ``n_streams``,
      ``gates_raw`` gives the trained reps, ``lo``/``hi`` give the loop range);
    * the state dict is cast tensor-by-tensor and the fp32 source dropped as it goes, so the fp32
      and bf16 copies are never both fully resident.

    ``model_dir`` supplies only the geometry (``config.json`` + safetensors); ``ckpt_dir`` supplies
    ``model.pt``. The two are separate paths because the measured endpoint ships no config of its
    own. ``torch`` and ``models.birwkv7_diffusion`` are imported here, inside the call, so importing
    this module stays a pure table read on a CPU box with no CUDA.
    """
    import torch
    from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion

    ckpt_path = Path(ckpt_dir)
    model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(model_dir, dtype=torch.bfloat16)
    state = torch.load(ckpt_path / "model.pt", map_location="cpu", weights_only=True)
    _refuse_block_t_cond_without_block_size(state.keys(), block_size)
    if any(str(k).startswith("block_t_cond.") for k in state):
        _ = model.attach_block_timestep_conditioner(default_block_size=block_size)
    if any(str(k).startswith("residual_streams.") for k in state):
        ns_v = int(state["residual_streams.m_res_raw"].shape[1])
        _ = model.attach_residual_streams(n_streams=ns_v)
        model.residual_streams = model.residual_streams.to(torch.bfloat16)
    if any(str(k).startswith("loop.") for k in state):
        lo_v = int(state["loop.lo"])
        hi_v = int(state["loop.hi"])
        reps_v = int(state["loop.gates_raw"].shape[0])
        _ = model.attach_backbone_loop((lo_v, hi_v), reps_v)
        model.loop = model.loop.to(torch.bfloat16)
    for key in list(state.keys()):
        state[key] = state[key].to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the fusion gates may be missing (they are fresh, warm-start-only parameters); anything
    # else is a mapping bug, and an unexpected key is a checkpoint that does not belong to this
    # architecture. Both abort before the checkpoint is measured.
    bad = [key for key in missing if "fuse_" not in key] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint/model mismatch: {bad[:8]}")
    del state  # release the bf16 copy before the next rung
    return model.to(device).eval()


def _mamba2_layer_indices(shapes: dict[str, tuple[int, ...]]) -> list[int]:
    """The mixer layer indices present in the state dict, in order.

    Read from the per-layer ``dt_bias`` keys rather than assumed equal to ``n_layer``, so a
    checkpoint whose tensor set disagrees with its own config is caught here instead of being built
    one layer short and measured as if it were complete.
    """
    prefix = "backbone.layers."
    suffix = ".mixer.dt_bias"
    indices: list[int] = []
    for name in shapes:
        if name.startswith(prefix) and name.endswith(suffix):
            middle = name[len(prefix):-len(suffix)]
            if middle.isdigit():
                indices.append(int(middle))
    return sorted(indices)


def _mamba2_hf_kwargs(
    config: dict[str, object],
    shapes: dict[str, tuple[int, ...]],
    *,
    geometry: RecurrentGeometry,
) -> dict[str, object]:
    """Translate the mamba_ssm config plus the measured tensor shapes into HF ``Mamba2Config`` kwargs.

    The release's ``config.json`` is the mamba_ssm format and names only ``ssm_cfg = {"layer":
    "Mamba2"}``: it carries none of ``d_state``/``headdim``/``expand``/``ngroups``, so the mixer
    geometry cannot be read from it and is recovered from the checkpoint's own tensor SHAPES
    instead. Deriving rather than hard-coding ties the architecture to the weights on disk -- a
    checkpoint that is not this model fails one of the identities below -- and the residual guess is
    named: ``n_groups * d_state`` is pinned by the shapes but its FACTORS are not, so ``d_state`` is
    taken from the recorded :class:`RecurrentGeometry` and the shapes are required to agree.

    Pure: it imports nothing and touches no disk, so the whole translation is exercisable on CPU.
    Every refusal below names the specific contradiction it caught, because a wrong mixer geometry
    either fails to load (loud) or, worse, loads with mis-grouped ``B``/``C`` and computes garbage
    silently -- the failure this function exists to make impossible.
    """
    if not isinstance(geometry, RecurrentGeometry):
        raise AdapterRefusal(
            f"mamba2 needs a RecurrentGeometry, got {geometry!r}; a constant recurrent state is the "
            f"whole point of the SSM row, and an attention geometry cannot express one")

    d_model = config.get("d_model")
    n_layer = config.get("n_layer")
    if not isinstance(d_model, int) or not isinstance(n_layer, int) or d_model <= 0 or n_layer <= 0:
        raise AdapterRefusal(
            "not the mamba_ssm Mamba2 config format: d_model/n_layer are absent, so the checkpoint "
            "does not describe an architecture this loader can rebuild")

    ssm_cfg = config.get("ssm_cfg")
    mixer_layer = ssm_cfg.get("layer") if isinstance(ssm_cfg, dict) else None
    if mixer_layer != "Mamba2":
        raise AdapterRefusal(
            f"ssm_cfg.layer is {mixer_layer!r}, not 'Mamba2'; only the Mamba2 mixer has the geometry "
            f"derived here")

    attn_layers = config.get("attn_layer_idx") or []
    if attn_layers:
        raise AdapterRefusal(
            f"attn_layer_idx is {attn_layers!r}; a hybrid Mamba2/attention model interleaves a block "
            f"type the uniform Mamba2 builder cannot represent")

    indices = _mamba2_layer_indices(shapes)
    if indices != list(range(n_layer)):
        raise AdapterRefusal(
            f"the state dict holds mixer layers {indices} but the config declares n_layer={n_layer}; "
            f"the two must agree for every layer to be built")

    embedding = shapes.get("backbone.embedding.weight")
    if embedding is None or len(embedding) != 2 or embedding[1] != d_model:
        raise AdapterRefusal(
            f"the state dict has no (vocab, {d_model}) backbone.embedding.weight; found {embedding!r}")
    # The embedding's ROW COUNT, not the config's vocab_size. The release pads the vocabulary to a
    # multiple of 16 (50277 -> 50288) and HF's Mamba2Config does not pad, so taking the config's
    # number would build a 50277-row embedding that the checkpoint's 50288-row tensor cannot load
    # into -- a shape error at load, never a silent wrong number.
    vocab_size = embedding[0]

    mixer = f"backbone.layers.{indices[0]}.mixer"

    def tensor(suffix: str) -> tuple[int, ...]:
        found = shapes.get(mixer + suffix)
        if found is None:
            raise AdapterRefusal(f"the state dict has no {mixer + suffix}")
        return found

    dt_bias = tensor(".dt_bias")
    num_heads = dt_bias[0]
    # ``A_log`` and ``D`` are per-head exactly like ``dt_bias``; a checkpoint where they disagree is
    # not the Mamba2 mixer these identities describe.
    for per_head in (".A_log", ".D"):
        found = tensor(per_head)
        if found != dt_bias:
            raise AdapterRefusal(
                f"{mixer + per_head} is {found!r}, not {dt_bias!r}; the per-head vectors of a "
                f"Mamba2 mixer are all num_heads long")
    # The gated RMSNorm (``mixer.norm``) is over the expanded width, unlike the block norm
    # (``.norm``, d_model); reading the wrong one would halve d_inner.
    d_inner = tensor(".norm.weight")[0]
    in_proj = tensor(".in_proj.weight")
    conv = tensor(".conv1d.weight")
    out_proj = tensor(".out_proj.weight")

    if len(conv) != 3 or conv[1] != 1:
        raise AdapterRefusal(
            f"{mixer + '.conv1d.weight'} is {conv!r}; a depthwise conv1d is (channels, 1, kernel)")
    conv_dim, _, conv_kernel = conv
    if len(in_proj) != 2 or in_proj[1] != d_model:
        raise AdapterRefusal(
            f"{mixer + '.in_proj.weight'} is {in_proj!r}; a Mamba2 in-proj maps (rows, {d_model})")
    if out_proj != (d_model, d_inner):
        raise AdapterRefusal(
            f"{mixer + '.out_proj.weight'} is {out_proj!r}, not {(d_model, d_inner)!r}")

    if num_heads <= 0 or d_inner % num_heads:
        raise AdapterRefusal(
            f"d_inner={d_inner} is not divisible by num_heads={num_heads}; the head width is not "
            f"recoverable")
    head_dim = d_inner // num_heads
    if d_inner % d_model:
        raise AdapterRefusal(f"d_inner={d_inner} is not a whole multiple of d_model={d_model}")
    expand = d_inner // d_model

    # group_width = conv_dim - d_inner = 2 * n_groups * d_state. The shapes pin the PRODUCT but not
    # its factors -- (n_groups=1, d_state=128) and (n_groups=8, d_state=16) are shape-identical and
    # group B/C differently -- so d_state is taken from the recorded geometry and the product is
    # required to divide by it. A wrong factor pair is the one error no shape check can see.
    group_width = conv_dim - d_inner
    if group_width <= 0 or group_width % 2:
        raise AdapterRefusal(
            f"conv_dim - d_inner = {group_width} is not a positive even number of B/C channels")
    if group_width % (2 * geometry.state_dim):
        raise AdapterRefusal(
            f"conv_dim - d_inner = {group_width} is not 2 * n_groups * state_dim for the recorded "
            f"state_dim={geometry.state_dim}; the recorded SSM state size contradicts the checkpoint")
    n_groups = group_width // (2 * geometry.state_dim)

    if in_proj[0] != 2 * d_inner + group_width + num_heads:
        raise AdapterRefusal(
            f"in_proj rows {in_proj[0]} != 2*{d_inner} + {group_width} + {num_heads}; the in-proj "
            f"packing is not the z, x, B, C, dt order this loader assumes")

    derived = (num_heads, head_dim, n_layer)
    recorded = (geometry.n_heads, geometry.head_dim, geometry.n_layers)
    if derived != recorded:
        raise AdapterRefusal(
            f"the checkpoint's tensors give (num_heads, head_dim, n_layer)={derived} but the recorded "
            f"geometry says {recorded}; the analytic state row would describe a different model than "
            f"the one loaded")

    return {
        "vocab_size": vocab_size,
        "hidden_size": d_model,
        "num_hidden_layers": n_layer,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "state_size": geometry.state_dim,
        "n_groups": n_groups,
        "expand": expand,
        "conv_kernel": conv_kernel,
        "rms_norm": bool(config.get("rms_norm", True)),
        "residual_in_fp32": bool(config.get("residual_in_fp32", True)),
        "tie_word_embeddings": bool(config.get("tie_embeddings", False)),
    }


def _load_mamba2_model(
    *, ckpt_dir: str, device: str, geometry: RecurrentGeometry
) -> object:
    """Build HF's ``Mamba2ForCausalLM`` from the mamba_ssm checkpoint and apply its weights.

    The mamba_ssm package is neither importable here nor needed: ``transformers`` ships a
    tensor-for-tensor port of the same Mamba2 mixer, so the checkpoint is loaded into that instead of
    into an unbuildable copy of the original. The state dict is renamed by the single
    :data:`MAMBA2_HF_KEY_RENAME` entry and checked key-by-key, so a checkpoint from another
    architecture or geometry is refused rather than silently half-applied.

    The model is built in fp16 BEFORE the state dict is applied. The checkpoint is fp16 and
    ``weights_bytes`` was measured at two bytes per parameter, so a default fp32 build would double
    the deployment the probe's peak reading is compared against; casting first keeps them the same
    object. ``torch`` and ``transformers`` are imported inside this call so importing the adapter
    module stays a pure table read on a CPU box with no CUDA (see the module docstring).
    """
    import torch
    from transformers import Mamba2Config, Mamba2ForCausalLM

    directory = Path(ckpt_dir)
    config = json.loads((directory / MAMBA2_CONFIG_FILE).read_text(encoding="utf-8"))
    weights_path = directory / MAMBA2_WEIGHTS_FILE
    if not weights_path.is_file():
        raise AdapterRefusal(
            f"the mamba2 checkpoint at {directory} carries no {MAMBA2_WEIGHTS_FILE}; the config "
            f"describes an architecture but there are no weights to measure")

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    shapes = {name: tuple(tensor.shape) for name, tensor in state.items()}
    kwargs = _mamba2_hf_kwargs(config, shapes, geometry=geometry)

    model = Mamba2ForCausalLM(Mamba2Config(**kwargs)).to(dtype=torch.float16)
    renamed = {MAMBA2_HF_KEY_RENAME.get(name, name): tensor for name, tensor in state.items()}
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    # Drop the fp16 source as soon as it has been copied into the model, so it is not resident
    # alongside the model through the rest of the run.
    del state, renamed
    bad = sorted(set(missing) | set(unexpected))
    if bad:
        raise AdapterRefusal(
            f"mamba2 checkpoint/model mismatch: {len(bad)} tensor(s) off, e.g. {bad[:8]}; the "
            f"checkpoint does not match the architecture derived from its own shapes")
    return model.to(device).eval()


#: The registry key of the LLaDA masked-diffusion Transformer, which ships its own modeling code
#: (``modeling_llada.LLaDAModelLM``, resolved through the checkpoint's ``auto_map``) rather than a
#: stock ``transformers`` architecture. Named so the dispatch can target it without a scattered
#: literal; its own load path exists because that code was written for transformers 4.x.
LLADA_KEY: Final = "llada-8b-base"

#: The remote modeling module and class the checkpoint's ``auto_map`` names. Resolved through
#: ``auto_map`` by :func:`_llada_base_model_class`, never imported at module scope.
LLADA_MODEL_REFERENCE: Final = "modeling_llada.LLaDAModelLM"

#: Tensors the LLaDA load is allowed to report missing or unexpected beyond a clean load.
#:
#: Both are EMPTY on purpose: the checkpoint and its remote ``modeling_llada`` are a matched pair, so
#: any nonzero count is a version incompatibility or a mapping bug, not an expected gap. Naming them --
#: rather than inlining empty sets at the check site -- keeps the contract explicit and gives a future,
#: justified exception exactly one place to be declared and reviewed.
LLADA_MISSING_ALLOWLIST: Final[frozenset[str]] = frozenset()
LLADA_UNEXPECTED_ALLOWLIST: Final[frozenset[str]] = frozenset()

#: The generation flag transformers 5.x drops from every config.
#:
#: ``PreTrainedConfig.__init__`` POPS the default generation parameters -- ``use_cache`` among them --
#: while a config is built, so ``config.use_cache`` no longer exists. LLaDA's remote forward reads it
#: unconditionally, so an otherwise-clean load would crash on the first forward. The value restored on
#: load is the one the checkpoint's own ``config.json`` declares; a checkpoint that asked for caching
#: is REFUSED rather than run uncached, because the table's unit is one uncached forward per answer.
#: Named ``*_FLAG`` and not ``*_KEY`` on purpose: the tests derive the set of registry keys with their
#: own loader from the module's ``*_KEY`` constants, and this is not a registry key.
LLADA_USE_CACHE_FLAG: Final = "use_cache"


def _llada_base_model_class(model_dir: str) -> type:
    """Resolve the checkpoint's own ``LLaDAModelLM`` from its remote modeling module.

    ``transformers`` is imported HERE, inside the call, so importing this module stays a pure table
    read on a CPU host without the runtime (see the module docstring). Resolving through the
    checkpoint's ``auto_map`` (via :data:`LLADA_MODEL_REFERENCE`) is what makes the model the
    CHECKPOINT's code rather than a stock ``transformers`` architecture.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(LLADA_MODEL_REFERENCE, model_dir)


def _llada_transformers5_compat_class(base: type) -> type:
    """Wrap the checkpoint's 4.x-era ``LLaDAModelLM`` in the transformers 5.x load contract.

    The remote ``modeling_llada.py`` was written for transformers 4.x, where ``all_tied_weights_keys``
    was established for the model and ``tie_weights`` took no arguments. Under 5.3.0 ``post_init`` is
    not called for it and ``tie_weights`` is called as ``tie_weights(missing_keys=..., recompute_mapping=...)``,
    so the unmodified class raises ``AttributeError: 'LLaDAModelLM' object has no attribute
    'all_tied_weights_keys'`` while ``from_pretrained`` FINALIZES a load whose tensors have all already
    been read. The subclass restores exactly those two contract points and changes no weight:

    * ``post_init()`` runs after construction, which is where 5.x sets ``all_tied_weights_keys`` (empty
      here -- the checkpoint ties nothing, ``weight_tying`` is false) and the rest of the tie bookkeeping;
    * ``tie_weights`` is widened to 5.x's signature and DELEGATES to the checkpoint's own 4.x body, so
      the config-driven module tie it implements is preserved rather than replaced by HF's default.

    Nothing here decides which tensors load. That judgment is :func:`_require_clean_llada_load`, which
    refuses unless the load reports zero missing/unexpected/shape-mismatched tensors -- so a shim that
    restored the attribute but mis-mapped weights would be refused, never measured.
    """
    class _LLaDACompatModel(base):  # type: ignore[misc, valid-type]
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            self.post_init()

        def tie_weights(self, missing_keys=None, recompute_mapping=True):
            # 5.x passes both; the checkpoint's 4.x body names neither. Delegate so its own
            # (config-driven) tying runs, rather than HF's default replacing it.
            return base.tie_weights(self)

    return _LLaDACompatModel


def _require_clean_llada_load(key: str, loading_info: Mapping[str, object]) -> None:
    """Refuse a LLaDA load unless every tensor mapped cleanly.

    ``from_pretrained`` reports what it did in ``loading_info``: tensors the model expected but the
    checkpoint lacked (``missing_keys``), tensors the checkpoint had that the model did not
    (``unexpected_keys``), and tensors whose shapes disagreed (``mismatched_keys``). A nonzero count
    past the declared allowlists means the shim did not make the 4.x code load under 5.x but merely
    stopped it crashing -- and a half-loaded model in an efficiency table is worse than a missing row,
    because the number would look measured. So the load is refused, and the launcher records a
    ``load_failed`` row carrying this reason.
    """
    missing = {str(name) for name in (loading_info.get("missing_keys") or ())}
    unexpected = {str(name) for name in (loading_info.get("unexpected_keys") or ())}
    mismatched = {str(entry[0]) for entry in (loading_info.get("mismatched_keys") or ())}

    leftover_missing = sorted(missing - LLADA_MISSING_ALLOWLIST)
    leftover_unexpected = sorted(unexpected - LLADA_UNEXPECTED_ALLOWLIST)
    leftover_mismatched = sorted(mismatched)
    if not (leftover_missing or leftover_unexpected or leftover_mismatched):
        return
    raise AdapterRefusal(
        f"{key}: the LLaDA load did not map every tensor (transformers 5.x vs the checkpoint's 4.x "
        f"remote code): {len(leftover_missing)} missing, {len(leftover_unexpected)} unexpected, "
        f"{len(leftover_mismatched)} shape-mismatched. A partially-loaded model would put a number "
        f"that was never measured into the efficiency table, so the row is refused instead. "
        f"missing={leftover_missing[:8]} unexpected={leftover_unexpected[:8]} "
        f"mismatched={leftover_mismatched[:8]}")


def _restore_dropped_use_cache(config: object, config_path: Path) -> None:
    """Restore the ``use_cache`` flag transformers 5.x drops from the config.

    ``PreTrainedConfig.__init__`` pops the default generation parameters while a config is built, so
    ``config.use_cache`` no longer exists -- but LLaDA's remote forward reads it unconditionally, so an
    otherwise-clean load crashes on the first forward. The value restored is NOT guessed: it is read
    from the checkpoint's own ``config.json``. A checkpoint that declares caching is refused rather
    than silently run uncached, because caching changes what an answer costs and the table's unit is a
    fixed number of forwards.
    """
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    declared = payload.get(LLADA_USE_CACHE_FLAG) if isinstance(payload, dict) else None
    if declared:
        raise AdapterRefusal(
            f"the LLaDA checkpoint at {config_path} declares {LLADA_USE_CACHE_FLAG}={declared!r}, but "
            f"transformers 5.x drops generation flags from the config and the probe measures one "
            f"uncached forward per answer; running it cached would change the measurement's unit")
    setattr(config, LLADA_USE_CACHE_FLAG, False)


def _load_llada_model(*, model_dir: str, device: str) -> object:
    """Build LLaDA's remote model class under the transformers 5.x load contract and apply its weights.

    Three steps, in order, each observable:

    1. the checkpoint's own ``LLaDAModelLM`` is resolved from its ``auto_map`` (4.x-era code);
    2. it is built through the 5.x-compatible subclass (:func:`_llada_transformers5_compat_class`) and
       loaded with ``output_loading_info=True`` so the tensor mapping can be INSPECTED;
    3. the reported mapping is required to be clean (:func:`_require_clean_llada_load`) before the model
       is moved to ``device`` -- a shim that silenced the crash without mapping every tensor is refused
       here rather than measured.

    ``dtype="auto"`` builds the model in the checkpoint's stored width (bf16), the deployment the row's
    ``weights_bytes`` describes; the generation flag transformers 5.x drops is restored from the
    checkpoint's config so the first forward can run. ``torch`` and the remote class are imported here,
    inside the call, so this module stays a pure table read on a CPU box with no CUDA.
    """
    import torch

    base = _llada_base_model_class(model_dir)
    compat = _llada_transformers5_compat_class(base)
    model, loading_info = compat.from_pretrained(
        model_dir, trust_remote_code=True, output_loading_info=True, dtype="auto")
    _require_clean_llada_load(LLADA_KEY, loading_info)
    _restore_dropped_use_cache(model.config, Path(model_dir) / "config.json")
    return model.to(torch.device(device)).eval()


@dataclass(slots=True)
class CheckpointAdapter:
    """The single adapter implementation, parameterized by spec and geometry."""

    spec: ModelSpec
    geometry: AttentionGeometry | RecurrentGeometry
    resolved_path: Path
    _model: object | None = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False, repr=False)
    #: The checkpoint step, recorded on load for a model whose release carries a ``meta.json``.
    _step: int | None = field(default=None, init=False, repr=False)

    @property
    def step(self) -> int | None:
        """The loaded checkpoint's training step, or ``None`` before load / for a hub model."""
        return self._step

    def analytic_state_bytes(self, context: int) -> int:
        """Running state in bytes at ``context`` tokens, from the closed forms.

        Transformer (KV cache, linear in context)::

            heads x head_dim x layers x 2 x context x dtype_bytes
            = 2 x layers x (hidden_size x kv_heads // heads) x context x dtype_bytes

        The right-hand form is the same quantity with the per-head width folded
        in; ``kv_heads < heads`` (GQA/MQA) shrinks it by ``kv_heads / heads``,
        which is why the registry records the real ``kv_heads`` instead of
        assuming full multi-head attention.

        Recurrent (linear attention / SSM, constant in context)::

            directions x layers x heads x head_dim x head_dim x dtype_bytes

        The square form assumes the state's key and value axes are both
        ``head_dim``, which is the RWKV WKV shape.  The value axis is kept
        explicit (``state_dim``) so a rectangular SSD state -- mamba2's
        ``head_dim x d_state`` -- is taken over both axes rather than squared.

        Both are arithmetic from the config, not measurements; the paper must
        not present them as measured.
        """
        if context <= 0:
            raise AdapterRefusal(f"{self.spec.key}: context must be positive, got {context}")
        # Imported here, not at module scope: ``theory_bounds`` imports torch,
        # and importing torch at module scope would initialise CUDA and make the
        # registry un-listable on a login node (see the module docstring).
        from ..theory_bounds import kv_cache_bytes, recurrent_state_bytes

        geometry = self.geometry
        if isinstance(geometry, AttentionGeometry):
            return kv_cache_bytes(
                n_layers=geometry.n_layers, hidden_size=geometry.hidden_size,
                seq_len=context, dtype_bytes=geometry.dtype_bytes,
                kv_heads=geometry.kv_heads, n_heads=geometry.n_heads)
        if not isinstance(geometry, RecurrentGeometry):
            raise AdapterRefusal(f"{self.spec.key}: unrecognized geometry {geometry!r}")
        if geometry.state_dim == geometry.head_dim:
            return recurrent_state_bytes(
                n_layers=geometry.n_layers, n_heads=geometry.n_heads,
                head_dim=geometry.head_dim, directions=geometry.directions,
                dtype_bytes=geometry.dtype_bytes)
        return (geometry.directions * geometry.n_layers * geometry.n_heads
                * geometry.head_dim * geometry.state_dim * geometry.dtype_bytes)

    def load(self, device: str) -> LoadedModel:
        """Materialize the checkpoint on ``device``.

        Deliberately not run at import time: the registry is a table of constants and the matrix
        must be enumerable without loading -- or even reading -- any weights. ``transformers`` and
        ``torch`` are imported inside the branch that needs them, for that reason.

        Three keys have their own load paths and the rest of the registry goes through the generic hub
        loader.  ``birwkv-2.9b`` cannot take the hub path because its checkpoint is a flat ``model.pt``
        plus ``meta.json`` with no ``config.json`` at all, so it takes :meth:`_load_birwkv`.
        ``mamba2-2.7b`` cannot either because its ``config.json`` is the mamba_ssm format with no
        ``model_type``, so ``AutoModelForCausalLM`` cannot resolve an architecture for it; it takes
        :meth:`_load_mamba2`, which rebuilds the same mixer from HF's tensor-for-tensor port.
        ``llada-8b-base`` IS a hub layout, but its ``auto_map`` points at remote modeling code written
        for transformers 4.x that the installed 5.3.0 cannot finish loading, so it takes
        :meth:`_load_llada`, which restores the 5.x contract and then REFUSES unless the load reports
        every tensor mapped.  A checkpoint that ships no modeling code and is NOT one of these still
        refuses with the message below rather than being routed somewhere that would invent an
        architecture for it.
        """
        if self.spec.key == BIRWKV_KEY:
            return self._load_birwkv(device)
        if self.spec.key == MAMBA2_KEY:
            return self._load_mamba2(device)
        if self.spec.key == LLADA_KEY:
            return self._load_llada(device)
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise AdapterRefusal(
                f"{self.spec.key}: the torch/transformers runtime is unavailable ({exc}), "
                f"so no efficiency measurement can be made") from exc
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.resolved_path), trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 - surfaced as a named refusal below
            raise AdapterRefusal(
                f"{self.spec.key}: could not be built by the generic hub loader; a "
                f"checkpoint that ships no modeling code (the BiRWKV release) or no "
                f"model_type needs its own loader. Underlying error: {exc}") from exc
        model = model.to(torch.device(device)).eval()
        self._model = model
        self._device = device
        return LoadedModel(spec=self.spec, path=str(self.resolved_path), device=device, model=model)

    def _load_mamba2(self, device: str) -> LoadedModel:
        """Build the mamba2 checkpoint through its own loader -- the hub loader cannot resolve it.

        The release ships a mamba_ssm-format ``config.json`` with no ``model_type``, so
        ``AutoModelForCausalLM`` refuses it (commit 5038e1d); the architecture is rebuilt instead from
        HF's port of the same mixer, against which every checkpoint tensor maps (see
        :data:`MAMBA2_HF_KEY_RENAME`). Two things are checked before the weights are touched:

        * the config file is present, and its absence refuses by path -- "wrong model_dir" and "the
          checkpoint is not downloaded" are different faults, and the adapter must let a reader tell
          them apart;
        * the geometry is a :class:`RecurrentGeometry`, because this row's whole claim is a CONSTANT
          recurrent state and an attention geometry would silently turn it into a growing cache.

        The heavy work lives in :func:`_load_mamba2_model`, whose import failure is turned into a
        refusal naming the missing runtime rather than a raw ``ImportError`` a caller cannot classify.
        """
        config_path = self.resolved_path / MAMBA2_CONFIG_FILE
        if not config_path.is_file():
            raise AdapterRefusal(
                f"{self.spec.key}: the mamba_ssm-format config.json is absent at {config_path}; "
                f"without it the checkpoint's mixer geometry cannot be read, so it cannot be built")
        if not isinstance(self.geometry, RecurrentGeometry):
            raise AdapterRefusal(
                f"{self.spec.key}: declared geometry {self.geometry!r} is not recurrent; a "
                f"state-space model's state is constant in context and cannot be modelled otherwise")
        try:
            model = _load_mamba2_model(
                ckpt_dir=str(self.resolved_path), device=device, geometry=self.geometry)
        except AdapterRefusal:
            raise
        except ImportError as exc:
            raise AdapterRefusal(
                f"{self.spec.key}: the torch/transformers runtime needed to build "
                f"transformers.Mamba2ForCausalLM is unavailable ({exc}); the mamba_ssm package is "
                f"also absent, so there is no other builder for this checkpoint") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a named refusal below
            raise AdapterRefusal(
                f"{self.spec.key}: the mamba2 loader failed on checkpoint {self.resolved_path}. "
                f"Underlying error: {exc}") from exc
        self._model = model
        self._device = device
        return LoadedModel(spec=self.spec, path=str(self.resolved_path), device=device, model=model)

    def _load_llada(self, device: str) -> LoadedModel:
        """Build LLaDA through its own transformers-5.x-compatible load path.

        ``llada-8b-base`` is a normal hub layout whose ``auto_map`` points ``AutoModelForCausalLM`` at
        remote modeling code (``modeling_llada.LLaDAModelLM``) written for transformers 4.x. Under the
        installed 5.3.0 that class cannot be finished loading -- it raises ``all_tied_weights_keys``
        AFTER every tensor has been read -- so, unlike the BiRWKV and mamba2 rows, the fault is a
        version shift in the load contract, not a missing ``config.json`` or ``model_type``. The heavy
        work lives in :func:`_load_llada_model`, which restores the contract and then refuses unless
        the load reports zero missing/unexpected/mismatched tensors; an import failure is turned into a
        refusal naming the missing runtime rather than a raw ``ImportError`` a caller cannot classify.
        """
        try:
            model = _load_llada_model(model_dir=str(self.resolved_path), device=device)
        except AdapterRefusal:
            raise
        except ImportError as exc:
            raise AdapterRefusal(
                f"{self.spec.key}: the torch/transformers runtime or the checkpoint's remote modeling "
                f"code (modeling_llada) is unavailable ({exc}), so the LLaDA denoiser cannot be built"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a named refusal below
            raise AdapterRefusal(
                f"{self.spec.key}: the LLaDA loader failed on checkpoint {self.resolved_path}. "
                f"Underlying error: {exc}") from exc
        self._model = model
        self._device = device
        return LoadedModel(spec=self.spec, path=str(self.resolved_path), device=device, model=model)

    def _load_birwkv(self, device: str) -> LoadedModel:
        """Build the BiRWKV denoiser from the HF geometry dir plus the flat endpoint checkpoint.

        Three things happen before the weights are touched, each a named refusal:

        * the HF geometry directory is resolved (env override first) and its ABSENCE is refused with
          the path it looked for -- "wrong model_dir" and "model not downloaded" are different
          faults, and the adapter must let a reader tell them apart;
        * the checkpoint step is read from ``meta.json``, because the manuscript labels every number
          by the step that produced it and a row without one is not usable;
        * the block size is read from its own knob, so a ``block_t_cond`` checkpoint is refused
          rather than mis-timed (the refusal lives in :func:`_load_birwkv_model`).
        """
        geometry_dir = _birwkv_geometry_dir()
        if not geometry_dir.exists():
            raise AdapterRefusal(
                f"{self.spec.key}: the HF geometry directory (config.json + safetensors) is absent "
                f"at {geometry_dir}; the measured checkpoint is a flat model.pt that carries no "
                f"config of its own, so the architecture cannot be built without it. Set "
                f"{BIRWKV_GEOMETRY_ENV} to the RWKV7-Goose-World3-2.9B-HF directory.")
        step = _read_birwkv_step(self.resolved_path)
        block_size = _birwkv_block_size()
        try:
            model = _load_birwkv_model(
                ckpt_dir=str(self.resolved_path), model_dir=str(geometry_dir),
                device=device, block_size=block_size)
        except AdapterRefusal:
            raise
        except ImportError as exc:
            raise AdapterRefusal(
                f"{self.spec.key}: the torch/fla runtime or the model code "
                f"(models.birwkv7_diffusion, on the DAN/v7_arch_round/code import root) is "
                f"unavailable ({exc}), so the BiRWKV denoiser cannot be built") from exc
        except Exception as exc:  # noqa: BLE001 - surfaced as a named refusal below
            raise AdapterRefusal(
                f"{self.spec.key}: the BiRWKV loader failed on checkpoint {self.resolved_path} "
                f"with geometry {geometry_dir}. Underlying error: {exc}") from exc
        self._model = model
        self._device = device
        self._step = step
        return LoadedModel(spec=self.spec, path=str(self.resolved_path), device=device,
                           model=model, step=step)

    def forward(self, tokens: object) -> object:
        """One forward over ``tokens``; the unit ``nfe`` counts.

        Refuses before ``load`` rather than silently materializing weights, so a
        measurement cannot be taken against a model the caller never asked for.
        """
        if self._model is None:
            raise AdapterRefusal(f"{self.spec.key}: load(device) must be called before forward()")
        return self._model(tokens)


def build(key: str, *, models_root: str) -> Adapter:
    """Resolve ``key`` to an adapter whose weights are present under ``models_root``.

    Two refusals are distinct on purpose.  An unknown key names the key, so a
    typo surfaces as the typo.  A known key whose weights are absent names the
    path, so a missing checkpoint cannot be scored as a present one -- the whole
    matrix would otherwise be a comparison against a model that was never there.
    """
    spec = REGISTRY.get(key)
    if spec is None:
        raise AdapterRefusal(
            f"unknown model key {key!r}; the registry declares {sorted(REGISTRY)}")
    geometry = _GEOMETRY[key]
    resolved = _absolute_path(spec, models_root)
    if not resolved.exists():
        hint = (f"; set {BIRWKV_RELEASE_ENV} to the measured checkpoint's directory"
                if key == BIRWKV_KEY else "check models_root")
        raise AdapterRefusal(
            f"weights for {key!r} are absent at {resolved}{hint}; models_root is "
            f"currently {models_root!r}")
    return CheckpointAdapter(spec=spec, geometry=geometry, resolved_path=resolved)


__all__ = [
    "AUTOREGRESSIVE", "Adapter", "AdapterRefusal", "AttentionGeometry", "BF16_BYTES",
    "CheckpointAdapter", "FAMILIES", "FP16_BYTES", "LINEAR_ATTENTION", "LoadedModel",
    "BIRWKV_BLOCK_SIZE_ENV", "BIRWKV_GEOMETRY_DIR", "BIRWKV_GEOMETRY_ENV",
    "BIRWKV_KEY", "BIRWKV_RELEASE_ENV", "LLADA_KEY", "LLADA_MISSING_ALLOWLIST", "LLADA_MODEL_REFERENCE",
    "LLADA_UNEXPECTED_ALLOWLIST", "MAMBA2_KEY", "MASKED_DIFFUSION", "ModelSpec", "NFE_AUTOREGRESSIVE", "NFE_BIRWKV", "NFE_LLADA",
    "OBJECTIVES", "RECORDED_MODELS_ROOT", "REGISTRY", "RecurrentGeometry", "STATE_SPACE",
    "TRANSFORMER", "build",
]
