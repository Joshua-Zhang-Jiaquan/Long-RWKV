"""Canonical, source-bound partial task-3 contract construction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Final

from .architecture_evidence import load_lifecycle_evidence, load_runtime_evidence, load_semantics_evidence
from .architecture_evidence_models import RuntimeQualification
from .architecture_cpu_evidence import load_cpu_evidence

CONTRACT_VERSION: Final = 2
CONTRACT_STATUS: Final = "BLOCKED_EXTERNAL"
CPU_STATUS: Final = "ENGINEERING_ONLY"
SOURCE_FILES: Final = {
    "model": "DAN/v7_arch_round/code/models/birwkv7_diffusion.py",
    "residual_streams": "DAN/v7_arch_round/code/models/residual_streams.py",
    "trainer": "DAN/v7_arch_round/code/train/train_birwkv_diffusion.py",
    "architecture_spec": "DAN/v7_arch_round/ARCHITECTURE.md",
    "run_spec": "DAN/v7_arch_round/report.md",
}
HELPER_PATHS: Final = (
    "DAN/v7_arch_round/code/models/latent_plan.py",
    "DAN/v7_arch_round/code/models/state_hijacking_cache.py",
    "DAN/v7_arch_round/code/models/state_hijacking_dit_torch_types.py",
)
CHECKPOINT_PATHS: Final = (
    "DAN/v7_arch_round/checkpoint_config.yaml",
    "DAN/v7_arch_round/model.safetensors.index.json",
)


def repo_default() -> Path:
    """Locate the repository from the task-local package path."""
    return Path(__file__).parents[3]


def contract_payload(repo_root: Path | None = None) -> dict[str, object]:
    """Derive the complete closed partial-contract payload from current safe inputs."""
    root = repo_default() if repo_root is None else repo_root.resolve()
    sources = {name: _source_identity(root, relative, "staged_source") for name, relative in SOURCE_FILES.items()}
    fla = _fla_identity()
    runtime = load_runtime_evidence(root)
    lifecycle = load_lifecycle_evidence(root)
    cpu = load_cpu_evidence(root)
    semantics = load_semantics_evidence(root)
    blockers = _blockers(root, fla, runtime)
    return {
        "contract_version": CONTRACT_VERSION,
        "status": CONTRACT_STATUS,
        "cpu_check_status": CPU_STATUS,
        "readiness": {"blocked_claims": len(blockers), "whole_architecture_ready": False},
        "sources": sources,
        "fla": fla,
        "runtime_evidence": None if runtime is None else runtime.model_dump(mode="json"),
        "lifecycle_v7_evidence": None if lifecycle is None else lifecycle.model_dump(mode="json"),
        "cpu_source_evidence": None if cpu is None else cpu.model_dump(mode="json"),
        "semantics_v8_evidence": None if semantics is None else semantics.model_dump(mode="json"),
        "task3_acceptance": {
            "nonlatent_full_canvas_path": "source_only" if lifecycle is None else "satisfied_bounded_gpu",
            "lifecycle_session_isolation": "engineering_only" if lifecycle is None else "satisfied_bounded_gpu",
            "actual_parameter_category_counts": "unavailable" if lifecycle is None else "satisfied_observed_v7_ranks",
            "block_evaluations_vs_outer_nfe": "source_only" if cpu is None else "satisfied_source_hash_bound_cpu_32_plus_16",
            "initialization_and_loop_identity": "source_only" if cpu is None else "satisfied_source_hash_bound_cpu",
            "full_canvas_vs_streaming": "distinct_interfaces_streaming_unsupported",
            "active_optimizer_membership_and_gradient_policy": "open_source_bound_cpu_only" if semantics is None else "satisfied_loaded_model_two_freeze_configs_stage_a_only",
            "active_input_output_mask_loss_semantics": "open_cpu_source_and_finite_gpu_forward_only" if semantics is None else "satisfied_tested_corruption_rank_local_actual_logit_loss_backward",
            "full_model_ffn_cache": "unsupported_unqualified_not_required_unless_enabled",
            "tied_loop_cache": "unsupported_unqualified_not_required_unless_enabled",
        },
        "historical_source_inventory": [
            {"path": relative, "present": (root / relative).is_file(), "role": "historical_snapshot_only"}
            for relative in (*HELPER_PATHS, *CHECKPOINT_PATHS)
        ],
        "claims": {
            "attention_pair": "clone_not_tied_source_backed",
            "backbone_loop": {"base_block_passes": 32, "recycled_block_passes": 16, "total_block_passes": 48},
            "outer_denoising_nfe": "distinct_full_canvas_forwards_source_backed",
            "freeze_optimizer": "source_selection_semantics_only" if semantics is None else "qualified_loaded_model_membership_two_freeze_configs_stage_a_no_optimizer_step",
            "trainer_cpu_semantics": "unavailable" if cpu is None else "source_hash_bound_cpu_verified_not_active_model",
            "initialization_cpu": "unavailable" if cpu is None else "source_hash_bound_cpu_verified",
            "three_clocks": "source_only" if cpu is None else "token_recurrence_block_32_plus_16_outer_nfe_distinct_cpu_verified",
            "cache_runtime": "unavailable",
            "masks": "unavailable" if semantics is None else "qualified_tested_corruption_actual_logits_rank_local_loss_backward",
            "state_owner": "engineering_control_only" if lifecycle is None else "bounded_gpu_full_canvas_lifecycle_qualified",
            "parameter_categories": "unavailable" if lifecycle is None else "observed_v7_rank_parameter_identities",
            "nonlatent_path": "source_only" if lifecycle is None else "bounded_gpu_full_canvas_no_cache",
            "full_checkpoint_count": "unavailable" if runtime is None else "qualified_deployed_checkpoint_only",
            "standalone_fla_cache": "unavailable" if runtime is None else "qualified_standalone_forward_rwkv7_attention_only",
            "masked_forward": "unavailable" if runtime is None else "finite_shape_only_not_mask_semantics",
            "full_model_prefix_cache": "unavailable" if runtime is None else runtime.full_model_prefix_cache,
            "loop_prefix_cache": "unavailable" if runtime is None else runtime.loop_prefix_cache,
            "optional_latent": "not_an_active_nonlatent_assumption",
        },
        "blockers": blockers,
    }


def contract_json(repo_root: Path | None = None) -> str:
    """Serialize the canonical current contract deterministically."""
    return _json_text(contract_payload(repo_root))


def receipt_payload(repo_root: Path, contract: str) -> dict[str, object]:
    """Bind a publication receipt to exact contract bytes and source manifest."""
    payload = contract_payload(repo_root)
    return {
        "receipt_version": 1,
        "status": CONTRACT_STATUS,
        "cpu_check_status": CPU_STATUS,
        "blocked_claims": blocked_claim_count(repo_root),
        "contract_sha256": _sha256_text(contract),
        "source_manifest_sha256": _sha256_text(_json_text(payload["sources"])),
    }


def receipt_json(repo_root: Path, contract: str) -> str:
    """Serialize a canonical contract receipt."""
    return _json_text(receipt_payload(repo_root, contract))


def blocked_claim_count(repo_root: Path) -> int:
    """Count currently absent external assets without treating installed source as runtime proof."""
    root = repo_root.resolve()
    return len(_blockers(root, _fla_identity(), load_runtime_evidence(root)))


def blocker_paths(repo_root: Path) -> tuple[str, ...]:
    """Expose dynamic exact blocker paths for a blocked-only public result."""
    root = repo_root.resolve()
    return tuple(blocker["path"] for blocker in _blockers(root, _fla_identity(), load_runtime_evidence(root)))


def _source_identity(root: Path, relative: str, role: str) -> dict[str, str]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return {"path": relative, "role": role, "sha256": _sha256_path(path)}


def _fla_identity() -> dict[str, object]:
    try:
        distribution = importlib.metadata.distribution("flash-linear-attention")
    except importlib.metadata.PackageNotFoundError:
        return {"availability": "not_installed", "distribution": "flash-linear-attention", "sources": [], "version": "unavailable"}
    spec = importlib.util.find_spec("fla")
    locations = None if spec is None else spec.submodule_search_locations
    root = Path(next(iter(locations))) if locations else None
    files = tuple(sorted(path for path in root.rglob("*rwkv7*.py") if path.is_file())) if root is not None else ()
    sources = [{"path": str(path), "sha256": _sha256_path(path)} for path in files]
    availability = "installed_source_available" if sources else "installed_source_unavailable"
    return {"availability": availability, "distribution": "flash-linear-attention", "sources": sources, "version": distribution.version}


def _blockers(root: Path, fla: dict[str, object], runtime: RuntimeQualification | None) -> list[dict[str, str]]:
    if runtime is not None:
        blockers = [
            {"kind": "full_model_prefix_cache", "path": runtime.full_model_prefix_cache},
            {"kind": "loop_prefix_cache", "path": runtime.loop_prefix_cache},
            {"kind": "long_context_performance", "path": "no long-context, performance or goodput proof"},
            {"kind": "image_identity", "path": "observed image digest unknown; no arbitrary torch binary identity"},
        ]
        if load_lifecycle_evidence(root) is None:
            blockers.append({"kind": "state_owner", "path": "canvas invalidation and cross-sample state isolation remain engineering-only"})
        if load_semantics_evidence(root) is None:
            blockers.extend([
                {"kind": "freeze_optimizer", "path": "active/model-bound optimizer membership and gradient policy remain open; historical CPU source semantics are not active-model observations"},
                {"kind": "masks", "path": "active input/output mask-loss semantics remain open; historical CPU tests and finite GPU logits do not close them"},
            ])
        return blockers
    blockers: list[dict[str, str]] = []
    for relative in HELPER_PATHS:
        if not (root / relative).is_file():
            blockers.append({"kind": "missing_caller_helper", "path": relative})
    for relative in CHECKPOINT_PATHS:
        if not (root / relative).is_file():
            blockers.append({"kind": "missing_checkpoint_identity", "path": relative})
    if fla["availability"] != "installed_source_available":
        blockers.append({"kind": "fla_source", "path": str(fla["availability"])})
    blockers.append({"kind": "unexecuted_fla_runtime", "path": "GPU kernel/cache equivalence authorization and runtime"})
    return blockers


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
