from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = 2
SNAPSHOT_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("R1", "README.md", "document"),
    ("R1", "ARCHITECTURE.md", "document"),
    ("R1", "TRAINING_HISTORY.md", "document"),
    ("R1", "report.md", "document"),
    ("R2", "results/gate_analysis_s4750_20260910.txt", "raw_record"),
    ("R2", "results/ability_curve_m4loop.csv", "raw_record"),
    ("R2", "results/ability_curve_m4mhc.csv", "raw_record"),
    ("R3", "scripts/wl_gate_analysis.py", "code"),
    ("R4", "code/eval/sampler_eval_offline.py", "code"),
    ("R5", "code/models/birwkv7_diffusion.py", "code"),
    ("R5", "code/models/residual_streams.py", "code"),
    ("R5", "code/train/test_backbone_loop.py", "code"),
    ("R6", "code/eval/lm_eval.py", "code"),
    ("R6", "code/train/train_birwkv_diffusion.py", "code"),
    ("R6", "specs/m4_loop_32gpu_half.json", "manifest"),
    ("R6", "JOBS.md", "document"),
)
PANEL_PATHS: tuple[str, ...] = (
    "cap_lm_m4loop_s4750",
    "cap_lm_n2_s4000",
    "cap_lm_n2_s6000",
    "cap_lm_m4loop_s4750_reps2",
    "cap_lm_m4loop_s4750_reps3",
    "cap_lm_m4loop_s4750_reps4",
    "sampler_gate_m4_loop_s4750",
    "sampler_gate_m4_n2_s4000",
    "sampler_gate_m4_n2_s6000",
    "sampler_gate_m4_loop_s4750_reps2",
    "sampler_gate_m4_loop_s4750_reps3",
    "sampler_gate_m4_loop_s4750_reps4",
    "m2_baseline_triangle/ability_curve_m4loop.csv",
)
METADATA_PATHS: tuple[str, ...] = (
    "outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/meta.json",
    "outputs_birwkv_diffusion/n2-knowpt-2p9b/step_00006000_probe_copy/meta.json",
    "models/RWKV7-Goose-World3-2.9B-HF/tokenizer_config.json",
)
ARTIFACT_CONTRACT = frozenset(
    [(source, "snapshot", path, kind) for source, path, kind in SNAPSHOT_ARTIFACTS]
    + [("R3", "external", path, "raw_record") for path in PANEL_PATHS]
    + [("R6", "external", path, "metadata") for path in METADATA_PATHS]
)


@dataclass(frozen=True, slots=True)
class ClaimContract:
    claim_id: str
    source: str
    records: tuple[str, ...]
    fields: tuple[str, ...]


GATE_RECORD = "results/gate_analysis_s4750_20260910.txt"
LM1B_RECORD = "cap_lm_m4loop_s4750/lm1b/merged/merged_m4loop_endpoint_ckpt_nll-ppl-bits.json"
WIKITEXT_RECORD = "cap_lm_m4loop_s4750/wikitext103/merged/merged_m4loop_endpoint_ckpt_nll-ppl-bits.json"
SAMPLER_RECORD = "sampler_gate_m4_loop_s4750/merged/merged_m4loop_endpoint_ckpt_em-tau-residue-max_run_frac-distinct_frac.json"
LOOP_METADATA = "outputs_birwkv_diffusion/m4-loop-2p9b/step_00004750/meta.json"
ABILITY_RECORD = "m2_baseline_triangle/ability_curve_m4loop.csv"
CLAIM_CONTRACTS: tuple[ClaimContract, ...] = (
    ClaimContract("lm1b_loss_ppl", "R3", (LM1B_RECORD, GATE_RECORD), ("record_count", "records[].metrics.nll", "records[].metrics.ppl")),
    ClaimContract("wikitext103_loss_ppl", "R3", (WIKITEXT_RECORD, GATE_RECORD), ("record_count", "records[].metrics.nll", "records[].metrics.ppl")),
    ClaimContract("causal_path_comparison", "R3", (LM1B_RECORD, WIKITEXT_RECORD), ("records[].arm", "records[].document_id", "records[].metrics.nll")),
    ClaimContract("capacity_4_98b", "R6", (LOOP_METADATA, ABILITY_RECORD), ("step", "tokens_seen", "csv.tokens_seen", "csv.score")),
    ClaimContract("max_run_frac_direction", "R3", (SAMPLER_RECORD, GATE_RECORD), ("records[].arm", "records[].metrics.max_run_frac", "delta", "ci95")),
    ClaimContract("masked_token_accuracy", "R3", (SAMPLER_RECORD, GATE_RECORD), ("records[].arm", "records[].metrics.em", "delta", "ci95")),
    ClaimContract("tau_commit_order", "R3", (SAMPLER_RECORD, GATE_RECORD), ("records[].arm", "records[].metrics.tau", "delta", "ci95")),
    ClaimContract("r100_fully_masked", "R3", (SAMPLER_RECORD, GATE_RECORD), ("records[].arm", "records[].metrics", "delta", "ci95")),
)
