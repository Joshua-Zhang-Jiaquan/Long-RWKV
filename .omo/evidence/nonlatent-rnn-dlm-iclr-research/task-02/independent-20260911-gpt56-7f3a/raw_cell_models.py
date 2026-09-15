from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceError(RuntimeError):
    pass


class MetricField(StrEnum):
    MAX_RUN_FRAC = "max_run_frac"
    MASKED_TOKEN_ACCURACY = "em"
    TAU_A_COMMIT_ORDER = "tau"
    DISTINCT_FRAC = "distinct_frac"
    RESIDUE = "residue"


class RawMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore", allow_inf_nan=False)
    max_run_frac: float
    em: float
    tau: float
    distinct_frac: float
    residue: float


class RawRecord(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    document_id: str
    seed: int
    arm: str
    failure: str | None
    metrics: RawMetrics
    target_tokens: tuple[int, ...] | None = None
    prediction_tokens: tuple[int, ...] | None = None


class RawGrid(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    mask_ratios: tuple[float, ...]
    steps: tuple[int, ...]


class RawShard(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    num_shards: int
    shard_index: int
    n_records: int
    grid: RawGrid
    records: tuple[RawRecord, ...]


class ReportedCorrection(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="allow")
    loop_panel: str
    control_panel: str
    loop_sha256: str
    control_sha256: str
    pair_count: int
    max_run_delta: float
    max_run_interpretation: str


class CorrectionReport(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="allow")
    corrections: tuple[ReportedCorrection, ...]
    legacy_ci: str


@dataclass(frozen=True, slots=True)
class Panel:
    name: str
    digest: str
    file_count: int
    records: tuple[RawRecord, ...]
    grid: RawGrid


class PanelControl(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    sha256: str
    file_count: int
    record_count: int
    arm_count: int
    records_per_arm: int
    failure_count: int
    complete_shard_indices: bool
    complete_grid_arms: bool


class MetricControl(BaseModel):
    model_config = ConfigDict(frozen=True)
    corrected_name: str
    raw_field: str
    pair_count: int
    candidate_mean: float
    control_mean: float
    mean_delta: float
    historical_normal_ci95_half_width: float
    ci_status: Literal["descriptive_only_not_multiseed_confirmation"]


class CellControl(BaseModel):
    model_config = ConfigDict(frozen=True)
    arm: str
    mask_ratio: float
    steps: int
    candidate_repetitions: int
    control_repetitions: int
    metrics: tuple[MetricControl, ...]


class ComparisonControl(BaseModel):
    model_config = ConfigDict(frozen=True)
    candidate_panel: str
    control_panel: str
    cell_count: int
    raw_pair_count: int
    reported_pair_count: int
    independently_computed_pooled_max_run_delta: float
    reported_pooled_max_run_delta: float
    pooled_delta_matches_raw: bool
    panel_hashes_match_report: bool
    max_run_positive_cell_count: int
    max_run_negative_cell_count: int
    heterogeneous_cell_directions_hidden_by_pooling: bool
    cells: tuple[CellControl, ...]


class ReportCoverage(BaseModel):
    model_config = ConfigDict(frozen=True)
    saved_report_has_per_cell_rows: bool
    saved_report_has_masked_token_accuracy_values: bool
    saved_report_has_tau_values: bool
    saved_report_has_diagnostic_values: bool
    historical_ci_clearly_labeled_descriptive: bool
    decoded_sequences_available_in_raw_records: bool
    whole_sequence_exact_match_reported: bool


class ProbeOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: int = Field(default=1, ge=1)
    panels: tuple[PanelControl, ...]
    comparisons: tuple[ComparisonControl, ...]
    coverage: ReportCoverage


type RecordKey = tuple[str, int, str]
type CellKey = tuple[str, int]
type Values = dict[CellKey, float]


@dataclass(frozen=True, slots=True)
class ComparisonInputs:
    candidate: Panel
    control: Panel
