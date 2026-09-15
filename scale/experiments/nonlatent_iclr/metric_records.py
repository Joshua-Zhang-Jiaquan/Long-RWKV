from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .metrics import MetricInputError, MetricName, Pair, PairedSummary, paired_delta


RAW_FIELDS = {
    MetricName.MAX_RUN_FRAC: "max_run_frac",
    MetricName.MASKED_TOKEN_ACCURACY: "em",
    MetricName.TAU_A_COMMIT_ORDER: "tau",
    MetricName.DISTINCT_FRAC: "distinct_frac",
    MetricName.RESIDUE: "residue",
}


@dataclass(frozen=True, slots=True)
class RecordPanel:
    name: str
    sha256: str
    file_count: int
    pairs: tuple[Pair, ...]


def load_sampler_panel(root: Path, name: str, metric: MetricName) -> RecordPanel:
    panel = root / name
    files = tuple(sorted(panel.glob("sampler.shard*of*.json")))
    if not files:
        raise MetricInputError(f"missing sampler panel: {name}")
    pairs: list[Pair] = []
    digest = hashlib.sha256()
    shard_indices: set[int] = set()
    expected_arms: set[str] | None = None
    for path in files:
        content = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(content)
        raw = json.loads(content)
        if not isinstance(raw, dict) or raw.get("schema") != "qz_capability_sampler_shard_v1":
            raise MetricInputError(f"malformed sampler shard: {path}")
        index = raw.get("shard_index")
        shard_count = raw.get("num_shards")
        if type(index) is not int or type(shard_count) is not int or shard_count != 8 or index < 0 or index >= shard_count:
            raise MetricInputError(f"invalid shard declaration: {path}")
        shard_indices.add(index)
        grid = raw.get("grid")
        if not isinstance(grid, dict):
            raise MetricInputError(f"missing grid: {path}")
        ratios = grid.get("mask_ratios")
        steps = grid.get("steps")
        if not isinstance(ratios, list) or not isinstance(steps, list) or any(type(item) not in {int, float} for item in ratios + steps):
            raise MetricInputError(f"invalid grid: {path}")
        arms = {f"r{round(float(ratio) * 100):d}_s{int(step)}" for ratio in ratios for step in steps}
        expected_arms = arms if expected_arms is None else expected_arms
        if expected_arms != arms:
            raise MetricInputError(f"inconsistent grid: {path}")
        records = raw.get("records")
        declared_records = raw.get("n_records")
        if not isinstance(records, list) or type(declared_records) is not int or declared_records != len(records):
            raise MetricInputError(f"missing records: {path}")
        for record in records:
            if not isinstance(record, dict):
                raise MetricInputError(f"malformed record: {path}")
            if record.get("failure") is not None:
                raise MetricInputError(f"failed record retained in panel: {path}")
            metrics = record.get("metrics")
            value = metrics.get(RAW_FIELDS[metric]) if isinstance(metrics, dict) else None
            document_id = record.get("document_id")
            seed = record.get("seed")
            arm = record.get("arm")
            if not isinstance(document_id, str) or type(seed) is not int or not isinstance(arm, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
                raise MetricInputError(f"invalid paired record: {path}")
            pairs.append(Pair(document_id, seed, arm, float(value)))
    if shard_indices != set(range(8)) or len(files) != 8:
        raise MetricInputError(f"incomplete shard set: {name}")
    actual_arms = {pair.arm for pair in pairs}
    if expected_arms is None or actual_arms != expected_arms or any(sum(pair.arm == arm for pair in pairs) != 256 for arm in expected_arms):
        raise MetricInputError(f"incomplete arm coverage: {name}")
    return RecordPanel(name, digest.hexdigest(), len(files), tuple(pairs))


def summarize_sampler_pair(root: Path, loop_name: str, control_name: str, metric: MetricName) -> tuple[RecordPanel, RecordPanel, PairedSummary]:
    loop = load_sampler_panel(root, loop_name, metric)
    control = load_sampler_panel(root, control_name, metric)
    return loop, control, paired_delta(metric, loop.pairs, control.pairs)


def summarize_sampler_cells(root: Path, loop_name: str, control_name: str, metric: MetricName) -> tuple[RecordPanel, RecordPanel, tuple[PairedSummary, ...]]:
    loop = load_sampler_panel(root, loop_name, metric)
    control = load_sampler_panel(root, control_name, metric)
    arms = tuple(sorted({pair.arm for pair in loop.pairs}))
    summaries = tuple(paired_delta(metric, tuple(pair for pair in loop.pairs if pair.arm == arm), tuple(pair for pair in control.pairs if pair.arm == arm)) for arm in arms)
    return loop, control, summaries
