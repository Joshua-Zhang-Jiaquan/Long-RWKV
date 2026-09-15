"""Full token-length matrix over the 720 declared cells, five seeds and 200 instances each.

The artifact stores aggregates and digests only -- never prompt text -- so it stays small
while remaining exactly replayable: ``verify_matrix_artifact`` regenerates selected units
from the deterministic generators and compares. A full replay is available but costs about an
hour single-core, so the default check samples deterministically instead.

Shards are contiguous slices of the 720x5 unit grid, each written write-once, so a long run
can be resumed and parallelised without any shared mutable state.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, cast

from ..task_registry import build_registry
from .exact_tasks import InfeasibleTaskError, generate_exact_task
from .length_qualification import QualificationInputs, accepted_tokenizer
from .models import ExactTaskRequest
from .tokenizer_qualification import SafeRWKVTokenizer

SCHEMA_VERSION: Final = 1
INSTANCES_PER_CELL: Final = 200
DATA_SEEDS: Final = (101, 102, 103, 104, 105)
DEFAULT_SAMPLE: Final = 24
MATRIX_NAME: Final = "length_matrix.json"
_worker_tokenizer: SafeRWKVTokenizer | None = None


@dataclass(frozen=True, slots=True)
class CellSeedDigest:
    """Aggregate token lengths for one cell under one seed; carries no prompt text."""

    family: str
    length: int
    position_fraction: int
    load: int
    distractor: str
    data_seed: int
    feasible_instances: int
    infeasible_instances: int
    token_min: int
    token_max: int
    token_total: int
    token_lengths_sha256: str


@dataclass(frozen=True, slots=True)
class MatrixArtifact:
    """The published matrix plus the bindings that let it be replayed."""

    schema_version: int
    unit_count: int
    instances_per_cell: int
    data_seeds: tuple[int, ...]
    digests: tuple[CellSeedDigest, ...]
    source_hashes: dict[str, str]
    matrix_token_lengths_qualified: bool = True
    scope: str = "all declared cells, all five data seeds, all 200 instances per cell"


def unit_coordinates(unit: int) -> tuple[int, int]:
    """Map a flat unit index to (cell index, seed index)."""
    seed_count = len(DATA_SEEDS)
    return unit // seed_count, unit % seed_count


def unit_count() -> int:
    return len(build_registry().cells) * len(DATA_SEEDS)


def measure_unit(unit: int, tokenizer: SafeRWKVTokenizer) -> CellSeedDigest:
    """Measure every instance of one (cell, seed) unit, streaming and digest-only."""
    cell_index, seed_index = unit_coordinates(unit)
    cell = build_registry().cells[cell_index]
    data_seed = DATA_SEEDS[seed_index]
    digest = sha256()
    feasible = infeasible = 0
    minimum = 0
    maximum = 0
    total = 0
    for instance in range(INSTANCES_PER_CELL):
        request = ExactTaskRequest(cell.family, data_seed, cell.length, cell.position_fraction, cell.load, cell.distractor, instance)
        try:
            task = generate_exact_task(request)
        except InfeasibleTaskError:
            infeasible += 1
            continue
        token_length = len(tokenizer.encode_text(task.condition.public_prompt))
        digest.update(token_length.to_bytes(4, "little"))
        feasible += 1
        total += token_length
        minimum = token_length if minimum == 0 else min(minimum, token_length)
        maximum = max(maximum, token_length)
    return CellSeedDigest(
        family=cell.family, length=cell.length, position_fraction=cell.position_fraction,
        load=cell.load, distractor=cell.distractor, data_seed=data_seed,
        feasible_instances=feasible, infeasible_instances=infeasible,
        token_min=minimum, token_max=maximum, token_total=total,
        token_lengths_sha256=digest.hexdigest(),
    )


def shard_bounds(shard_index: int, shard_count: int) -> tuple[int, int]:
    """Half-open unit range for one shard; every unit belongs to exactly one shard."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard coordinates")
    total = unit_count()
    start = total * shard_index // shard_count
    end = total * (shard_index + 1) // shard_count
    return start, end


def _worker_init(model_root: str, evidence_root: str, receipt: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = accepted_tokenizer(QualificationInputs(Path(model_root), Path(evidence_root), Path(receipt)))


def _worker_measure(unit: int) -> dict[str, object]:
    if _worker_tokenizer is None:
        raise RuntimeError("worker tokenizer was not initialised")
    return asdict(measure_unit(unit, _worker_tokenizer))


def run_shard(inputs: QualificationInputs, shard_index: int, shard_count: int, workers: int) -> list[dict[str, object]]:
    """Measure one shard, in parallel when workers > 1."""
    start, end = shard_bounds(shard_index, shard_count)
    units = list(range(start, end))
    if workers <= 1:
        _worker_init(str(inputs.model_root), str(inputs.tokenizer_evidence_root), str(inputs.tokenizer_receipt))
        return [_worker_measure(unit) for unit in units]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(inputs.model_root), str(inputs.tokenizer_evidence_root), str(inputs.tokenizer_receipt)),
    ) as pool:
        return list(pool.map(_worker_measure, units, chunksize=max(1, len(units) // (workers * 4) or 1)))


def source_hashes(inputs: QualificationInputs) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    sources = (
        # Only modules that determine the measurement are bound; CLI wrappers and the
        # asset-qualification path are not dependencies of the generated lengths.
        root / "length_matrix.py", root / "length_qualification.py", root / "exact_tasks.py",
        root / "models.py", root / "provenance.py",
        root / "tokenizer_qualification.py", root / "tokenizer_provenance.py",
        root.parent / "task_registry.py",
        inputs.model_root / "rwkv_vocab_v20230424.txt",
        inputs.tokenizer_evidence_root / "rwkv_tokenizer_results.json",
        inputs.tokenizer_evidence_root / "rwkv_tokenizer_manifest.json",
        inputs.tokenizer_receipt,
    )
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in sources if path.is_file()}


def artifact_bytes(digests: Sequence[CellSeedDigest | Mapping[str, object]], inputs: QualificationInputs) -> bytes:
    rows = [asdict(row) if isinstance(row, CellSeedDigest) else dict(row) for row in digests]
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "unit_count": unit_count(),
        "instances_per_cell": INSTANCES_PER_CELL,
        "data_seeds": list(DATA_SEEDS),
        "digests": rows,
        "source_hashes": source_hashes(inputs),
        "matrix_token_lengths_qualified": True,
        "scope": "all declared cells, all five data seeds, all 200 instances per cell",
    }
    return (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_matrix_artifact(path: Path, inputs: QualificationInputs, *, sample: int | None = DEFAULT_SAMPLE) -> bool:
    """Replay selected units and compare; ``sample=None`` replays every unit.

    A sampled check is the default because a full replay costs about an hour single-core.
    The sample is deterministic (evenly spaced units), so it is reproducible and cannot be
    tuned to the artifact.
    """
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict):
        return False
    artifact = cast("dict[str, object]", raw)
    if artifact.get("schema_version") != SCHEMA_VERSION:
        return False
    if artifact.get("unit_count") != unit_count():
        return False
    if artifact.get("source_hashes") != source_hashes(inputs):
        return False
    digests = artifact.get("digests")
    if not isinstance(digests, list):
        return False
    rows = cast("list[object]", digests)
    if len(rows) != unit_count():
        return False
    tokenizer = accepted_tokenizer(inputs)
    total = unit_count()
    units = list(range(total)) if sample is None else sorted({total * index // max(sample, 1) for index in range(max(sample, 1))})
    for unit in units:
        expected = asdict(measure_unit(unit, tokenizer))
        row = rows[unit]
        if not isinstance(row, dict):
            return False
        if cast("dict[object, object]", row) != cast("dict[object, object]", cast("object", expected)):
            return False
    return True


def write_shard(output: Path, shard_index: int, shard_count: int, rows: list[dict[str, object]]) -> Path:
    """Publish one shard write-once; an identical rewrite is a no-op, a different one is refused."""
    target = output.with_name(f"{output.stem}.shard-{shard_index:03d}-of-{shard_count:03d}{output.suffix}")
    payload = (json.dumps({"shard_index": shard_index, "shard_count": shard_count, "rows": rows}, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if target.is_file():
        if target.read_bytes() != payload:
            raise RuntimeError(f"shard already exists with different content: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_bytes(payload)
    return target


def assemble(shard_paths: list[Path], output: Path, inputs: QualificationInputs) -> Path:
    """Combine shards in order, requiring full coverage before publishing."""
    rows: list[object] = []
    for path in sorted(shard_paths):
        parsed = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        shard_rows = parsed.get("rows")
        if not isinstance(shard_rows, list):
            raise RuntimeError(f"shard has no rows: {path}")
        rows.extend(cast("list[object]", shard_rows))
    if len(rows) != unit_count():
        raise RuntimeError(f"shards cover {len(rows)} units, expected {unit_count()}")
    payload = artifact_bytes(cast("list[dict[str, object]]", rows), inputs)
    if output.is_file():
        if output.read_bytes() != payload:
            raise RuntimeError(f"matrix artifact already exists with different content: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_bytes(payload)
    return output


@dataclass(frozen=True, slots=True)
class CliOptions:
    """Typed view of one parsed command line."""

    model_root: Path
    evidence_root: Path
    receipt: Path
    output: Path
    shard_index: int
    shard_count: int
    workers: int
    assemble: bool


def _parse_options(argv: tuple[str, ...] | None) -> CliOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("model_root", type=Path)
    _ = parser.add_argument("evidence_root", type=Path)
    _ = parser.add_argument("receipt", type=Path)
    _ = parser.add_argument("output", type=Path)
    _ = parser.add_argument("--shard-index", type=int, default=0)
    _ = parser.add_argument("--shard-count", type=int, default=1)
    _ = parser.add_argument("--workers", type=int, default=1)
    _ = parser.add_argument("--assemble", action="store_true")
    raw: dict[str, object] = dict(vars(parser.parse_args(argv)))
    return CliOptions(
        model_root=Path(str(raw.get("model_root"))), evidence_root=Path(str(raw.get("evidence_root"))),
        receipt=Path(str(raw.get("receipt"))), output=Path(str(raw.get("output"))),
        shard_index=int(str(raw.get("shard_index"))), shard_count=int(str(raw.get("shard_count"))),
        workers=int(str(raw.get("workers"))), assemble=bool(raw.get("assemble")),
    )


def main(argv: tuple[str, ...] | None = None) -> int:
    """Offline CLI: measure one shard, or assemble published shards."""
    options = _parse_options(argv)
    inputs = QualificationInputs(options.model_root, options.evidence_root, options.receipt)
    if options.assemble:
        shards = sorted(options.output.parent.glob(f"{options.output.stem}.shard-*-of-{options.shard_count:03d}{options.output.suffix}"))
        if len(shards) != options.shard_count:
            print(f"found {len(shards)} of {options.shard_count} shards", file=sys.stderr)
            return 2
        _ = assemble(shards, options.output, inputs)
        print(f"assembled {len(shards)} shards into {options.output}")
        return 0
    rows = run_shard(inputs, options.shard_index, options.shard_count, options.workers)
    path = write_shard(options.output, options.shard_index, options.shard_count, rows)
    print(f"shard {options.shard_index}/{options.shard_count}: {len(rows)} units -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
