from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).parents[3]
MODULE = "scale.experiments.nonlatent_iclr.qualification.cli"


def _manifest_arguments(tmp_path: Path) -> list[str]:
    return [
        "--manifest", str(tmp_path / "runtime-manifest.json"),
        "--manifest-sha256", "0" * 64,
        "--run-id", "qualification-20260912-01-deadbeef",
        "--controller-receipt", str(tmp_path / "controller-receipt.json"),
    ]


def test_cli_returns_structured_nonzero_overbudget_rejection(tmp_path: Path) -> None:
    # Given: a CPU-only probe request over the local launcher deadline.
    result = subprocess.run(
        [
            sys.executable, "-m", MODULE, "validate",
            "--output-dir", str(tmp_path / "unique"),
            "--nonce", "0" * 32,
            "--timeout-seconds", "1201",
            *_manifest_arguments(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # When: the isolated payload CLI parses the invalid request.
    payload = json.loads(result.stdout)

    # Then: it reports a structured failure and never imports or probes CUDA.
    assert result.returncode == 2
    assert payload == {"detail": "timeout_seconds_exceeds_1200", "status": "INVALID_ARGUMENT"}


def test_probe_module_entrypoint_handles_help_and_bad_input_without_cuda() -> None:
    # Given: the exact module torchrun will invoke, outside a CUDA job.
    help_result = subprocess.run([sys.executable, "-m", "scale.experiments.nonlatent_iclr.qualification.probe", "--help"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    bad_result = subprocess.run([sys.executable, "-m", "scale.experiments.nonlatent_iclr.qualification.probe", "--nonce", "bad"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)

    # When / Then: both boundary paths execute main without creating rank output or importing CUDA.
    assert help_result.returncode == 0
    assert "usage:" in help_result.stdout
    assert bad_result.returncode == 2
    assert json.loads(bad_result.stdout)["status"] == "INVALID_ARGUMENT"


def test_cli_validate_publishes_terminal_failure_for_manifest_digest_mismatch(
    tmp_path: Path,
) -> None:
    # Given: a manifest path whose bytes do not match the externally pinned digest.
    manifest_path = tmp_path / "runtime-manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    # When: the real subprocess validation boundary runs without CUDA.
    result = subprocess.run(
        [
            sys.executable, "-m", MODULE, "validate",
            "--output-dir", str(tmp_path / "unique"),
            "--nonce", "0" * 32,
            *_manifest_arguments(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    # Then: source identity is a terminal ordinary failure, not readiness.
    assert result.returncode == 1
    assert payload["status"] == "FAILED"
    assert payload["detail"] == "source_identity:manifest_digest_mismatch"


def test_cli_aggregate_publishes_terminal_failure_when_ranks_are_missing(
    tmp_path: Path,
) -> None:
    # Given: a fresh output directory with no rank records.
    output_dir = tmp_path / "unique"
    output_dir.mkdir()

    # When: the real owner aggregation subprocess consumes it.
    result = subprocess.run(
        [
            sys.executable, "-m", MODULE, "aggregate",
            "--output-dir", str(output_dir),
            "--nonce", "0" * 32,
            *_manifest_arguments(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout)

    # Then: it emits parseable FAILED evidence and exits nonzero.
    assert result.returncode == 1
    assert payload["status"] == "FAILED"
    assert payload["detail"] == "missing_rank_results:0,1,2,3,4,5,6,7"
