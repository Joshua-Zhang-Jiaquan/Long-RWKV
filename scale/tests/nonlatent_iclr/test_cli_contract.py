from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scale.experiments.nonlatent_iclr import service


REPO_ROOT = Path(__file__).parents[3]
MODULE = "scale.experiments.nonlatent_iclr.cli"


def _roots(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    snapshot = repo / "DAN/v7_arch_round"
    for _, relative, _ in service.SOURCES:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture:{relative}\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    return repo, tmp_path / "evidence", external, tmp_path / "live", tmp_path / "staged"


def _common(roots: tuple[Path, Path, Path, Path, Path]) -> list[str]:
    repo, evidence, external, live, staged = roots
    return [
        "--task", "1",
        "--repo-root", str(repo),
        "--evidence-root", str(evidence),
        "--external-root", str(external),
        "--live-root", str(live),
        "--staged-root", str(staged),
    ]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", MODULE, *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _json(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.stdout.startswith("{\n  ")
    return json.loads(result.stdout)


def test_all_four_commands_execute_real_task_one_paths(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    prepare = _run("prepare", *_common(roots))
    assert prepare.returncode == 0
    assert _json(prepare)["status"] == "AUDIT_COMPLETE"

    analyze = _run("analyze", *_common(roots))
    assert analyze.returncode == 0
    analyze_data = _json(analyze)
    assert analyze_data["status"] == "AUDIT_COMPLETE"
    assert "artifacts=" in str(analyze_data["detail"])
    assert "blocked_claims=" in str(analyze_data["detail"])

    verify = _run("verify", "--case", "happy", *_common(roots))
    assert verify.returncode == 0
    assert _json(verify)["status"] == "AUDIT_COMPLETE"

    failure = _run("verify", "--case", "failure", *_common(roots))
    assert failure.returncode == 0
    assert _json(failure)["status"] == "EXPECTED_FAILURE_CONFIRMED"

    output = roots[0] / "DAN/nonlatent_iclr/rendered_audit.md"
    render = _run("render", "--out", str(output), *_common(roots))
    assert render.returncode == 0
    assert _json(render)["status"] == "AUDIT_COMPLETE"
    body = output.read_text(encoding="utf-8")
    for heading in ("## Source roots", "## Artifacts", "## Historical claims", "## Snapshot/live/staged views"):
        assert heading in body


def test_invalid_cli_scalar_has_structured_nonzero_exit(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    arguments = _common(roots)
    arguments[1] = "not-an-integer"
    result = _run("prepare", *arguments)
    assert result.returncode == 2
    assert _json(result)["status"] == "INVALID_ARGUMENT"
    assert "Traceback" not in result.stderr


def test_render_refuses_destination_outside_output_root(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    assert _run("prepare", *_common(roots)).returncode == 0
    outside = tmp_path / "outside.md"
    result = _run("render", "--out", str(outside), *_common(roots))
    assert result.returncode == 2
    assert _json(result)["status"] == "UNSAFE_DESTINATION"
    assert not outside.exists()


def test_render_refuses_existing_destination_without_overwrite(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    assert _run("prepare", *_common(roots)).returncode == 0
    output = roots[0] / "DAN/nonlatent_iclr/existing.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("original\n", encoding="utf-8")
    result = _run("render", "--out", str(output), *_common(roots))
    assert result.returncode == 2
    assert _json(result)["status"] == "PUBLICATION_EXISTS"
    assert output.read_text(encoding="utf-8") == "original\n"
