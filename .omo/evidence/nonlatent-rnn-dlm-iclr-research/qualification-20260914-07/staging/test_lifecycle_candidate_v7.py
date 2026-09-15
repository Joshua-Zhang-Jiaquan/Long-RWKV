from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict


V7_ROOT = Path(
    "/inspire/hdd/global_user/zhangjiaquan-253108540222/"
    "nonlatent_iclr_qualification/payloads/"
    "qualification-20260914-07-global-ffaa464d-lifecycle-v7"
)
WORKSPACE = Path(
    "/inspire/hdd/project/multimodal-diffusion-language-model/"
    "zhangjiaquan-253108540222/DiffRWKV-RELAY"
)
PYTHON = Path("/usr/bin/python")
APPROVED_TESTS = (
    "test_lifecycle_runtime.py",
    "test_lifecycle_sidecar.py",
    "test_qualification_lifecycle.py",
    "test_lifecycle_evidence.py",
    "test_qualification_masked_forward.py",
    "test_full_canvas.py",
    "test_full_canvas_ownership.py",
    "test_qualification_probe.py",
    "test_qualification_runtime_contracts.py",
    "test_qualification_manifest.py",
    "test_qualification_parameter_evidence.py",
    "test_qualification_controls.py",
    "test_qualification_controller_receipt.py",
    "test_qualification_cli.py",
)
LIFECYCLE_MODULES = (
    "lifecycle.py",
    "lifecycle_checks.py",
    "lifecycle_evidence.py",
    "lifecycle_binding.py",
    "lifecycle_settings.py",
    "lifecycle_sidecar.py",
    "lifecycle_runtime.py",
)

APPROVED_SUITE_SCRIPT = r"""
from __future__ import annotations
import json
import os
from pathlib import Path
import scale

release = Path(os.environ["RELEASE_ROOT"]).resolve()
workspace = Path(os.environ["WORKSPACE_ROOT"]).resolve()
scale.__path__.append(str(workspace / "scale"))

from scale.experiments.nonlatent_iclr.qualification import runtime_checks
import pytest

origin = Path(runtime_checks.__file__).resolve()
if not origin.is_relative_to(release):
    raise AssertionError(f"runtime_checks_origin_invalid:{origin}")
test_root = workspace / "scale/tests/nonlatent_iclr"
tests = json.loads(os.environ["APPROVED_TESTS_JSON"])
arguments = [str(test_root / name) for name in tests]
arguments.extend(("-q", "--tb=short", "--import-mode=importlib", "-p", "no:cacheprovider",
                  "--junitxml", os.environ["JUNIT_PATH"]))
exit_code = pytest.main(arguments)
print("CANDIDATE_RESULT=" + json.dumps({"exit_code": int(exit_code), "origin": str(origin)}, sort_keys=True))
raise SystemExit(int(exit_code))
"""

OUTPUT_CONTRACT_SCRIPT = r"""
from __future__ import annotations
import json
import os
from pathlib import Path
import scale

release = Path(os.environ["RELEASE_ROOT"]).resolve()
workspace = Path(os.environ["WORKSPACE_ROOT"]).resolve()
scale.__path__.append(str(workspace / "scale"))

from scale.experiments.nonlatent_iclr.qualification.contracts import RankResult, SuccessEvidence
from scale.experiments.nonlatent_iclr.qualification.controller_receipt import ControllerReceipt
from scale.experiments.nonlatent_iclr.qualification.lifecycle_binding import LifecycleBinding
from scale.experiments.nonlatent_iclr.qualification.lifecycle_sidecar import LifecycleIssue, LifecycleSidecar
from scale.experiments.nonlatent_iclr.qualification.manifest import ManifestFile
from scale.tests.nonlatent_iclr.test_qualification_controls import _success_evidence
from scale.tests.nonlatent_iclr.test_qualification_controller_receipt import _receipt_payload

output = Path(os.environ["OUTPUT_ROOT"])
success = SuccessEvidence.model_validate(_success_evidence(0))
receipt = ControllerReceipt.model_validate(_receipt_payload()).to_binding_evidence()
checkpoint = ManifestFile(
    path=Path("/cpu-contract-only/model.pt"), role="checkpoint_model",
    sha256=success.checkpoint_sha256, size_bytes=success.checkpoint_size_bytes,
)
for rank in range(8):
    binding = LifecycleBinding(
        controller=receipt, run_id=receipt.run_id, nonce=receipt.nonce,
        source_manifest_sha256=receipt.source_manifest_sha256, checkpoint=checkpoint,
        checkpoint_step=4750, checkpoint_state_tensors=1955, geometry=success.geometry,
        rank=rank, local_rank=rank, world_size=8, device=f"cuda:{rank}",
    )
    sidecar = LifecycleSidecar(
        binding=binding, before=None, after=None, observations=None,
        issue=LifecycleIssue(code="execution_error", exception_type="CPUContractFixture", detail="not_runtime_evidence"),
    )
    if sidecar.passed or sidecar.runtime_promoted:
        raise AssertionError("partial_sidecar_promoted")
    sidecar.write_once(output)
    RankResult.failed(rank=rank, nonce=receipt.nonce, detail="cpu_contract_fixture").write_once(output)
names = sorted(path.name for path in output.iterdir())
print("OUTPUT_RESULT=" + json.dumps({"names": names}, sort_keys=True))
"""


class CandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    exit_code: int
    origin: Path


class OutputResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    names: tuple[str, ...]


def run_candidate(tmp_path: Path, script: str) -> subprocess.CompletedProcess[str]:
    environment = {
        "APPROVED_TESTS_JSON": json.dumps(APPROVED_TESTS),
        "JUNIT_PATH": str(tmp_path / "candidate-suite.xml"),
        "LC_ALL": "C",
        "OUTPUT_ROOT": str(tmp_path / "runtime-output"),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(V7_ROOT),
        "PYTHONPYCACHEPREFIX": str(tmp_path / "candidate-pycache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "RELEASE_ROOT": str(V7_ROOT),
        "WORKSPACE_ROOT": str(WORKSPACE),
    }
    return subprocess.run(
        (str(PYTHON), "-P", "-c", script),
        cwd=V7_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def marked_result(stdout: str, marker: str) -> str:
    matches = tuple(line.removeprefix(marker) for line in stdout.splitlines() if line.startswith(marker))
    assert len(matches) == 1
    return matches[0]


def test_candidate_release_runs_exact_approved_191_test_scope(tmp_path: Path) -> None:
    # Given: all lifecycle modules exist only inside the fresh successor.
    assert all(
        (V7_ROOT / "scale/experiments/nonlatent_iclr/qualification" / name).is_file()
        for name in LIFECYCLE_MODULES
    )

    # When: the approved CPU scope imports and exercises the candidate caller.
    result = run_candidate(tmp_path, APPROVED_SUITE_SCRIPT)

    # Then: the release origin is exact and all 191 approved cases pass.
    assert result.returncode == 0, result.stdout + result.stderr
    observation = CandidateResult.model_validate_json(
        marked_result(result.stdout, "CANDIDATE_RESULT=")
    )
    assert observation.exit_code == 0
    assert observation.origin == (
        V7_ROOT / "scale/experiments/nonlatent_iclr/qualification/runtime_checks.py"
    )
    report = ElementTree.parse(tmp_path / "candidate-suite.xml").getroot()
    assert int(report.attrib["tests"]) == 191
    assert int(report.attrib["failures"]) == 0
    assert int(report.attrib["errors"]) == 0


def test_candidate_sidecars_coexist_with_all_core_rank_records(tmp_path: Path) -> None:
    # Given: CPU-only typed partial sidecars for every rank and unchanged core records.
    result = run_candidate(tmp_path, OUTPUT_CONTRACT_SCRIPT)

    # When/Then: release writers emit both exact eight-file sets without promotion.
    assert result.returncode == 0, result.stdout + result.stderr
    observation = OutputResult.model_validate_json(
        marked_result(result.stdout, "OUTPUT_RESULT=")
    )
    expected = tuple(sorted(
        tuple(f"lifecycle-{rank}.json" for rank in range(8))
        + tuple(f"rank-{rank}.json" for rank in range(8))
    ))
    assert observation.names == expected
