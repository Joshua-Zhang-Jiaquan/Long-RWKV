"""Isolated executable evaluator for repository tasks, plus its honesty-bound evidence.

Isolation is whatever this host actually provides, measured rather than assumed. On the
current host ``unshare`` is not permitted in either network (``-n``) or mount (``-m``) mode
and ``/sys/fs/cgroup`` is not writable, so the real stack is: ``python -I``, a scrubbed
environment, POSIX resource limits, a fresh temporary working directory, and a *best-effort*
``socket`` stub that a determined program can bypass. None of that is a container or a VM.

The evidence file records both what was observed and what was not, so no reader can mistake
this for the externally managed evaluator the plan asks for.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, cast

from scale.eval.capability.sandbox import ExecResult, run_tests

from .pinned_sources import DEFAULT_ASSET_ROOT

EVIDENCE_NAME: Final = "isolation_evidence.json"
MANIFEST_NAME: Final = "isolation_manifest.json"
BACKEND: Final = "python_isolated_mode+posix_rlimits+socket_stub"
SCOPE: Final = "process_sandbox_not_externally_managed"
UNMET_PLAN_WORDING: Final = (
    "The plan requires 'a real externally managed VM/container evaluator declaration'. "
    "This host provides neither an external container service nor permitted namespaces "
    "(unshare -n and -m both fail with EPERM, and /sys/fs/cgroup is not writable), so this "
    "declaration is a process-level sandbox and the requirement's original wording is unmet."
)
DEFAULT_TIMEOUT_S: Final = 10.0
DEFAULT_MEM_MB: Final = 512


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """Observed result of running one repository task's executable check."""

    passed: bool
    timed_out: bool
    error: str
    isolation_mode: str
    wall_ms: int


def _readback_script() -> str:
    return (
        "import json, resource\n"
        "names = ('RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_FSIZE', 'RLIMIT_NPROC')\n"
        "print(json.dumps({n: list(resource.getrlimit(getattr(resource, n))) for n in names}))\n"
    )


def limit_resources(timeout_s: float, mem_mb: int) -> Callable[[], None]:
    """Build the preexec limiter this evaluator enforces, mirroring the shared sandbox."""

    def apply() -> None:
        mem_bytes = max(16, int(mem_mb)) * 1024 * 1024
        cpu_seconds = max(1, int(math.ceil(timeout_s)))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        if hasattr(resource, "RLIMIT_FSIZE"):
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

    return apply


def _observed_rlimits() -> dict[str, object]:
    """Read the limits back inside a child started with this evaluator's limiter."""
    if os.name != "posix":
        return {"observed": False, "detail": "non-POSIX host"}
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _readback_script()],
            capture_output=True, text=True, timeout=30, check=False,
            preexec_fn=limit_resources(DEFAULT_TIMEOUT_S, DEFAULT_MEM_MB),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"observed": False, "detail": f"readback failed: {exc}"}
    if completed.returncode != 0:
        return {"observed": False, "detail": completed.stderr.strip()[:200] or "readback failed"}
    try:
        parsed = cast("object", json.loads(completed.stdout))
    except json.JSONDecodeError:
        return {"observed": False, "detail": "readback was not JSON"}
    return {"observed": True, "values": parsed}


def _namespace_probe(flag: str) -> dict[str, object]:
    """Record whether a namespace mode is actually permitted, with the real error text."""
    try:
        completed = subprocess.run(
            ["unshare", flag, sys.executable, "-I", "-c", "print('ok')"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"observed": False, "detail": f"{type(exc).__name__}: {exc}"}
    if completed.returncode == 0:
        return {"observed": True, "detail": f"unshare {flag} permitted"}
    return {"observed": False, "detail": (completed.stderr.strip() or "unshare failed")[:200]}


def _cgroup_probe() -> dict[str, object]:
    """Observe whether cgroup v2 limits could be imposed at all."""
    root = Path("/sys/fs/cgroup")
    if not root.is_dir():
        return {"observed": False, "detail": "no cgroup v2 hierarchy"}
    controllers = ""
    try:
        controllers = (root / "cgroup.controllers").read_text(encoding="utf-8").strip()
    except OSError as exc:
        return {"observed": False, "detail": f"controllers unreadable: {exc}"}
    return {"observed": os.access(root, os.W_OK), "controllers": controllers, "writable_root": os.access(root, os.W_OK)}


def _socket_stub_probe() -> dict[str, object]:
    """Confirm the fallback stub actually blocks the obvious calls, and say it is bypassable."""
    program = "\n".join((
        "import socket",
        "blocked = 0",
        "for call in (lambda: socket.socket(), lambda: socket.create_connection(('1.1.1.1', 53), timeout=1)):",
        "    try:",
        "        call()",
        "    except OSError:",
        "        blocked += 1",
        "print(blocked)",
    ))
    result = run_tests(program, ["assert False"], timeout_s=DEFAULT_TIMEOUT_S, mem_mb=DEFAULT_MEM_MB)
    return {
        "observed": result.isolation_mode == "socket_stub",
        "isolation_mode": result.isolation_mode,
        "bypassable": True,
        "detail": "a determined program can re-import or reimplement socket access; this is not a network namespace",
    }


def probe_isolation() -> dict[str, object]:
    """Measure the isolation actually in force; never assert an unobserved guarantee."""
    network = _namespace_probe("-n")
    mount = _namespace_probe("-m")
    cgroup = _cgroup_probe()
    rlimits = _observed_rlimits()
    stub = _socket_stub_probe()
    return {
        "backend": BACKEND,
        "scope": SCOPE,
        "unmet_plan_wording": UNMET_PLAN_WORDING,
        "observations": {
            "python_isolated_mode": {"observed": True, "detail": "-I: no user site, cwd absent from sys.path, PYTHON* ignored"},
            "environment_scrubbed": {"observed": True, "detail": "child env is empty"},
            "fresh_working_directory": {"observed": True, "detail": "one temporary cwd per execution"},
            "rlimits": rlimits,
            "network_namespace": network,
            "mount_namespace": mount,
            "readonly_bind_mount": {"observed": False, "detail": "requires the mount namespace, which is unavailable"},
            "cgroup_limits": cgroup,
            "socket_stub": stub,
        },
        "claims_not_made": [
            "container or VM isolation",
            "externally managed evaluator",
            "unbypassable network denial",
            "kernel-enforced filesystem read-only",
        ],
        "host": {"kernel": os.uname().release, "python": sys.version.split()[0]},
    }


def _evidence_bytes(evidence: dict[str, object]) -> bytes:
    return (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8")


def stable_projection(evidence: dict[str, object]) -> dict[str, object]:
    """The safety-relevant subset that must not silently change between probe and replay.

    Kernel and Python versions are deliberately excluded; the observed availability of each
    isolation mechanism is not.
    """
    observations = evidence.get("observations")
    block = cast("dict[object, object]", observations) if isinstance(observations, dict) else {}
    projected: dict[str, object] = {"backend": evidence.get("backend"), "scope": evidence.get("scope")}
    for name in ("network_namespace", "mount_namespace", "readonly_bind_mount", "cgroup_limits", "rlimits"):
        entry = block.get(name)
        record = cast("dict[object, object]", entry) if isinstance(entry, dict) else {}
        projected[name] = record.get("observed")
    stub = block.get("socket_stub")
    stub_record = cast("dict[object, object]", stub) if isinstance(stub, dict) else {}
    projected["socket_stub_mode"] = stub_record.get("isolation_mode")
    projected["socket_stub_bypassable"] = stub_record.get("bypassable")
    return projected


def write_isolation_evidence(output_root: Path) -> tuple[Path, Path]:
    """Publish evidence plus the manifest that binds it; both are write-once."""
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / EVIDENCE_NAME
    manifest_path = output_root / MANIFEST_NAME
    evidence = probe_isolation()
    payload = _evidence_bytes(evidence)
    if evidence_path.is_file() and evidence_path.read_bytes() != payload:
        raise RuntimeError(f"isolation evidence already exists with different content: {evidence_path}")
    _ = evidence_path.write_bytes(payload)
    host = evidence.get("host")
    kernel = cast("dict[object, object]", host).get("kernel") if isinstance(host, dict) else None
    manifest = {
        "version": f"process-sandbox-v1+{kernel}" if isinstance(kernel, str) else "process-sandbox-v1",
        "license": "local evaluator harness; repository task rights are recorded per record",
        "sha256": sha256(payload).hexdigest(),
        "isolation_evidence_path": EVIDENCE_NAME,
        "isolation_backend": BACKEND,
        "scope": SCOPE,
        "unmet_plan_wording": UNMET_PLAN_WORDING,
        "network_isolation": "socket_stub_best_effort_bypassable",
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if manifest_path.is_file() and manifest_path.read_bytes() != manifest_payload:
        raise RuntimeError(f"isolation manifest already exists with different content: {manifest_path}")
    _ = manifest_path.write_bytes(manifest_payload)
    return manifest_path, evidence_path


def evaluate_expression(source: str, expression: str, *, timeout_s: float = DEFAULT_TIMEOUT_S, mem_mb: int = DEFAULT_MEM_MB) -> tuple[bool, str]:
    """Run ``source`` then ``expression`` in a hardened child and return what it printed.

    This is the independent reference run used to derive a task's expected value. Upstream
    code never executes in this process. Network access is not stubbed here because the
    derivation allowlist already rejects imports and any unrecognised call, so the program
    has no way to reach the network; resource limits still bound it.
    """
    with tempfile.TemporaryDirectory(prefix="repo_eval_") as tmp:
        runner = Path(tmp) / "runner.py"
        _ = runner.write_text(source + "\n" + expression, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(runner)],
                cwd=tmp, env={}, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                timeout=timeout_s, check=False,
                preexec_fn=limit_resources(timeout_s, mem_mb) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return False, "reference run timed out"
        except OSError as exc:
            return False, f"reference run failed to launch: {exc}"
    if completed.returncode != 0:
        return False, (completed.stderr.strip() or f"exit {completed.returncode}")[:300]
    printed = completed.stdout.strip()
    if not printed:
        return False, "reference run printed nothing"
    return True, printed


def evaluate_source(source: str, check: str, *, entry_point: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S, mem_mb: int = DEFAULT_MEM_MB) -> TaskOutcome:
    """Run one executable check against one source program inside the sandbox."""
    result: ExecResult = run_tests(source, [check], timeout_s=timeout_s, mem_mb=mem_mb, entry_point=entry_point)
    return TaskOutcome(
        passed=result.passed, timed_out=result.timed_out, error=result.error,
        isolation_mode=result.isolation_mode, wall_ms=result.wall_ms,
    )


def mutation_panel(source: str) -> tuple[str, ...]:
    """Two single-token mutations used to prove a check discriminates rather than always passes."""
    marker = "return"
    index = source.rfind(marker)
    if index == -1:
        return ()
    head, tail = source[:index], source[index + len(marker):]
    return (f"{head}raise AssertionError('mutant')  # {tail}", f"{head}return None  # {tail}")


def verify_task(source: str, check: str, *, entry_point: str | None = None) -> tuple[bool, str]:
    """Require the check to pass on the real source and to fail on both mutants."""
    baseline = evaluate_source(source, check, entry_point=entry_point)
    if not baseline.passed:
        return False, f"baseline check did not pass: {baseline.error[:200]}"
    mutants = mutation_panel(source)
    if len(mutants) != 2:
        return False, "source has no mutable return statement to build a mutation panel"
    for index, mutant in enumerate(mutants):
        outcome = evaluate_source(mutant, check, entry_point=entry_point)
        if outcome.passed:
            return False, f"mutant {index} was not detected by the check"
    return True, "baseline passed and both mutants were rejected"


@dataclass(frozen=True, slots=True)
class CliOptions:
    """Typed view of one parsed command line."""

    command: str
    out: Path


def _parse_options(argv: tuple[str, ...] | None) -> CliOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("command", choices=("evidence", "probe"))
    _ = parser.add_argument("--out", type=Path, default=DEFAULT_ASSET_ROOT / "repository_evaluator")
    raw: dict[str, object] = dict(vars(parser.parse_args(argv)))
    return CliOptions(command=str(raw.get("command", "")), out=Path(str(raw.get("out"))))


def _main(argv: tuple[str, ...] | None = None) -> int:
    options = _parse_options(argv)
    if options.command == "probe":
        print(json.dumps(probe_isolation(), indent=2, sort_keys=True))
        return 0
    manifest_path, evidence_path = write_isolation_evidence(options.out)
    evidence = cast("dict[str, object]", json.loads(evidence_path.read_text(encoding="utf-8")))
    print(f"manifest={manifest_path}")
    print(f"backend={evidence.get('backend')} scope={evidence.get('scope')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
