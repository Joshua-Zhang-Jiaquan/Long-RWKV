from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
REPORT = ROOT / "DAN/nonlatent_iclr/metric_corrections.json"
MARKDOWN = ROOT / "DAN/nonlatent_iclr/metric_corrections.md"
PRIOR_PROOF = ROOT / ".omo/evidence/nonlatent-rnn-dlm-iclr-research/task-02/recheck-final/raw-comparison-controls.json"
EXPECTED_REPORT_SHA = "e08dff9441ee0dd78c33bbff33675ebc02fe97875606bfb9a11415ca905381b6"
EXPECTED_MARKDOWN_SHA = "a1d87e99eb1146ea8fd5824024d93a65c565994f63c7e6cdc2ae297b7bc98efd"
EXPECTED_RECEIPT_SHA = "a06b4fe800d7ebbe12941b26e85a5155982689699c45006ae0a3a04c4aef118b"
NUMERIC_FIELDS = (
    "candidate_mean",
    "control_mean",
    "mean_delta",
    "legacy_normal_ci95_half_width",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analysis_hash() -> str:
    digest = hashlib.sha256()
    source = ROOT / "scale/experiments/nonlatent_iclr"
    for name in ("metrics.py", "metric_records.py", "metrics_task.py"):
        digest.update((source / name).read_bytes())
    return digest.hexdigest()


def interpretation(metric: str, delta: float) -> str:
    if metric in {"tau_a_commit_order", "distinct_frac", "residue"}:
        return "diagnostic_only"
    if abs(delta) <= 1e-12:
        return "neutral"
    if metric == "max_run_frac":
        return "worse" if delta > 0 else "better"
    return "better" if delta > 0 else "worse"


def main() -> None:
    report_bytes = REPORT.read_bytes()
    report = json.loads(report_bytes)
    prior = json.loads(PRIOR_PROOF.read_bytes())
    attempt = report["attempt"]
    receipt_path = (
        ROOT
        / ".omo/evidence/nonlatent-rnn-dlm-iclr-research/task-02"
        / attempt
        / "prepare_receipt.json"
    )
    receipt = json.loads(receipt_path.read_bytes())
    current_rows = {}
    panel_hashes_match = True
    prior_panels = {panel["name"]: panel["sha256"] for panel in prior["summary"]["panels"]}
    for comparison in report["comparisons"]:
        candidate = comparison["candidate_panel"]
        control = comparison["control_panel"]
        panel_hashes_match = panel_hashes_match and (
            comparison["candidate_sha256"] == prior_panels[candidate]
            and comparison["control_sha256"] == prior_panels[control]
        )
        for cell in comparison["cells"]:
            for metric in cell["metrics"]:
                key = (candidate, control, cell["arm"], metric["name"])
                current_rows[key] = metric
    prior_rows = {
        (row["candidate_panel"], row["control_panel"], row["arm"], row["metric"]): row
        for row in prior["rows"]
    }
    numeric_matches = 0
    metadata_matches = 0
    for key, prior_row in prior_rows.items():
        current = current_rows[key]
        numeric_matches += sum(current[field] == prior_row["computed"][field] for field in NUMERIC_FIELDS)
        metadata_matches += int(
            current["pair_count"] == 256
            and current["ci_status"] == "descriptive_only_not_multiseed_confirmation"
            and current["interpretation"] == interpretation(current["name"], current["mean_delta"])
        )
    code_hash = analysis_hash()
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    expected_receipt = {
        "analysis_code_sha256": code_hash,
        "attempt": attempt,
        "payload_sha256": report_hash,
        "schema_version": 2,
        "status": "METRICS_COMPLETE",
    }
    checks = {
        "prior_proof_passed": prior["summary"]["overall_pass"] is True,
        "comparison_count": len(report["comparisons"]) == 5,
        "cell_count": sum(len(item["cells"]) for item in report["comparisons"]) == 100,
        "metric_row_count": len(current_rows) == len(prior_rows) == 500,
        "numeric_fields_unchanged": numeric_matches == 2000,
        "metadata_unchanged": metadata_matches == 500,
        "panel_hashes_unchanged": panel_hashes_match,
        "report_sha": report_hash == EXPECTED_REPORT_SHA,
        "markdown_sha": sha256(MARKDOWN) == EXPECTED_MARKDOWN_SHA,
        "receipt_sha": sha256(receipt_path) == EXPECTED_RECEIPT_SHA,
        "analysis_hash": report["analysis_code_sha256"] == code_hash,
        "receipt_exact": receipt == expected_receipt,
    }
    output = {
        "checks": checks,
        "overall_pass": all(checks.values()),
        "attempt": attempt,
        "comparison_count": len(report["comparisons"]),
        "cell_count": sum(len(item["cells"]) for item in report["comparisons"]),
        "metric_row_count": len(current_rows),
        "numeric_match_count": numeric_matches,
        "metadata_match_count": metadata_matches,
        "report_sha256": report_hash,
        "markdown_sha256": sha256(MARKDOWN),
        "receipt_sha256": sha256(receipt_path),
        "analysis_code_sha256": code_hash,
        "prior_proof_sha256": sha256(PRIOR_PROOF),
        "raw_panels_reread": False,
    }
    output_path = Path(__file__).with_name("binding-outcomes.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
