from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


metrics = import_module("scale.experiments.nonlatent_iclr.metrics")
metrics_task = import_module("scale.experiments.nonlatent_iclr.metrics_task")


def main() -> None:
    metric = metrics.MetricName.MAX_RUN_FRAC
    original_direction = metrics.METRIC_DIRECTIONS[metric]
    direct = metrics.paired_delta(
        metric,
        (metrics.Pair("probe", 1, "r100_s1", 0.9),),
        (metrics.Pair("probe", 1, "r100_s1", 0.1),),
    )
    normal = metrics_task.failure_probe_task_two()
    metrics.METRIC_DIRECTIONS[metric] = metrics.MetricDirection.HIGHER_IS_BETTER
    try:
        reversed_direction = metrics_task.failure_probe_task_two()
    finally:
        metrics.METRIC_DIRECTIONS[metric] = original_direction
    checks = {
        "direct_delta": direct.mean_delta == 0.8,
        "direct_interpretation": direct.interpretation == "worse",
        "normal_probe": normal.status == "EXPECTED_FAILURE_CONFIRMED",
        "reversed_direction_rejected": reversed_direction.status == "PROBE_FAILED",
        "direction_restored": metrics.METRIC_DIRECTIONS[metric] == original_direction,
    }
    output = {
        "checks": checks,
        "overall_pass": all(checks.values()),
        "direct": {"mean_delta": direct.mean_delta, "interpretation": direct.interpretation},
        "normal_probe": {"status": normal.status, "detail": normal.detail},
        "reversed_direction_probe": {
            "status": reversed_direction.status,
            "detail": reversed_direction.detail,
        },
    }
    output_path = Path(__file__).with_name("polarity-outcomes.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
