from __future__ import annotations

import json
import math
from importlib import import_module
from pathlib import Path
from typing import Protocol


metrics = import_module("scale.experiments.nonlatent_iclr.metrics")
metrics_task = import_module("scale.experiments.nonlatent_iclr.metrics_task")


class MetricLike(Protocol):
    value: str


class PairLike(Protocol):
    document_id: str
    seed: int
    arm: str
    value: float


def main() -> None:
    original_paired = metrics.paired_delta
    original_prompt = metrics_task.prompt_preserving_corruption
    pair_calls: list[dict[str, object]] = []
    prompt_calls: list[dict[str, object]] = []

    def traced_paired(metric: MetricLike, left: tuple[PairLike, ...], right: tuple[PairLike, ...]) -> object:
        raw_left_values = [pair.value for pair in left]
        raw_right_values = [pair.value for pair in right]
        left_values = [value if math.isfinite(value) else "NaN" for value in raw_left_values]
        right_values = [value if math.isfinite(value) else "NaN" for value in raw_right_values]
        left_keys = [(pair.document_id, pair.seed, pair.arm) for pair in left]
        right_keys = [(pair.document_id, pair.seed, pair.arm) for pair in right]
        pair_calls.append({
            "metric": metric.value,
            "left_values": left_values,
            "right_values": right_values,
            "keys_match": left_keys == right_keys,
            "all_finite": all(math.isfinite(value) for value in raw_left_values + raw_right_values),
        })
        return original_paired(metric, left, right)

    def traced_prompt(*args: object, **kwargs: object) -> object:
        prompt_calls.append({"positional_count": len(args), "keyword_names": sorted(kwargs)})
        return original_prompt(*args, **kwargs)

    setattr(metrics, "paired_delta", traced_paired)
    setattr(metrics_task, "prompt_preserving_corruption", traced_prompt)
    try:
        result = metrics_task.failure_probe_task_two()
    finally:
        setattr(metrics, "paired_delta", original_paired)
        setattr(metrics_task, "prompt_preserving_corruption", original_prompt)

    typed = original_paired(
        metrics.MetricName.MAX_RUN_FRAC,
        (metrics.Pair("doc", 42, "r70_s1", 0.9),),
        (metrics.Pair("doc", 42, "r70_s1", 0.1),),
    )
    explicit_polarity_calls = sum(
        call["all_finite"] is True
        and call["keys_match"] is True
        and call["left_values"] != call["right_values"]
        for call in pair_calls
    )
    checks = {
        "cli_status_success": result.status == "EXPECTED_FAILURE_CONFIRMED",
        "prompt_probe_executed": len(prompt_calls) == 1,
        "nonfinite_probe_executed": any(call["all_finite"] is False for call in pair_calls),
        "missing_pair_probe_executed": any(call["keys_match"] is False for call in pair_calls),
        "polarity_probe_executed": explicit_polarity_calls == 1,
        "typed_polarity_correct": typed.mean_delta == 0.8 and typed.interpretation == "worse",
    }
    output = {
        "checks": checks,
        "cli_claim_complete": all(checks.values()),
        "result": {"status": result.status, "detail": result.detail},
        "pair_calls": pair_calls,
        "prompt_calls": prompt_calls,
        "explicit_polarity_call_count": explicit_polarity_calls,
        "typed_polarity": {"mean_delta": typed.mean_delta, "interpretation": typed.interpretation},
        "plan_assessment": "failure CLI omits the finite reversed-polarity control even though the typed metric and scoped unit test cover the behavior",
        "strict_plan_acceptance": False,
    }
    output_path = Path(__file__).with_name("failure-cli-trace.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
