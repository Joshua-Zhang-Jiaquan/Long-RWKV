from __future__ import annotations

import inspect
import json
from importlib import import_module

metrics_module = import_module("scale.experiments.nonlatent_iclr.metrics")
MetricInputError = metrics_module.MetricInputError
prompt_preserving_corruption = metrics_module.prompt_preserving_corruption
sequence_metrics = metrics_module.sequence_metrics
tau_a_commit_order = metrics_module.tau_a_commit_order


def rejects_partition(condition: tuple[bool, ...], prediction: tuple[bool, ...]) -> tuple[bool, str]:
    try:
        prompt_preserving_corruption(
            (11, 12, 21, 22, 0),
            condition,
            prediction,
            pad_id=0,
            mask_token_id=99,
        )
    except MetricInputError as error:
        return True, str(error)
    return False, "accepted"


def main() -> None:
    prediction_control = sequence_metrics(
        (1, 2, 3, 4, 0),
        (9, 9, 9, 4, 0),
        (True, True, True, True, False),
        pad_id=0,
    )
    zero_mask = sequence_metrics(
        (1, 2, 0),
        (1, 2, 0),
        (False, False, False),
        pad_id=0,
    )
    valid = prompt_preserving_corruption(
        (11, 12, 21, 22, 0),
        (True, True, False, False, False),
        (False, False, True, True, False),
        pad_id=0,
        mask_token_id=99,
    )
    unclassified_rejected, unclassified_detail = rejects_partition(
        (True, False, False, False, False),
        (False, False, True, False, False),
    )
    overlap_rejected, overlap_detail = rejects_partition(
        (True, True, True, False, False),
        (False, False, True, True, False),
    )
    tau_actual = tau_a_commit_order((1, 1, 2), (True, True, True))
    signature = inspect.signature(prompt_preserving_corruption)
    checks = {
        "prediction_run": prediction_control.max_run_frac == 0.75,
        "tau_a_ties": tau_actual == 2 / 3,
        "zero_mask_nullable": zero_mask.masked_token_accuracy is None,
        "valid_partition": valid.corrupted == (11, 12, 99, 99, 0),
        "unclassified_rejected": unclassified_rejected,
        "overlap_rejected": overlap_rejected,
        "seed_parameter_removed": "seed" not in signature.parameters,
    }
    output = {
        "checks": checks,
        "overall_pass": all(checks.values()),
        "prediction_run": {
            "actual": prediction_control.max_run_frac,
            "expected": 0.75,
        },
        "tau_a_hand_control": {
            "actual": tau_actual,
            "expected": 2 / 3,
            "commit_steps": [1, 1, 2],
        },
        "zero_mask": {
            "masked_token_accuracy": zero_mask.masked_token_accuracy,
            "whole_sequence_exact_match_observed_not_used_as_generation_qualification": zero_mask.whole_sequence_exact_match,
        },
        "partition": {
            "valid_corrupted": list(valid.corrupted),
            "unclassified_detail": unclassified_detail,
            "overlap_detail": overlap_detail,
        },
        "public_helper_signature": str(signature),
        "scope": "CPU public helper qualification only; no generated sequence or GPU sampling was run",
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
