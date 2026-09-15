"""Checks of the per-rank arm table: coverage, derived arms, and runtime pinning."""

from __future__ import annotations

from pathlib import Path

import pytest

from scale.experiments.nonlatent_iclr.qualification import calibration_arms as arms


def test_rank_table_covers_every_rank_exactly_once() -> None:
    # Given / When: the declared rank assignment.
    table = arms.RANK_ARM_TABLE

    # Then: every rank 0..7 has exactly one arm, and every named arm exists.
    assert sorted(table) == list(range(arms.RANK_COUNT))
    assert set(table.values()) <= {spec.arm_id for spec in arms.ARM_SPECS}


def test_arm_digests_are_unique_and_stable() -> None:
    # Given: the declared arms.
    digests = {spec.arm_id: spec.digest() for spec in arms.ARM_SPECS}

    # Then: each arm has its own digest, and re-deriving it gives the same value.
    assert len(set(digests.values())) == len(arms.ARM_SPECS)
    assert all(len(digest) == 64 for digest in digests.values())
    assert arms.arm_by_id("A3_large_loop").digest() == digests["A3_large_loop"]


def test_derived_arms_are_declared_and_measure_nothing() -> None:
    # Given: the arms this job cannot run.
    assert set(arms.DERIVED_ARMS) == {"A0", "A4"}

    # Then: their derivation targets are real measured arms, and no derived arm is in the table.
    declared = {spec.arm_id for spec in arms.ARM_SPECS}
    assert set(arms.DERIVED_ARMS.values()) <= declared
    assert not (set(arms.DERIVED_ARMS) & declared)


def test_only_checkpoint_arms_declare_a_checkpoint() -> None:
    # Given / When: the arms are inspected.
    checkpointed = [spec for spec in arms.ARM_SPECS if spec.checkpoint_dir is not None]

    # Then: exactly the two large arms load the verified step-4750 checkpoint, and nothing else does.
    assert {spec.arm_id for spec in checkpointed} == {"A2_large_noloop", "A3_large_loop"}
    for spec in checkpointed:
        assert spec.checkpoint_step == arms.LARGE_CHECKPOINT_STEP
        assert spec.model_root == arms.LARGE_MODEL_ROOT
    for spec in arms.ARM_SPECS:
        if spec.checkpoint_dir is None:
            assert spec.checkpoint_step is None


def test_small_arms_use_a_worker_reachable_root() -> None:
    # Given: the small arms.
    small = [spec for spec in arms.ARM_SPECS if spec.model_root == arms.SMALL_MODEL_ROOT]

    # When / Then: their root is a worker-reachable external root, never the
    # project-volume copy that qz workers cannot mount.
    #
    # This used to assert a hard-coded /inspire/hdd/global_user/ prefix, which
    # made the test pass only on the host that produced it. The property that
    # actually matters is that the root is configurable and lives OUTSIDE the
    # project volume -- not that it spells one particular host path.
    assert small
    root = arms.SMALL_MODEL_ROOT
    assert root == arms.GLOBAL_USER / "models/rwkv7-0.4B"
    assert "base_models" not in str(root)
    assert not str(root).startswith("/inspire/hdd/project/")


def test_no_loop_arm_reuses_the_configured_depth_for_a_strict_load() -> None:
    # Given: the large no-loop arm, which must still load a checkpoint that contains loop keys.
    spec = arms.arm_by_id("A2_large_noloop")

    # When / Then: it is built at the configured depth and measured at zero recycled passes.
    assert spec.loop_reps == 1
    assert spec.measured_reps() == 0
    assert arms.arm_by_id("A3_large_loop").measured_reps() == 1


def test_loop_range_and_reps_are_declared_together() -> None:
    # Given / When / Then: a loop without a range, and a range without loop passes, are both refused.
    with pytest.raises(ValueError, match="loop_reps_requires_loop_range"):
        _ = arms.ArmSpec(
            arm_id="bad", nominal_label="bad", model_root=arms.SMALL_MODEL_ROOT, loop_reps=1,
            force_forward=False, optimizer_step_measured=False, measured=True, notes="x",
        )
    with pytest.raises(ValueError, match="loop_range_without_loop_reps"):
        _ = arms.ArmSpec(
            arm_id="bad", nominal_label="bad", model_root=arms.SMALL_MODEL_ROOT,
            loop_range=(1, 2), loop_reps=0, force_forward=False,
            optimizer_step_measured=False, measured=True, notes="x",
        )


def test_an_arm_must_be_either_measured_or_derived() -> None:
    # Given / When / Then: an arm claiming both, and one claiming neither, are refused.
    with pytest.raises(ValueError, match="either_measured_or_derived"):
        _ = arms.ArmSpec(
            arm_id="bad", nominal_label="bad", model_root=arms.SMALL_MODEL_ROOT, loop_reps=0,
            force_forward=False, optimizer_step_measured=False, measured=True,
            derived_from="A1_small_noloop", notes="x",
        )
    with pytest.raises(ValueError, match="either_measured_or_derived"):
        _ = arms.ArmSpec(
            arm_id="bad", nominal_label="bad", model_root=arms.SMALL_MODEL_ROOT, loop_reps=0,
            force_forward=False, optimizer_step_measured=False, measured=False, notes="x",
        )


def test_a_derived_arm_cannot_claim_a_checkpoint_or_an_optimizer() -> None:
    # Given / When / Then: a derived arm carrying measurement machinery is refused.
    with pytest.raises(ValueError, match="derived_arm_cannot_claim_measurements"):
        _ = arms.ArmSpec(
            arm_id="A4", nominal_label="untied extra blocks", model_root=arms.SMALL_MODEL_ROOT,
            loop_reps=0, force_forward=False, optimizer_step_measured=True, measured=False,
            derived_from="A3_small_loop", notes="x",
        )


def test_verify_arm_weights_pins_the_small_model(monkeypatch, tmp_path: Path) -> None:
    # Given: the small arm pointed at a directory whose bytes do not match the pinned digest.
    root = tmp_path / "small"
    root.mkdir()
    _ = (root / "model.safetensors").write_bytes(b"not the real weights")
    _ = (root / "config.json").write_text("{}", encoding="utf-8")
    spec = arms.arm_by_id("A2_small_noloop").model_copy(update={"model_root": root})

    # When / Then: the in-job pin refuses to measure a different model.
    with pytest.raises(ValueError, match="arm_weights_digest_mismatch"):
        _ = arms.verify_arm_weights(spec)


def test_verify_arm_weights_reports_absent_weights(tmp_path: Path) -> None:
    # Given: an arm root with a config but no weights at all.
    root = tmp_path / "empty"
    root.mkdir()
    _ = (root / "config.json").write_text("{}", encoding="utf-8")
    spec = arms.arm_by_id("A3_large_loop").model_copy(
        update={"model_root": root, "checkpoint_dir": None, "checkpoint_step": None}
    )

    # When / Then: absence is named rather than silently measuring nothing.
    with pytest.raises(ValueError, match="arm_weights_absent"):
        _ = arms.verify_arm_weights(spec)


def test_measure_parameters_separates_the_loop_share() -> None:
    # Given: a toy module with one loop tensor and two trunk tensors.
    import torch

    class _Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Linear(4, 4)
            self.loop = torch.nn.Linear(4, 4)

    measured = arms.measure_parameters(_Toy())

    # Then: totals cover everything while the loop's own share is reported separately.
    assert measured.total_tensors == 4  # two weights, two biases
    assert measured.loop_tensors == 2
    assert measured.loop_numel == 20
    assert measured.total_numel == 40


def test_measure_geometry_accepts_a_model_without_grouped_heads() -> None:
    # Given: a config shape like the small model, where num_heads is null.
    import torch

    class _Cfg:
        num_hidden_layers = 24
        hidden_size = 1024
        vocab_size = 65536
        num_heads = None
        head_dim = 64
        intermediate_size = 4096

    class _Model(torch.nn.Module):
        config = _Cfg()

    geometry = arms.measure_geometry(_Model())

    # Then: the ungrouped-head case is representable rather than forced into an int.
    assert geometry.num_hidden_layers == 24
    assert geometry.num_heads is None


def test_the_forward_call_carries_the_measured_depth_override() -> None:
    # Given: the large no-loop arm, built at loop_reps=1 but measured at zero recycled passes.
    from scale.experiments.nonlatent_iclr.qualification import calibration as cal

    arm = arms.arm_by_id("A2_large_noloop")

    # When: the model call kwargs are built for the timed forward.
    kwargs = cal._call_kwargs(arm)

    # Then: the forward runs the depth the record declares, not the depth the model was built at.
    # Without this the timed forward runs the loop while the record says it did not, and the
    # forward stops being comparable with its own backward.
    assert kwargs["loop_reps_override"] == 0
    assert kwargs["force_forward"] is False
    assert kwargs["use_cache"] is False


def test_an_arm_that_executes_its_configured_depth_carries_no_override() -> None:
    # Given / When: an arm whose configured and measured depths agree.
    from scale.experiments.nonlatent_iclr.qualification import calibration as cal

    for arm_id in ("A1_small_noloop", "A2_small_noloop", "A3_small_loop", "A3_large_loop"):
        kwargs = cal._call_kwargs(arms.arm_by_id(arm_id))

        # Then: no override is sent, so the model's own configuration governs.
        assert "loop_reps_override" not in kwargs
