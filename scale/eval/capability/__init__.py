"""Capability evaluation harness for DiffRWKV (RELAY/DAN) checkpoints.

Loads a ``step_*/model.pt`` checkpoint, rebuilds ``StateInjectionDiTRELAY``, and
scores it on downstream capability benchmarks:

* **Multichoice knowledge** (MMLU, ARC-Challenge, HellaSwag): length-normalized
  choice-span loglikelihood, scored under one or more "conditions" — the raw
  frozen backbone (``raw``) and diffusion-injected states (``ddpm<N>`` /
  ``ddim<N>`` / ``flow<N>``). Data is the pre-tokenized ``.npz`` format already
  present in ``preprocessed_data/`` (``input_ids[N_choices, L]``,
  ``attention_mask``, ``choice_start``, ``label``).
* **Generation** (GSM8K / MATH-500, HumanEval / MBPP): prompted generation via
  the RELAY injected-cache decoder, then answer extraction / sandboxed exec
  (plugged in by the task-family modules; see ``run_eval.py``).

The scorer logic is ported from the external
``research/DiffRwkv/scripts/eval/eval_state_hijack_multichoice_shared.py`` so the
numbers are directly comparable to the prior MMLU smoke, but imports the model
class from this repo's ``models.state_hijacking_dit`` and uses the repo's
``compatible_trainable_state`` checkpoint shim for fidelity to the trainer.

Local work is CPU-only (repo rule); real scoring runs inside an approved qz job
(see ``scale/qz/launch_capability_eval.sh``). Every run is sharded by GPU rank
and emits a per-shard JSON that ``merge_shards.py`` folds into a bootstrap-CI
summary.
"""
