"""The HuggingFace-checkpoint key remap, and what counts as a new module.

This lives outside ``model.py`` on purpose.  ``model.py`` imports ``fla`` at module
scope, so on a host without a Triton driver nothing in it can be imported -- which
means a pure function inside it has no CPU test, and its bugs are invisible until a
GPU job runs one.  That is exactly what happened: the new-module allowlist matched
its prefixes against the *start* of the whole parameter name, so
``layers.7.fuse_down.weight`` was rejected while ``input_cond.trunk.0.weight`` was
accepted, and a ``from_hf_pretrained`` call failed 90 seconds into a queued H100 job
with ``72 model parameters have no checkpoint source``.

The rule itself is small and worth stating once:

* ``model.layers.{i}.attn.{tail}`` -> ``layers.{i}.attn.{tail}`` -- **one**
  destination per tensor.  The prior implementation's ``remap_hf_key`` wrote every
  attention tensor into *both* of its two mixers, which is the parameter doubling
  the tied design removes.
* ``model.embeddings.*`` / ``model.norm.*`` -> drop the ``model.`` prefix.
* ``lm_head.weight`` -> unchanged.
* Anything else is refused, because a silently ignored tensor is a silently
  untrained module.
"""
from __future__ import annotations

import re

from .refusal import Refusal

#: ``layers.<index>.`` at the start of a parameter name.
LAYER_PREFIX = re.compile(r"^layers\.\d+\.")

#: Modules the released checkpoint has no source for, in the order they appear in
#: a parameter name after the layer prefix is removed.
NEW_MODULE_PREFIXES: tuple[str, ...] = (
    "fuse_down",     # directional fusion, low rank
    "fuse_up",
    "fuse_bias",
    "gate_mods.",    # per-layer gate modulator (A3)
    "input_cond.",   # input-level noise conditioner (A1/A2/A3)
)

#: Top-level parameters that keep their keys.
PASSTHROUGH_KEYS: frozenset[str] = frozenset({"lm_head.weight"})


def is_new_module_parameter(name: str,
                            allow_prefixes: tuple[str, ...] = NEW_MODULE_PREFIXES
                            ) -> bool:
    """Is this model parameter one the released checkpoint has no source for?

    The layer prefix is stripped before matching.  Matching against the start of the
    whole name accepts ``input_cond.trunk.0.weight`` and rejects
    ``layers.7.fuse_down.weight``, which for a 24-layer model means 72 fusion
    parameters look like a remap failure.
    """
    tail = LAYER_PREFIX.sub("", name)
    return any(tail.startswith(prefix) for prefix in allow_prefixes)


def remap_hf_key(key: str) -> str:
    """Map one checkpoint key to its destination, or refuse it."""
    if key in PASSTHROUGH_KEYS:
        return key
    if key.startswith("model.layers.") or key.startswith("model.embeddings.") \
            or key.startswith("model.norm."):
        return key[len("model."):]
    raise Refusal(f"unrecognized checkpoint key {key!r}")
