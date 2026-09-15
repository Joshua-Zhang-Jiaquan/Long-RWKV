"""Make the repository runnable from a clean clone.

Two things need help, and neither is fixed by editing the sources that need it:

1. ``DAN/v7_arch_round/code/`` is written for a flat import root -- its modules
   do ``from models.x import y`` -- so the directory has to be on ``sys.path``
   when its own tests are collected. Those files are hash-bound to the recorded
   evidence, so the path is supplied here instead of the import being rewritten.

2. ``pytest`` run bare from the repository root would otherwise collect archived
   submission payloads under ``.omo/evidence/**/submission/``, which are records
   of what was submitted, not tests of this repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_MODEL_CODE = ROOT / "DAN" / "v7_arch_round" / "code"
for _path in (_MODEL_CODE, _MODEL_CODE / "models", _MODEL_CODE / "train"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

collect_ignore_glob = [".omo/evidence/**/submission/*"]
