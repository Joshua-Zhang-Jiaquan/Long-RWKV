from __future__ import annotations

from pathlib import Path

import pytest
import os

from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import qualification_projection
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import qualification_sources_current
from scale.experiments.nonlatent_iclr.tasks.tokenizer_qualification import write_qualification


MODEL_ROOT: Final = Path(os.environ.get(
    "NONLATENT_MODEL_DIR",
    Path(__file__).resolve().parents[3] / "external/models/RWKV7-Goose-World3-2.9B-HF"))


def test_actual_local_qualification_projects_few_public_prompts_without_mutating_registry(tmp_path: Path) -> None:
    # Given: the reviewed local RWKV tokenizer artifact root
    output = tmp_path / "tokenizer"

    # When: CPU-only qualification writes a manifest and sampled projection
    result = write_qualification(MODEL_ROOT, output, tmp_path / "evidence")

    # Then: encoder results are qualified separately from matrix token budgets
    assert result.encoding_qualified is True
    assert result.matrix_token_lengths_qualified is False
    assert len(qualification_projection(result)) == 4
    assert (output / "rwkv_tokenizer_manifest.json").is_file()


def test_stale_qualification_procedure_hash_is_rejected(tmp_path: Path) -> None:
    # Given: source hashes from a current local CPU qualification
    result = write_qualification(MODEL_ROOT, tmp_path / "output", tmp_path / "evidence")
    stale = dict(result.source_hashes)
    stale["tokenizer_qualification.py"] = "0" * 64

    # When: a receipt carries a stale procedure hash
    # Then: it is not eligible as current qualification evidence
    assert qualification_sources_current(stale, MODEL_ROOT) is False


# The artifacts these tests exercise are not part of the repository: they are
# large, rights-gated, and produced on a cluster. A clone must SKIP rather than
# fail, so the suite reports honestly what it can verify without them. Point
# NONLATENT_MODEL_DIR at a prepared root to run them.
pytestmark = pytest.mark.skipif(
    not MODEL_ROOT.exists(),
    reason=f"external artifact absent: {MODEL_ROOT} -- set the matching "
           f"NONLATENT_* environment variable to a prepared root")
