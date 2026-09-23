"""Token ids for the RWKV World vocabulary, and the pack remap that repairs it.

Every number below was measured against the deployed artifacts, not read off a
model card.  The measurements are repeated in tests/test_tokenizer_ids.py so a
vocabulary swap fails loudly instead of silently retraining on different tokens.

The vocabulary
--------------
``rwkv_vocab_v20230424.txt`` is a *trie* file whose lines are
``"<id> '<escaped token>' <byte length>"``.  Its ids run 1..65529 with no gaps.
Notably the line index is not the id: line 260 carries id 261 = ``b'\\n\\n'``.

  * id 0      -> ``<|rwkv_tokenizer_end_of_text|>``, the pad/EOT slot (not in the trie)
  * id 261    -> ``b'\\n\\n'``
  * id 65529  -> 63 spaces, the last defined id
  * id 65530  -> **undefined**; 65531..65535 likewise

The defect
----------
The packs were tokenized through the HuggingFace ``RwkvTokenizer`` wrapper, whose
``special_tokens_map.json`` declares ``eos_token = "\\n\\n"``.  ``PreTrainedTokenizer``
therefore appends ``"\\n\\n"`` as an *added token* past the end of the vocabulary,
at id 65530 -- while the native trie encodes the same string as 261.  Measured on
``dclm_4096_packed``: 65530 is **1.457 %** of tokens over a 24.6M-token stride
across the whole pack and 1.521 % over 8.19M contiguous tokens.

That matters because 65530's embedding row is untrained.  Its row norm is
2.9e-19 -- the bf16 subnormal floor, i.e. an all-but-zero row -- against a
median row norm of 0.517.  (``lm_head`` row 65530 is a normal-looking 0.79, so
this is an untrained *embedding*, not an unused id.)  Training on the packs as
written feeds a zero embedding to one token in 68.

The repair
----------
``remap_pack_ids`` rewrites 65530 -> 261.  It is lossless: id 65530 is not in
``idx2token`` at all, and both ids denote the identical string ``"\\n\\n"``.
Decoding a pack row through the remap yields clean English.

The remap is recorded, not just applied.  A load-time remap changes every row of
a corpus while any digest taken over the source shard list stays byte-identical,
so two token generations would otherwise be indistinguishable to a later reader.
``remap_receipt`` writes what was changed into the run record and the corpus
identity, which is what makes the two generations tellable apart.
"""
from __future__ import annotations

from pathlib import Path

import torch

# --- structural ids -------------------------------------------------------
PAD_ID = 0
"""Pad / EOT.  ``added_tokens.json`` maps the EOT string to 0, and the pack
convention is ``mask = ids != 0``."""

MASK_ID = 65535
"""Absorbing-mask slot.  Unassigned in the trie (65530..65535 are all free) and
its embedding row is at the subnormal floor, so it carries no pretrained meaning
and cannot collide with real data."""

# --- the newline pair -----------------------------------------------------
NEWLINE_PAIR_ID = 261
"""``b'\\n\\n'`` under the native trie.  Confirmed by encoding and by the vocab
file's own line for id 261."""

LEGACY_WRAPPER_NEWLINE_PAIR_ID = 65530
"""The HuggingFace wrapper's added-token id for the same string.  Undefined in
the trie; present in the packs at ~1.5 % of tokens."""

# --- the vocabulary's real extent ----------------------------------------
VOCAB_MIN_ID = 1
VOCAB_MAX_ID = 65529
VOCAB_SIZE = 65536  # the embedding table's row count

DEFAULT_VOCAB_PATH = ("/inspire/hdd/global_user/zhangjiaquan-253108540222/models/"
                      "rwkv7-0.4B-world/rwkv_vocab_v20230424.txt")
"""The vocabulary file every component reads, so a build and an evaluation cannot
disagree about which ids mean what."""

# --- untrained embedding rows --------------------------------------------
UNTRAINED_EMBEDDING_IDS = (
    193, 194, 196,
    246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256,
    58597,
    65530, 65531, 65532, 65533, 65534, 65535,
)
"""Rows whose embedding norm is below 1e-6 (all but the 65530..65535 tail sit at
1e-19, the bf16 subnormal floor) while the table's median row norm is 0.517.

Measured over the whole pack: **only 65530 is ever emitted** by any pack.  Every
other row here has zero occurrences in a 24.6M-token stride, so one remap is the
complete repair -- there is no second site of this defect."""

REMAP = {LEGACY_WRAPPER_NEWLINE_PAIR_ID: NEWLINE_PAIR_ID}

FORBIDDEN_SAMPLING_IDS = (0, 65530, 65531, 65532, 65533, 65534, 65535)
"""Ids the sampler may never emit: the pad/EOT slot, the undefined tail, and the
mask slot.  Recorded verbatim as ``vocab_restriction`` on every sampler run."""


def remap_pack_ids(ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Rewrite the wrapper's added-token id to the native trie's id.

    Returns ``(remapped, n_changed)``.  The count is returned rather than
    discarded because it is the evidence that a corpus generation was remapped;
    a caller that drops it cannot later distinguish the two.
    """
    changed = int((ids == LEGACY_WRAPPER_NEWLINE_PAIR_ID).sum())
    if changed == 0:
        return ids, 0
    return torch.where(ids == LEGACY_WRAPPER_NEWLINE_PAIR_ID,
                       torch.full_like(ids, NEWLINE_PAIR_ID), ids), changed


def remap_receipt(before: int, after: int, tokens_scanned: int) -> dict:
    """A record of one remap, for the run registry and the corpus identity."""
    return {
        "remap": {str(k): v for k, v in REMAP.items()},
        "reason": ("id 65530 is the HuggingFace wrapper's added token for '\\n\\n' "
                   "and is unassigned in rwkv_vocab_v20230424.txt; id 261 is the "
                   "native trie's id for the same string"),
        "lossless": True,
        "tokens_scanned": tokens_scanned,
        "tokens_changed_before": before,
        "tokens_changed_after": after,
        "fraction_before": (before / tokens_scanned) if tokens_scanned else 0.0,
    }


def load_vocab_ids(vocab_path: str | Path) -> dict[int, str]:
    """Parse the trie vocabulary into ``{id: escaped token body}``.

    The token body is returned exactly as the file writes it -- escaped, so
    ``id 261`` is the four characters ``\\n\\n`` rather than two newlines.  The
    vocabulary contains non-Latin-1 code points, and unescaping to raw bytes
    would make the parse depend on the codec; the escaped form is what the file
    actually asserts and is comparable without any decoding.

    Exists so tests can assert the id space rather than trusting a docstring.
    """
    table: dict[int, str] = {}
    with open(vocab_path, "rb") as handle:
        text = handle.read().decode("utf-8")
    for line in text.split("\n"):
        if not line.strip():
            continue
        id_str, rest = line.split(" ", 1)
        if not id_str.isdigit():
            continue
        # ``'<escaped token>' <byte length>``; the token may contain spaces and
        # quotes, so split the trailing length field off from the right.
        body = rest.rsplit(" ", 1)[0]
        if body.startswith("'") and body.endswith("'"):
            body = body[1:-1]
        table[int(id_str)] = body
    return table
