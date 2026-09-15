"""Item 4 (data_plan.md): frozen tokenizer contract for the RNN and DLM views.

Freezes the token ids and semantics ALREADY load-bearing across every packed
corpus (v6_data §2), qualifies round-trip fidelity on a multilingual +
code-heavy + math sample set, and publishes the contract hash that every
derived view must cite.

The contract is immutable. Changing any field requires a new contract version
and a new view family.

Qualification evidence: item04_token_contract.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODEL_DIR = "/inspire/hdd/global_user/zhangjiaquan-253108540222/models/RWKV7-Goose-World3-2.9B-HF"

CONTRACT = {
    "contract_version": 1,
    "tokenizer": "RWKV7-Goose-World3 (trie, AutoTokenizer trust_remote_code)",
    "tokenizer_file": f"{MODEL_DIR}/tokenizer.json",
    "vocab_size": 65531,
    "special_tokens": {
        "PAD": {"id": 0, "semantics": "padding only; attention_mask=0; never loss-bearing; appears only in inter-doc gaps in packed views"},
        "EOS": {"id": 65530, "semantics": "legacy separator, decodes to \\n\\n; read-only in new packs; never written by the current packer"},
        "SEP": {"id": 65531, "semantics": "document separator; written at every doc_end-1 in packed views; never inside a document"},
        "MASK": {"id": 65535, "semantics": "absorbing-mask corruption token (DLM view); never present in packed/canonical data; trainer writes it at corruption time"},
        "RWKV_TOKENIZER_END_OF_TEXT": {"id": 0, "semantics": "tokenizer-level special mapped to the PAD slot; not used as a boundary in packed views"},
    },
    "reserved_ids": "ids >= 65531 (except SEP at 65531) are reserved and must not appear in any packed or canonical record",
    "control_semantics": {
        "block_t": "per-64-token-block realised masked fraction; derived-view quantity only (B3 conditioning)",
        "prompt_end": "doc-relative token index of the prompt/completion boundary; derived-view sidecar; feeds the docgen/FIM lane",
        "state_reset": "per-doc bool; RNN-view sidecar; True = recurrent state resets at this doc start",
    },
    "cjk_note": "zh at ~1.0 char/token (8,244 single-CJK tokens + byte fallback); accepted inefficiency — switching tokenizer abandons the pretrained World3 init",
    "evidence": "D7 checker enforces SEP placement and the >=65532 exclusion on every shard; round-trip qualification below",
}

QUALIFICATION_SAMPLES = [
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    return quicksort([x for x in a[1:] if x < a[0]]) + a[:1] + quicksort([x for x in a[1:] if x >= a[0]])\n",
    "Solve for x: 2x + 3 = 11. The derivative of x^2 is 2x. The integral of 1/x is ln|x|.\n\\frac{d}{dx}\\sin(x) = \\cos(x)\n",
    "人工智能正在改变世界。扩散语言模型通过任意顺序回归生成文本，这与传统的自回归方法截然不同。求解方程：3x - 6 = 0，得 x = 2。\n",
    "Mixed English 中文 and code `for i in range(10): print(i)` with symbols ∑ ∫ ≤ ≥ α β γ.\n",
]


def contract_hash() -> str:
    """sha256 over the contract fields + tokenizer.json bytes (frozen)."""
    payload = json.dumps(CONTRACT, sort_keys=True).encode()
    tok = Path(CONTRACT["tokenizer_file"]).read_bytes()
    return hashlib.sha256(payload + tok).hexdigest()


def qualify() -> dict:
    """Round-trip qualification on the sample set; fail-closed on any loss."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    results = []
    all_ok = True
    for i, text in enumerate(QUALIFICATION_SAMPLES):
        ids = tok(text)["input_ids"]
        back = tok.decode(ids)
        ok = back == text
        reserved = [t for t in ids if t >= 65532] + [t for t in ids if t == 65531]
        clean = not reserved
        all_ok = all_ok and ok and clean
        results.append({
            "sample": i,
            "chars": len(text),
            "tokens": len(ids),
            "roundtrip_lossless": ok,
            "no_reserved_ids": clean,
        })
    return {"samples": results, "pass": all_ok, "tokenizer_sha256_prefix": hashlib.sha256(Path(CONTRACT['tokenizer_file']).read_bytes()).hexdigest()[:24]}


def main() -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="/inspire/hdd/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/DiffRWKV-RELAY/DAN/data_plan_evidence/item04_token_contract.json")
    args = parser.parse_args()

    qual = qualify()
    report = dict(CONTRACT)
    report["contract_hash"] = contract_hash()
    report["qualification"] = qual
    out = Path(args.evidence)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"contract_hash: {report['contract_hash'][:24]}")
    print(f"qualification pass: {qual['pass']} ({len(qual['samples'])} samples)")
    print(f"evidence -> {out}")
    return 0 if qual["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
