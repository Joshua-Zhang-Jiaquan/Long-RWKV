"""Build MMLU task files under the paper's **one-token answer-label** convention.

The manuscript is explicit (Long_RWKV_ICLR2027.tex:510):

    "MMLU uses a common zero-shot question/options template and a one-token
     answer-label convention verified separately for each tokenizer ...
     Multi-token option pseudolikelihood is not silently substituted for a
     generative probability."

The pre-tokenized MMLU already on GPFS
(``research/DiffRwkv/preprocessed_data/mmlu/test/``) does the forbidden thing: its
choice spans are the full option *texts*, measured at 1 to 44 tokens across a
400-file sample.  Scoring those is exactly the multi-token pseudolikelihood the
paper rules out, so those files cannot serve as this study's primary MMLU
protocol no matter how convenient they are.  This module builds the compliant
files instead: the four scored continuations are " A", " B", " C", " D", each a
single token, so the comparison is one categorical next-token distribution over
four ids -- the same quantity for every arm and every tokenizer.

Verified, not assumed: " A".." D" encode to exactly one token under all six
tokenizers in this study -- RWKV world (ids 300-303, shared by the 2.9B and both
0.4B bases), Qwen2.5 (362/425/356/422), Llama-3.2 (362/426/356/423) and LLaDA
(355/413/348/435).  :func:`label_ids` re-checks this at build time and refuses
rather than falling back, because a silent fallback to multi-token scoring for one
arm is precisely the substitution the paper forbids, and it would be invisible in
the output files.

The emitted format is the ``.npz`` layout ``eval/capability/multichoice.py``
already reads (``input_ids [N,L]``, ``attention_mask``, ``choice_start [N]``,
``label [1]``), so no new scorer is needed: with ``choice_start = L-1`` the
existing ``_span_nll`` reduces to the NLL of a single label token, and ``fwdce``
(causal, shifted) is the protocol-faithful condition.  All four rows share one
prompt and differ only in the final token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Final

LABELS: Final = ("A", "B", "C", "D")

#: The scored continuations.  The leading space is load-bearing: " A" is one token
#: in every tokenizer here, while "A" after a newline would re-tokenize against the
#: preceding character and is not guaranteed to be.
LABEL_STRINGS: Final = tuple(f" {c}" for c in LABELS)

#: The zero-shot template, frozen here and identical for every arm.  It ends with
#: "Answer:" and no trailing space, because the space belongs to the scored token.
#: Changing this string changes every number, so it lives in one place and its
#: sha256 goes into the output manifest.
TEMPLATE: Final = (
    "The following is a multiple choice question about {subject}.\n\n"
    "{question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)


def template_sha256() -> str:
    return hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()


def subject_name(subject: str) -> str:
    return subject.replace("_", " ")


def render_prompt(row: dict) -> str:
    choices = list(row["choices"])
    if len(choices) != 4:
        raise ValueError(f"expected 4 choices, got {len(choices)}")
    return TEMPLATE.format(subject=subject_name(row["subject"]),
                           question=row["question"].strip(),
                           a=choices[0], b=choices[1], c=choices[2], d=choices[3])


def label_ids(tokenizer) -> list[int]:
    """The four label token ids, or a refusal.

    A tokenizer that splits " A" cannot host the paper's convention.  This raises
    instead of falling back to multi-token option scoring: the fallback would
    silently reintroduce the substitution tex:510 forbids, and it would look
    identical in the emitted files.
    """
    ids = []
    offenders = []
    for text in LABEL_STRINGS:
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) != 1:
            offenders.append((text, enc))
        else:
            ids.append(int(enc[0]))
    if offenders:
        raise SystemExit(
            f"this tokenizer does not encode {[t for t, _ in offenders]} as single "
            f"tokens (got {offenders}), so the paper's one-token label convention "
            f"cannot be applied. Refusing to fall back to multi-token option "
            f"pseudolikelihood, which Long_RWKV_ICLR2027.tex:510 forbids.")
    if len(set(ids)) != 4:
        raise SystemExit(
            f"the four labels collide onto {sorted(set(ids))}: the four choices "
            f"would not be distinguishable")
    return ids


def load_split(parquet_dir: Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq
    matches = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not matches:
        raise SystemExit(f"no {split}-*.parquet under {parquet_dir}")
    rows: list[dict] = []
    for path in matches:
        table = pq.read_table(path)
        rows.extend(table.to_pylist())
    return rows


def interleave_by_subject(rows: list[dict]) -> list[dict]:
    """Reorder rows round-robin across subjects, preserving order within each.

    ``cais/mmlu`` ships its splits **subject-grouped**: the test parquet's 14,042
    rows form exactly 57 contiguous runs, one per subject.  The consumer
    (``multichoice.list_npz_files``) sorts by filename and ``--max_samples`` takes a
    *prefix*, so writing files in source order would make any subset a few whole
    subjects rather than a sample of MMLU -- a 200-item prefix would cover two
    subjects out of 57.  That is a different statistic with the same name, and it
    would be invisible in the score.

    Interleaving makes every prefix stratified by construction, so the dev-grid
    subsets and the full run measure the same thing.  Round-robin sharding is
    unaffected either way (it already strides across the whole list), so this costs
    nothing and removes the hazard for every future caller.
    """
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    order = sorted(by_subject)
    out: list[dict] = []
    for i in range(max(len(v) for v in by_subject.values())):
        for subject in order:
            bucket = by_subject[subject]
            if i < len(bucket):
                out.append(bucket[i])
    return out


def build(rows: list[dict], tokenizer, out_dir: Path,
          max_prompt_tokens: int = 2048, interleave: bool = True) -> dict:
    """Write one ``.npz`` per example; return the manifest.

    A prompt that exceeds ``max_prompt_tokens`` is recorded as ``skipped_too_long``
    rather than truncated: truncating a multiple-choice question can remove the
    option the answer names, which silently converts a hard item into an
    unanswerable one and shows up as a lower score rather than as a missing item.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    ids = label_ids(tokenizer)
    if interleave:
        rows = interleave_by_subject(rows)
    written, skipped = 0, []
    answer_hist = {c: 0 for c in LABELS}
    lengths = []
    index_rows: list[dict] = []
    for index, row in enumerate(rows):
        prompt = render_prompt(row)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_tokens:
            skipped.append({"index": index, "subject": row["subject"],
                            "prompt_tokens": len(prompt_ids),
                            "reason": "skipped_too_long"})
            continue
        length = len(prompt_ids) + 1
        input_ids = np.empty((4, length), dtype=np.int32)
        for j, label_id in enumerate(ids):
            input_ids[j, :-1] = prompt_ids
            input_ids[j, -1] = label_id
        # Every row is the same length and fully valid: one shared prompt plus one
        # label token, so there is nothing to pad and the mask is all-ones.
        npz = {
            "input_ids": input_ids,
            "attention_mask": np.ones((4, length), dtype=bool),
            # length-1 span: _span_nll scores logits[:, L-2] against the label.
            "choice_start": np.full((4,), length - 1, dtype=np.int32),
            "label": np.asarray([int(row["answer"])], dtype=np.int32),
        }
        np.savez(out_dir / f"{written:06d}.npz", **npz)
        # The file name is the only handle a per-shard result JSON carries, so the
        # mapping back to subject/answer is written out too. Without it a
        # per-subject breakdown cannot be recovered from the records at all.
        index_rows.append({"file": f"{written:06d}.npz", "subject": row["subject"],
                           "answer": LABELS[int(row["answer"])],
                           "prompt_tokens": len(prompt_ids)})
        answer_hist[LABELS[int(row["answer"])]] += 1
        lengths.append(length)
        written += 1

    (out_dir / "index.json").write_text(
        json.dumps(index_rows, indent=1) + "\n", encoding="utf-8")
    subjects = sorted({r["subject"] for r in index_rows})

    return {
        "written": written,
        "skipped": len(skipped),
        "skipped_detail": skipped[:50],
        "answer_distribution": answer_hist,
        "label_token_ids": dict(zip(LABEL_STRINGS, ids)),
        "subjects": len(subjects),
        "interleaved_by_subject": interleave,
        "prefix_is_stratified": interleave,
        "first_20_subjects": [r["subject"] for r in index_rows[:20]],
        "prompt_tokens_min": min(lengths) - 1 if lengths else None,
        "prompt_tokens_max": max(lengths) - 1 if lengths else None,
        "max_prompt_tokens": max_prompt_tokens,
        "template_sha256": template_sha256(),
        "convention": "one_token_answer_label",
        "scoring_note": (
            "choice_start = L-1, so eval/capability/multichoice.py's _span_nll "
            "scores exactly one target token per choice. Use --conditions fwdce "
            "for the causal (shifted) protocol; maskce is the denoiser-native "
            "variant and is a DIFFERENT statistic, not a substitute."
        ),
        "forbidden_alternative": (
            "research/DiffRwkv/preprocessed_data/mmlu/ scores full option text "
            "(spans of 1-44 tokens) and is multi-token option pseudolikelihood, "
            "which tex:510 rules out as a substitute for this protocol."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", type=Path, required=True,
                    help="dir holding cais/mmlu <split>-*.parquet")
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="HF dir supplying the tokenizer to verify and encode with")
    ap.add_argument("--split", default="test", choices=("test", "validation", "dev"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--no-interleave", action="store_true",
                    help="keep the source (subject-grouped) order; makes any "
                         "--max_samples prefix a few whole subjects, not a sample")
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir),
                                              trust_remote_code=True)
    rows = load_split(args.parquet_dir, args.split)
    if args.limit:
        # Interleave FIRST, then cut: slicing the subject-grouped source order
        # would hand `build` a couple of whole subjects, and interleaving those is
        # still a couple of whole subjects. The cut has to happen on the stratified
        # order for a limit to mean "a sample of MMLU".
        if not args.no_interleave:
            rows = interleave_by_subject(rows)
        rows = rows[:args.limit]
    manifest = build(rows, tokenizer, args.out_dir,
                     max_prompt_tokens=args.max_prompt_tokens,
                     interleave=not args.no_interleave)
    manifest.update({"split": args.split, "source_rows": len(rows),
                     "parquet_dir": str(args.parquet_dir),
                     "model_dir": str(args.model_dir),
                     "out_dir": str(args.out_dir),
                     "tokenizer_vocab_size": int(tokenizer.vocab_size)})
    print(json.dumps(manifest, indent=2))
    target = args.manifest or (args.out_dir / "manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
