"""Per-document reversal and the packed-sequence segment layout.

Two things live here, and both exist because the prior implementation got them
wrong in a way that trains silently:

1. ``build_seq_ctx`` turns ``(doc_starts, attention_mask)`` into the segment
   boundaries the RWKV-7 kernel needs.  Each document's content span is its own
   segment, and **each pad run is its own segment too** -- padding is not a
   document separator, so pad tokens must never share recurrence state with a
   real document in either direction.

2. ``segment_reverse_index`` reverses a packed row *document by document*.
   ``torch.flip(x, dims=(1,))`` -- what the earlier bidirectional model used --
   reverses the whole row, which reorders documents relative to each other and
   leaks one document's tail into the next one's head.

The reversal is an involution, so the identical map converts the backward
kernel's output back into the forward frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .refusal import Refusal


@dataclass(frozen=True)
class SeqCtx:
    """Segment layout for one packed batch, flattened to ``B=1``.

    ``cu`` is int32 on the activations' device (the Triton kernel reads it
    directly and cannot take a host pointer); ``cu_cpu`` is a Python-int view so
    bookkeeping never forces a device sync.
    """

    cu: torch.Tensor          # int32 [n_segments+1]
    cu_cpu: tuple[int, ...]   # the same boundaries as Python ints
    rev: torch.Tensor         # int64 [N], involution
    batch: int
    row_len: int

    @property
    def numel(self) -> int:
        return self.batch * self.row_len

    @property
    def n_segments(self) -> int:
        return len(self.cu_cpu) - 1


def build_seq_ctx(doc_starts: torch.Tensor,
                  attention_mask: torch.Tensor,
                  row_len: int | None = None) -> SeqCtx:
    """Build the segment layout for a packed batch.

    ``doc_starts[b]`` lists where each document begins.  Segments are cut at every
    live/dead transition **and** at every document start, so a document interrupted
    by padding cannot silently merge with its neighbour.

    The pack convention matters here.  Measured on ``dclm_4096_packed``: 1850 of
    3000 rows contain interior padding, because each document is padded up to the
    next 64-aligned start.  The live region is therefore **not** a contiguous
    prefix, and a layout rule that assumed one would refuse most of the corpus.
    Padding runs become segments of their own, so ``cu_seqlens`` still tiles the row
    exactly and no padding token ever shares recurrence state with a document.

    Raises ``Refusal`` on a malformed layout rather than producing a plausible wrong
    one: silently splitting or merging documents would change the recurrence
    boundaries of every position after the change.
    """
    if doc_starts.dim() != 2 or attention_mask.dim() != 2:
        raise Refusal("doc_starts and attention_mask must both be [B, T]")
    batch, width = attention_mask.shape
    if doc_starts.shape[0] != batch:
        raise Refusal(f"doc_starts has {doc_starts.shape[0]} rows, mask has {batch}")
    if row_len is not None and row_len != width:
        raise Refusal(f"row_len {row_len} disagrees with mask width {width}")

    live = attention_mask.bool()
    starts: list[int] = []
    ends: list[int] = []
    for b in range(batch):
        base = b * width
        row_live = live[b]
        if int(row_live.sum()) == 0:
            # An all-padding row is one padding segment; dropping it would leave a
            # hole in cu_seqlens.
            starts.append(base)
            ends.append(base + width)
            continue

        valid = sorted({int(v) for v in doc_starts[b].tolist() if int(v) >= 0})
        if valid and valid[0] != 0 and bool(row_live[0]):
            valid = [0, *valid]
        for value in valid:
            if not 0 <= value < width:
                raise Refusal(
                    f"row {b}: document start {value} is outside [0, {width})")
            if not bool(row_live[value]):
                # A start inside a padding run claims a document where the mask says
                # there is none.  Ignoring it would drop that document's content
                # without recording that anything was lost.
                raise Refusal(
                    f"row {b}: document start {value} falls in padding; the sidecar "
                    f"claims a document where the mask has none")

        # Break points: row edges, every live/dead transition, every document start.
        breaks = {0, width}
        transitions = torch.nonzero(row_live[1:] != row_live[:-1],
                                    as_tuple=False).flatten()
        for position in transitions.tolist():
            breaks.add(position + 1)
        breaks.update(valid)
        ordered = sorted(breaks)

        for left, right in zip(ordered, ordered[1:]):
            if left >= right:
                raise Refusal(f"row {b}: empty span at {left}")
            span_live = bool(row_live[left:right].all())
            span_dead = not bool(row_live[left:right].any())
            if not (span_live or span_dead):
                raise Refusal(
                    f"row {b}: span [{left}, {right}) mixes live and padding; the "
                    f"break points should have separated them")
            if span_dead:
                # Padding never shares state with a document, in either direction.
                starts.append(base + left)
                ends.append(base + right)
                continue
            # A live span belongs to exactly one document, which must start at its
            # left edge -- otherwise a document is split across a padding run.
            if left not in valid and left != 0:
                raise Refusal(
                    f"row {b}: live span [{left}, {right}) starts inside a document, "
                    f"so that document is interrupted by padding")
            starts.append(base + left)
            ends.append(base + right)

    if not starts:
        raise Refusal("no segments: the batch has no live tokens")

    total = batch * width
    order = sorted(range(len(starts)), key=lambda i: starts[i])
    cu_cpu = [0]
    for i in order:
        if starts[i] != cu_cpu[-1]:
            raise Refusal(
                f"segment {i} starts at {starts[i]} but the previous segment ends at "
                f"{cu_cpu[-1]}; segments must tile the row exactly")
        cu_cpu.append(ends[i])
    if cu_cpu[-1] != total:
        raise Refusal(f"segments cover {cu_cpu[-1]} positions, expected {total}")

    device = attention_mask.device
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device=device)
    rev = segment_reverse_index(cu, total)
    return SeqCtx(cu=cu, cu_cpu=tuple(cu_cpu), rev=rev, batch=batch, row_len=width)


def segment_reverse_index(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Index map reversing each segment in place, preserving segment order.

    Position ``p`` in segment ``[s, e)`` maps to ``s + e - 1 - p``.  The map is
    an involution, so the same call converts the kernel's output back.

    Copied from ``scale/scratch_1b/delta_rwkv.py:64-95``, which got two things
    right that a naive version gets wrong: the index is built on
    ``cu_seqlens``'s own device (indexing a tensor with a host index is the
    "found at least two devices" failure), and segment *order* is preserved so
    documents do not trade places.
    """
    if total <= 0:
        raise Refusal(f"row length must be positive, got {total}")
    if cu_seqlens.numel() < 2:
        raise Refusal("cu_seqlens must describe at least one segment")
    if int(cu_seqlens[-1]) != total:
        raise Refusal(
            f"cu_seqlens ends at {int(cu_seqlens[-1])}, not the row's {total} tokens")
    positions = torch.arange(total, dtype=torch.long, device=cu_seqlens.device)
    segment = torch.searchsorted(cu_seqlens, positions, right=True) - 1
    starts = cu_seqlens[segment].long()
    ends = cu_seqlens[segment + 1].long()
    return starts + ends - 1 - positions


def flatten_microbatch(input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       doc_starts: torch.Tensor,
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, SeqCtx]:
    """Fold a ``[B, T]`` microbatch into the ``[1, B*T]`` row the kernel requires.

    The varlen contract is ``B == 1`` with every row concatenated and the document
    boundaries carried by ``cu_seqlens``; a ``[B, T]`` batch has to be flattened
    before the model can see it, and the document starts have to be offset by
    ``row * T`` or row 1's documents claim row 0's positions.

    This exists because the alternative was silently taken.  ``train.py`` built a
    loader with ``batch_size=microbatch`` and then passed ``batch[...][0:1]`` --
    row 0 only -- to the model, because the unflattened batch is exactly what
    ``build_seq_ctx`` describes and the model refuses (``ctx describes 8x4096 but
    input is 1x32768``).  The run still trained, the loss still looked healthy,
    and the global-batch assertion still passed, because that assertion multiplies
    the *declared* microbatch.  At microbatch 8 that is one eighth of the tokens
    the record claims: a 3B-token run would have been a 375M-token run reporting
    3B.  Nothing downstream could have caught it -- the number the paper would
    report is computed from the config, not from what the loader served.

    Returns the flattened ids, mask and offset doc starts alongside the ``SeqCtx``
    for the flattened row, so the caller cannot build one and forget the other.
    """
    if input_ids.dim() != 2 or attention_mask.dim() != 2 or doc_starts.dim() != 2:
        raise Refusal(
            f"expected [B, T] ids/mask and [B, D] doc_starts, got "
            f"{tuple(input_ids.shape)}, {tuple(attention_mask.shape)}, "
            f"{tuple(doc_starts.shape)}")
    batch, width = input_ids.shape
    if attention_mask.shape != input_ids.shape:
        raise Refusal(
            f"mask {tuple(attention_mask.shape)} does not match ids "
            f"{tuple(input_ids.shape)}")
    if doc_starts.shape[0] != batch:
        raise Refusal(
            f"doc_starts has {doc_starts.shape[0]} rows, ids have {batch}")

    flat_ids = input_ids.reshape(1, batch * width)
    flat_mask = attention_mask.reshape(1, batch * width)

    # Offset each row's starts into the concatenated row, dropping the -1 padding
    # ``collate_packed`` uses for "no document here".  Offsetting a -1 would make
    # it a legitimate position and invent a document.
    offsets: list[int] = []
    for row in range(batch):
        base = row * width
        # Row 0's implicit start is added by build_seq_ctx for position 0 only, so
        # every later row needs its own start even when the pack omitted it.
        offsets.append(base)
        for value in doc_starts[row].tolist():
            value = int(value)
            if value < 0:
                continue
            if value >= width:
                raise Refusal(
                    f"row {row}: document start {value} is outside [0, {width})")
            if base + value not in offsets:
                offsets.append(base + value)
    flat_docs = torch.tensor([sorted(offsets)], dtype=torch.long,
                             device=input_ids.device)
    ctx = build_seq_ctx(flat_docs, flat_mask.bool(), row_len=batch * width)
    return flat_ids, flat_mask, flat_docs, ctx


def one_request_ctx(length: int, device: torch.device | None = None) -> SeqCtx:
    """A single unsegmented request, for evaluation prompts."""
    if length <= 0:
        raise Refusal(f"request length must be positive, got {length}")
    cu_cpu = (0, length)
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device=device)
    rev = segment_reverse_index(cu, length)
    return SeqCtx(cu=cu, cu_cpu=cu_cpu, rev=rev, batch=1, row_len=length)
