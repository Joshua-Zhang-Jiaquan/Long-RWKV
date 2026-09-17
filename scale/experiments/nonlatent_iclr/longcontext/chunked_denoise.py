"""Chunked reduction of the denoiser's commit decision over a 64K canvas.

The masked-diffusion sampler does not need the full token distribution.  For
every masked position it needs exactly two numbers: the argmax token id, and the
max probability that decides whether that position is committed this step.  The
unchunked path nevertheless materializes the whole logits tensor *and* a
probability tensor of the same shape, because it calls ``softmax`` on
``[1, T, V]`` before reducing:

    logits : 1 * T * V * 4 bytes
    probs  : 1 * T * V * 4 bytes

At the frozen 64K canvas (``T = 65536``, ``V = 65536``) each is
``65536 * 65536 * 4 = 17_179_869_184`` bytes (16 GiB), so the pair is
``34_359_738_368`` bytes -- 32 GiB, the 34.4 GB the plan quotes in decimal units
-- before any activation.  That allocation, not the compute, is the wall that
kept every measurement in this program at or below 4096 tokens.

Chunking over positions and reducing each chunk to its ``(argmax id, max
probability)`` vectors leaves ``chunk * V * 4`` bytes live per tensor at any
moment, and never the whole canvas.  The reduction is lossless FOR THE COMMIT
DECISION: the sampler commits on the argmax id and on the confidence alone, and
both are invariant under processing the vocab axis one chunk of positions at a
time.  Nothing here claims the full *distribution* survives -- a caller that
needs more than the commit decision must not use this path, and the guard
``assert_no_full_logits`` exists so a regression that quietly materializes the
whole canvas fails loudly instead of passing an equivalence test.

Pure torch CPU; no CUDA calls, so this runs in the CPU suite.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch

#: The frozen 64K canvas length (``DAN/nonlatent_iclr/task_registry.json`` LENGTHS
#: top cell, read in RWKV tokens).  Named so the arithmetic below is not a magic
#: number -- and named after the file that actually holds it, so the provenance
#: the number claims can be checked.
CANVAS_TOKENS: Final = 65536

#: The model vocabulary width; logits are ``[1, T, MODEL_VOCAB_SIZE]``.
MODEL_VOCAB_SIZE: Final = 65536

#: fp32 element width.  The denoiser computes in fp32 even when weights are bf16,
#: which is what makes the workspace below as large as it is.
FP32_BYTES: Final = 4

#: Bytes of ONE unchunked ``[1, 65536, 65536]`` fp32 logits tensor:
#: ``65536 * 65536 * 4 = 17_179_869_184`` (16 GiB).
FULL_LOGITS_BYTES: Final = CANVAS_TOKENS * MODEL_VOCAB_SIZE * FP32_BYTES

#: Bytes of the logits+probability PAIR the unchunked path holds live at once:
#: ``34_359_738_368`` -- 32 GiB.  This is the number the plan reports as 34.4 GB.
FULL_DENOISE_BYTES: Final = 2 * FULL_LOGITS_BYTES

#: A logits provider: ``provider(start, end)`` returns ``[1, end-start, vocab]``.
LogitsProvider = Callable[[int, int], torch.Tensor]


class ChunkRefusal(ValueError):
    """A chunked denoise call cannot honour the memory contract it promises.

    Raised instead of returning a partial or silently mis-shaped result.  The
    failure it prevents is the one that would make every other guarantee here
    meaningless: a caller asks for a bounded workspace, the path instead
    materializes the full ``[1, T, V]`` tensor, and nothing downstream can tell
    the difference because the numbers it returns are still correct.  A refusal
    is the only signal that survives that.
    """


def _require_positive(value: int, name: str) -> None:
    """A zero or negative span cannot cover a canvas; refuse rather than loop forever."""
    if value <= 0:
        msg = (
            f"{name} must be positive, got {value}; a chunked reduction over a "
            f"zero- or negative-length range is either empty or ill-defined, and "
            f"the commit decision it is meant to produce would be missing positions"
        )
        raise ChunkRefusal(msg)


def chunk_denoise_bytes(chunk: int, vocab: int, dtype_bytes: int = FP32_BYTES) -> int:
    """Live bytes of the logits+probability pair for ONE chunk of positions.

    This is the workspace the chunked path actually needs, and it is what the
    docstring's arithmetic turns on: at ``chunk = 4096`` and ``vocab = 65536`` it
    is ``2 * 4096 * 65536 * 4 = 2_147_483_648`` bytes (2 GiB), independent of how
    long the canvas is.
    """
    _require_positive(chunk, "chunk")
    _require_positive(vocab, "vocab")
    return 2 * chunk * vocab * dtype_bytes


def full_denoise_bytes(total_tokens: int, vocab: int, dtype_bytes: int = FP32_BYTES) -> int:
    """Live bytes of the unchunked logits+probability pair.

    Kept as a function so the size of the allocation that motivates chunking is
    reproducible from the library rather than only quoted in prose; the module
    constants pin the 64K canvas instance of it.
    """
    _require_positive(total_tokens, "total_tokens")
    _require_positive(vocab, "vocab")
    return 2 * total_tokens * vocab * dtype_bytes


def chunked_argmax_confidence(
    logits_provider: LogitsProvider, *, total_tokens: int, chunk: int
) -> tuple[list[int], list[float]]:
    """Argmax ids and max probabilities for the whole range, one chunk at a time.

    Returns ``(ids, confidences)``, both of length ``total_tokens``.  The
    provider is called on spans ``[0, chunk), [chunk, 2*chunk), ...`` -- each
    span at most ``chunk`` wide, the last possibly shorter -- and each chunk is
    reduced to its two vectors before the next call, so no more than one chunk
    of logits (and its chunk-sized softmax) is ever live.

    Lossless FOR THE COMMIT DECISION, and only for it: the argmax id and the max
    probability are exactly the quantities the sampler commits on, and both are
    computed per position from that position's own vocab row, so chunking the
    position axis cannot change them.  The full distribution is deliberately
    discarded; a caller that needs it must not call this.

    Memory arithmetic for the frozen 64K canvas, ``T = V = 65536``: the unchunked
    pair is ``2 * 65536 * 65536 * 4 = 34_359_738_368`` bytes (``FULL_DENOISE_BYTES``,
    32 GiB); this path's live workspace is ``chunk * V * 4`` per tensor
    (``chunk_denoise_bytes``), independent of ``total_tokens``.

    Raises ``ChunkRefusal`` for a non-positive ``total_tokens`` or ``chunk``, and
    for a provider that returns a chunk wider than the span it was asked for --
    the shape a "silently materializes everything" regression takes.
    """
    _require_positive(total_tokens, "total_tokens")
    _require_positive(chunk, "chunk")

    ids: list[int] = []
    confidences: list[float] = []
    vocab: int | None = None
    for start in range(0, total_tokens, chunk):
        end = min(start + chunk, total_tokens)
        span = end - start
        logits = logits_provider(start, end)
        if logits.dim() != 3 or int(logits.shape[0]) != 1:
            msg = (
                f"provider returned shape {tuple(logits.shape)} for span [{start}, "
                f"{end}); the contract is [1, {span}, vocab]"
            )
            raise ChunkRefusal(msg)
        width = int(logits.shape[1])
        if width != span:
            # A provider that ignores the span and hands back the whole canvas
            # would return `total_tokens` here and blow the workspace budget
            # while still producing correct ids -- exactly the regression that
            # would slip past an equivalence test, so it is refused on shape.
            msg = (
                f"provider returned {width} positions for the {span}-token span "
                f"[{start}, {end}); a result wider than the request is the "
                f"full-canvas allocation this path exists to avoid"
            )
            raise ChunkRefusal(msg)
        if vocab is None:
            vocab = int(logits.shape[2])
            if vocab <= 0:
                msg = f"provider returned a zero-width vocabulary in span [{start}, {end})"
                raise ChunkRefusal(msg)
        elif int(logits.shape[2]) != vocab:
            msg = (
                f"provider vocabulary changed mid-run: {int(logits.shape[2])} at "
                f"span [{start}, {end}) vs {vocab} earlier; the argmax ids would "
                f"not be comparable across chunks"
            )
            raise ChunkRefusal(msg)
        # Softmax over the vocab axis only, for this chunk of positions.  The
        # fp32 cast mirrors the unchunked path, so the confidence is the same
        # value it would have produced, not an approximation of it.
        probabilities = torch.softmax(logits.to(torch.float32), dim=-1)
        ids.extend(int(token) for token in probabilities.argmax(dim=-1).reshape(-1).tolist())
        confidences.extend(float(value) for value in probabilities.amax(dim=-1).reshape(-1).tolist())
    return ids, confidences


def assert_no_full_logits(
    logits_provider: LogitsProvider, *, total_tokens: int, vocab: int, chunk: int
) -> None:
    """Run the chunked path under a monitor and refuse any full-canvas request.

    The whole point of the chunked path is that no single call is ever asked for
    -- or ever returns -- more than ``chunk`` positions.  An implementation that
    quietly allocated the full ``[1, total_tokens, vocab]`` tensor would still
    return the correct ids and confidences, so an equivalence test cannot catch
    it; this guard is the instrument that can.  It wraps the provider, records
    every requested span, and raises ``ChunkRefusal`` if a span exceeds ``chunk``
    or if the provider returns a tensor wider than the span it was asked for
    (which is how a provider that materializes the full canvas hides).  It also
    confirms the spans cover ``total_tokens`` exactly, so a run that silently
    drops positions is refused too.

    Raises ``ChunkRefusal`` for the same non-positive arguments as
    ``chunked_argmax_confidence``, and for a ``vocab`` that disagrees with the
    provider's width.
    """
    _require_positive(total_tokens, "total_tokens")
    _require_positive(vocab, "vocab")
    _require_positive(chunk, "chunk")

    requested: list[tuple[int, int]] = []

    def guarded(start: int, end: int) -> torch.Tensor:
        span = end - start
        requested.append((start, end))
        if span > chunk:
            # The regression tripwire the task names: this fires only if the
            # caller above stops chunking, which is the failure the guard exists
            # to make loud.
            msg = (
                f"a provider call was asked for {span} positions, wider than the "
                f"chunk of {chunk}; the memory contract is one chunk at a time"
            )
            raise ChunkRefusal(msg)
        tensor = logits_provider(start, end)
        if tensor.dim() == 3:
            if int(tensor.shape[1]) != span:
                msg = (
                    f"provider returned {int(tensor.shape[1])} positions for the "
                    f"{span}-token span [{start}, {end}); it materialized more than "
                    f"it was asked for"
                )
                raise ChunkRefusal(msg)
            if int(tensor.shape[2]) != vocab:
                msg = (
                    f"provider width {int(tensor.shape[2])} disagrees with the "
                    f"declared vocab {vocab} at span [{start}, {end})"
                )
                raise ChunkRefusal(msg)
        return tensor

    chunked_argmax_confidence(guarded, total_tokens=total_tokens, chunk=chunk)
    covered = sum(end - start for start, end in requested)
    if covered != total_tokens:
        msg = (
            f"the recorded provider spans cover {covered} of {total_tokens} "
            f"positions; the commit decision would be missing positions"
        )
        raise ChunkRefusal(msg)
