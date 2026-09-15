"""One real gradient step per canvas for the bounded throughput calibration.

The loss and corruption functions come from the *pinned* trainer snapshot (``SOURCE_SHA256``),
extracted by ``trainer_semantics.load_functions`` — the same path the accepted semantics-v8 run
exercised. Reusing that path is deliberate: it means the measured gradient step is the project's
actual masked-diffusion objective rather than a stand-in cross-entropy.

What this module deliberately does **not** claim:

* The optimizer step is timed with a plain ``torch.optim.AdamW`` over the model's parameters.
  ``trainer_semantics.build_optimizer`` refuses any model above 100k CPU parameters, so it cannot
  build an optimizer for a real arm. The measurement therefore matches the trainer's optimizer
  *class*, not its parameter-group or freeze policy, and says so in the record.
* No optimizer step is timed for the 4.09B model on one GPU: that is not the training
  configuration (training shards across 32 GPUs), so the record carries a reason instead of a
  number that would misrepresent the cost.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Final, Protocol

MASK_TOKEN_ID: Final = 65_535
TRAIN_BLOCK_SIZE: Final = 256
CORRUPTION_SEED: Final = 20_260_915

OPTIMIZER_NOT_MEASURED_SHARDED: Final = (
    "single-GPU AdamW state for 4.09B parameters is not the training configuration "
    "(training sharded across 32 GPUs), so no optimizer step was timed"
)
OPTIMIZER_NOT_MEASURED_SCHEDULED_OFF: Final = (
    "optimizer-step timing was not scheduled for this arm"
)
OPTIMIZER_MEASUREMENT_NOTE: Final = (
    "measured with a plain torch.optim.AdamW over all parameters; matches the trainer's optimizer "
    "class but not its parameter-group or freeze policy"
)
WARMUP_NOTE: Final = (
    "each rung's first forward is an untimed warm-up and the allocator cache is released between "
    "rungs, so the reported forward values exclude first-touch kernels and cross-rung accumulation"
)
ALLOC_CONF_NOTE: Final = (
    "the job runs with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; without it the dplr chunk "
    "kernels fragment the 79 GiB device and a rung that fits alone fails"
)
CONTENDED_NODE_NOTE: Final = (
    "rows were measured while this job's other ranks were resident on the same node; treat them as "
    "contended-node throughput, not solo-GPU peak"
)
EMPTY_CORRUPTION_MASK: Final = "sample_corruption selected no positions on this canvas"
GRADIENT_STEP_REPETITIONS: Final = 3


class TrainableTorch(Protocol):
    """Only the Torch surface this module uses, so it stays importable without a GPU."""

    def enable_grad(self): ...
    def randint(self, low: int, high: int, size: tuple[int, int], *, device): ...
    def ones_like(self, input, *, dtype): ...
    def tensor(self, data, *, device): ...
    def Generator(self, *, device): ...


class TrainableModel(Protocol):
    def train(self) -> None: ...
    def eval(self) -> None: ...
    def parameters(self) -> Iterator: ...
    def __call__(self, input_ids, *, force_forward: bool, z_slots, state_cache, use_cache): ...


def _no_synchronize() -> None:
    """CPU default: there is no device queue to drain."""


def _canvas_batch(torch_module: TrainableTorch, device, canvas_tokens: int):
    """A non-pad token canvas with a matching all-attention mask."""
    ids = torch_module.randint(1, MASK_TOKEN_ID, (1, canvas_tokens), device=device)
    attention = torch_module.ones_like(ids, dtype=bool)
    return ids, attention


def _doc_spans(torch_module: TrainableTorch, device, canvas_tokens: int):
    """One document per row whose masked completion span is the middle third of the canvas.

    The pinned corruption lane takes ``(doc_starts, prompt_ends, doc_ends)``, each ``[B, D]``
    int64, and masks ``prompt_ends[k] .. doc_ends[k]`` for one chosen document. Setting the
    document start equal to the prompt end masks exactly ``[start, stop)``; every position stays
    eligible because the canvas uses non-pad token ids and an all-true attention mask.
    """
    start = max(1, canvas_tokens // 3)
    stop = min(canvas_tokens, start + max(1, canvas_tokens // 3))
    if stop <= start:
        stop = start + 1
    doc_starts = torch_module.tensor([[start]], device=device)
    prompt_ends = torch_module.tensor([[start]], device=device)
    doc_ends = torch_module.tensor([[stop]], device=device)
    return (doc_starts, prompt_ends, doc_ends)


def build_gradient_step(
    model: TrainableModel,
    functions,
    torch_module: TrainableTorch,
    device,
    *,
    loop_reps_override: int | None = None,
    force_forward: bool = False,
    optimizer: object | None = None,
    optimizer_reason: str | None = None,
    synchronize: Callable[[], None] | None = None,
    observer: Callable[[int], None] | None = None,
) -> tuple[Callable[[int], float], Callable[[int], float] | None, str | None]:
    """Return ``(backward_step, optimizer_step, optimizer_not_measured_reason)``.

    ``backward_step`` times one real loss backward on a masked canvas. ``optimizer_step`` is
    ``None`` when no optimizer was supplied, in which case the third element explains why rather
    than leaving a zero that a reader could mistake for a fast step.

    ``synchronize`` exists because device work is asynchronous: on CUDA it must be
    ``torch.cuda.synchronize`` or every timing here would measure kernel-launch overhead only.
    It defaults to a no-op so the CPU path needs no CUDA, and a CPU timing is therefore never
    comparable to a device timing.
    """
    sync = synchronize if synchronize is not None else _no_synchronize

    def _forward_kwargs() -> dict[str, object]:
        kwargs: dict[str, object] = {
            "force_forward": force_forward,
            "z_slots": None,
            "state_cache": None,
            "use_cache": False,
        }
        if loop_reps_override is not None:
            kwargs["loop_reps_override"] = loop_reps_override
        return kwargs

    def _corrupted_inputs(canvas_tokens: int):
        ids, attention = _canvas_batch(torch_module, device, canvas_tokens)
        generator = torch_module.Generator(device=device)
        generator.manual_seed(CORRUPTION_SEED)
        corrupted, mask, bucket, _, _ = functions.sample_corruption(
            ids,
            attention,
            min(TRAIN_BLOCK_SIZE, canvas_tokens),
            0.0,
            generator,
            gen_prob=0.0,
            docgen_prob=1.0,
            doc_spans=_doc_spans(torch_module, device, canvas_tokens),
        )
        return ids, corrupted, mask, bucket

    def backward_step(canvas_tokens: int) -> float:
        ids, corrupted, mask, bucket = _corrupted_inputs(canvas_tokens)
        selected = int(mask.sum().item())
        if selected <= 0:
            raise ValueError(EMPTY_CORRUPTION_MASK)
        if observer is not None:
            observer(selected)
        model.train()
        try:
            for parameter in model.parameters():
                parameter.grad = None
            with torch_module.enable_grad():
                logits = model(corrupted, **_forward_kwargs())
                loss, _ = functions.masked_diffusion_loss(
                    logits, ids, mask, bucket, min(TRAIN_BLOCK_SIZE, canvas_tokens)
                )
                sync()
                started = time.perf_counter()
                loss.backward()
                sync()
                elapsed = time.perf_counter() - started
        finally:
            # Gradients are captured by now; leaving train mode on would make the next canvas's
            # forward timing a train-mode timing, and the two would not be comparable.
            model.eval()
        if elapsed <= 0:
            raise ValueError("backward_step_not_positive")
        return float(elapsed)

    if optimizer is None:
        return backward_step, None, optimizer_reason or OPTIMIZER_NOT_MEASURED_SCHEDULED_OFF

    def optimizer_step(canvas_tokens: int) -> float:
        """Time one AdamW step; the caller's backward ran first, so gradients already exist."""
        del canvas_tokens
        sync()
        started = time.perf_counter()
        optimizer.step()
        sync()
        elapsed = time.perf_counter() - started
        if elapsed <= 0:
            raise ValueError("optimizer_step_not_positive")
        return float(elapsed)

    return backward_step, optimizer_step, None


def build_adamw(model: TrainableModel, torch_module, *, learning_rate: float = 1e-5):
    """A plain AdamW over every parameter; see the module docstring for what that does not match."""
    import torch  # local import keeps this module importable without torch

    del torch_module
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
