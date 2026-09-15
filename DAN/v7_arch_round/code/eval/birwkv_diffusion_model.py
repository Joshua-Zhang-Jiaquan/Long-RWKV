"""Eval adapter for the BiRWKV token-diffusion denoiser.

Loads a flat-state-dict checkpoint (written by train_birwkv_diffusion.py) into
BiRWKV7ForMaskedDiffusion and exposes the duck-typed ``generate(prompt, **kw)
-> str`` interface that eval_code/eval_math call — the denoiser's own lm_head
produces the tokens via the iterative confidence-commit sampler. No frozen
renderer anywhere in the path.

Requires CUDA (fla kernels). ``model_dir`` supplies the HF geometry + tokenizer
(config.json / tokenizer files); ``ckpt_dir`` supplies model.pt + meta.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LoadedBiRWKVDiffusion:
    model: object  # BiRWKV7ForMaskedDiffusion
    tokenizer: object
    step: int
    ckpt_dir: str
    device: str = "cuda"
    dtype: object = torch.bfloat16


def load_birwkv_diffusion(
    ckpt_dir: str,
    model_dir: str,
    device: str = "cuda",
    latent_cond: dict[str, object] | None = None,
    block_size: int = 0,
) -> LoadedBiRWKVDiffusion:
    """Build geometry from the HF dir, then load the flat training state dict.

    ``latent_cond`` optionally attaches a Gate-D0 conditioning arm **before** the
    state dict is applied, e.g.
    ``{"kind": "film", "latent_dim": 32, "num_prefix": 8, "out_scale": 0.5}``.
    It must be attached first: a latent-trained checkpoint carries
    ``latent_cond.*`` tensors, and a bare model would report them as *unexpected*
    and trip the mismatch guard below. Attaching afterwards is not an option
    either — the arm's trained weights would be silently discarded and every
    conditioned metric would be measuring a randomly-initialised adapter.
    """
    from transformers import AutoTokenizer

    from eval.capability.load_gate import load_slot
    from models.birwkv7_diffusion import BiRWKV7ForMaskedDiffusion

    ckpt_path = Path(ckpt_dir)
    # Gated: 8 shards loading a 4.1B denoiser at once exhaust the pod's host RAM
    # and wedge every process with no traceback (see eval/capability/load_gate.py).
    # The gate is only needed where 8 x per-proc peak exceeds the cgroup (H100 eval
    # pods: 393 GB needed vs 300 GB). Set BIRWKV_LOAD_SLOTS=8 on big-RAM pods
    # (H200: 1.93 TB cgroup) to load fully in parallel — load is ~145 s and dwarfs
    # the ~48 s of actual scoring, so serializing it is the main throughput cost.
    with load_slot(label="birwkv-load"):
        model = BiRWKV7ForMaskedDiffusion.from_hf_pretrained(model_dir, dtype=torch.bfloat16)
        if latent_cond is not None:
            _ = model.attach_latent_conditioner(
                str(latent_cond["kind"]),
                latent_dim=int(latent_cond.get("latent_dim", 32)),  # type: ignore[arg-type]
                num_prefix=int(latent_cond.get("num_prefix", 8)),  # type: ignore[arg-type]
                out_scale=float(latent_cond.get("out_scale", 0.5)),  # type: ignore[arg-type]
            )
            if model.latent_cond is not None:
                model.latent_cond = model.latent_cond.to(torch.bfloat16)
        # Cast tensor-by-tensor and drop each fp32 source immediately. The old
        # dict comprehension materialized the whole bf16 copy while the fp32 dict
        # was still alive (16.4 + 8.2 GB); this keeps only one fp32 tensor extra.
        state = torch.load(ckpt_path / "model.pt", map_location="cpu", weights_only=True)
        # A U3 checkpoint (v5.2 B3) carries block_t_cond.* keys that the freshly built
        # model lacks; without attaching the arm first, the load's `unexpected` filter
        # below raises and the job dies before its first cell. This killed gate (ii)
        # attempt 1 (job-74707291) with an empty output dir -- the fourth defect of the
        # warm-start class, and the second one on a path the trainer's fix never covered.
        if any(k.startswith("block_t_cond.") for k in state):
            if block_size <= 0:
                # The block size is a training-time contract (arm B used 64); guessing it
                # would silently misalign every block's t with its positions, which no
                # downstream metric would reveal. Refuse instead.
                msg = ("checkpoint has block_t_cond.* keys; pass block_size=<training "
                       "block size> so the eval attaches the timestep arm correctly")
                raise ValueError(msg)
            _ = model.attach_block_timestep_conditioner(default_block_size=block_size)
        # v7 W-A0: architecture-mechanism checkpoints carry residual_streams.* /
        # loop.* keys; sniff + attach BEFORE the load so they map onto the live
        # module (the same reason block_t_cond must attach first — the shape of
        # the stored tensors recovers the config: m_res_raw [L,n,n] gives n_streams,
        # gates_raw gives trained reps, lo/hi give the loop range).
        if any(k.startswith("residual_streams.") for k in state):
            ns_v = int(state["residual_streams.m_res_raw"].shape[1])
            _ = model.attach_residual_streams(n_streams=ns_v)
            model.residual_streams = model.residual_streams.to(torch.bfloat16)
        if any(k.startswith("loop.") for k in state):
            lo_v = int(state["loop.lo"])
            hi_v = int(state["loop.hi"])
            reps_v = int(state["loop.gates_raw"].shape[0])
            _ = model.attach_backbone_loop((lo_v, hi_v), reps_v)
            model.loop = model.loop.to(torch.bfloat16)
            # W-A5 depth extrapolation: eval-time loop reps beyond the trained
            # depth. Env knob keeps every existing caller unchanged; the model
            # forward indexes gates[min(p, ...)], so extra passes reuse the
            # last trained gate by construction.
            reps_env = os.environ.get("LOOP_REPS_OVERRIDE", "")
            if reps_env:
                reps_o = int(reps_env)
                if reps_o < 1:
                    msg = f"LOOP_REPS_OVERRIDE must be >=1, got {reps_o}"
                    raise ValueError(msg)
                orig_forward = model.forward

                def _loop_reps_forward(*args, _f=orig_forward, _r=reps_o, **kwargs):
                    kwargs.setdefault("loop_reps_override", _r)
                    return _f(*args, **kwargs)

                model.forward = _loop_reps_forward  # type: ignore[method-assign]
        for k in list(state.keys()):
            state[k] = state[k].to(torch.bfloat16)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad = [k for k in missing if "fuse_" not in k] + list(unexpected)
        if bad:
            raise RuntimeError(f"checkpoint/model mismatch: {bad[:8]}")
        del state  # release the bf16 copy before the next shard takes the slot
        model = model.to(device).eval()

    step = 0
    meta = ckpt_path / "meta.json"
    if meta.exists():
        step = int(json.loads(meta.read_text()).get("step", 0))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return LoadedBiRWKVDiffusion(model=model, tokenizer=tokenizer, step=step,
                                 ckpt_dir=str(ckpt_path.resolve()), device=device)


class BiRWKVDiffusionGenerator:
    """Duck-typed generator: prompt -> masked continuation -> iterative denoise."""

    # The masked-diffusion training corpora (capability mixture + code mixture)
    # are SFT/agent-trajectory formatted with plain-text "Assistant:" turn
    # markers. Bare-code prompts are out-of-distribution and the sampler
    # commits degenerate chat-marker tokens ("</istant>...") first — verified
    # by completion dumps 2026-08-12. Wrapping the prompt in the training
    # frame + opening a fenced code block puts generation in-distribution;
    # extract_code() prefers fenced blocks, and "```" is the stop.
    CHAT_PREFIX = "User: {prompt}\n\nAssistant:\n```python\n"
    # Same SFT turn frame without the code fence -- for NON-code tasks (CoLA-style
    # QA/generate-then-match). The fence is task-specific scaffolding for
    # extract_code(), not part of the training distribution.
    CHAT_PREFIX_PLAIN = "User: {prompt}\n\nAssistant:"

    def __init__(
        self,
        loaded: LoadedBiRWKVDiffusion,
        steps: int = 16,
        seed: int = 42,
        self_correction: bool = False,
        block_round: int = 32,
        chat_frame: bool = True,
        code_fence: bool = True,
        commit_group_size: int = 0,
    ):
        # self_correction defaults OFF: the offline sampler grid (2026-08-11)
        # showed remask_threshold=0.25 reopens most CORRECT commits (typical
        # true-token prob << 0.25 at this checkpoint's entropy), collapsing em
        # by 4-6x vs single-shot. Re-enable only with a recalibrated threshold.
        self.loaded = loaded
        self.steps = steps
        self.seed = seed
        self.self_correction = self_correction
        self.block_round = block_round
        self.chat_frame = chat_frame
        # code_fence only matters when chat_frame is on: True = code tasks (python
        # fence + re-wrap), False = plain turn frame (QA/cloze tasks).
        self.code_fence = code_fence
        # commit_group_size=0 keeps the legacy single-forward multi-commit path, so every
        # published generation number is reproduced exactly. g>=1 bounds commits per
        # forward and refreshes the canvas between groups (v5.2 B2).
        #
        # This is the path where the degeneracy actually lives: FREE GENERATION reached
        # max_run/len 0.298-0.874, while the Phase-1a infilling sweep found only
        # 0.0021-0.0345 at g=0 -- an order of magnitude milder. Grouping cut infilling
        # repetition 2-5x with no em cost, but that says nothing about this regime, so
        # the sweep has to be run here to make any claim about the code failure.
        self.commit_group_size = commit_group_size

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        stop_strings: tuple[str, ...] | None = None,
    ) -> str:
        # top_k/top_p/repetition_penalty accepted for interface parity; the
        # confidence-commit sampler uses temperature only.
        del top_k, top_p, repetition_penalty
        from models.birwkv7_diffusion import MASK_TOKEN_ID, iterative_denoise

        model = self.loaded.model
        device = next(iter(model.parameters())).device
        torch.manual_seed(self.seed)

        if self.chat_frame:
            prefix_t = self.CHAT_PREFIX if self.code_fence else self.CHAT_PREFIX_PLAIN
            framed = prefix_t.format(prompt=prompt)
        else:
            framed = prompt
        prompt_ids = self.loaded.tokenizer(framed, return_tensors=None)["input_ids"]
        gen_len = min(int(max_new_tokens), 512)
        gen_len = ((gen_len + self.block_round - 1) // self.block_round) * self.block_round

        ids = torch.tensor([prompt_ids + [MASK_TOKEN_ID] * gen_len], device=device)
        masked = torch.zeros_like(ids, dtype=torch.bool)
        masked[:, len(prompt_ids):] = True

        denoised, _ = iterative_denoise(
            model, ids, masked,
            steps=self.steps,
            temperature=temperature,
            self_correction=self.self_correction,
            commit_group_size=self.commit_group_size,
        )
        out_ids = denoised[0, len(prompt_ids):].tolist()
        # The denoiser fills the whole masked canvas, so trailing noise must be cut.
        #
        # THIS WAS THE ROOT CAUSE OF pass@1 == 0.0. The previous code used
        # ``eos_id = 65530`` and called it EOS, but the RWKV World tokenizer decodes
        # 65530 to ``"\n\n"`` -- an ordinary BLANK LINE. The real end-of-text is
        # 65531 (and 0 also decodes to it). Since CHAT_PREFIX ends with
        # "```python\n", a generation that opens with a blank line was truncated to
        # NOTHING, leaving only the re-wrapped fence. Every completion came out
        # empty, ``prompt + ""`` is a docstring-only body, and ast.parse reported
        # syntax_error -- on 164/164 tasks at every step count, across three training
        # rounds that could never have fixed a decode-side string bug.
        #
        # Verified against the tokenizer rather than trusting the comment:
        #   65530 -> '\n\n'   65531 -> '<|rwkv_tokenizer_end_of_text|>'
        # A comment asserting a constant's meaning is not verification.
        eot_ids = {65531, 0}
        tok_eos = getattr(self.loaded.tokenizer, "eos_token_id", None)
        if isinstance(tok_eos, int):
            eot_ids.add(tok_eos)
        cut = len(out_ids)
        for i, tid in enumerate(out_ids):
            if tid in eot_ids:
                cut = i
                break
        out_ids = out_ids[:cut]
        text = self.loaded.tokenizer.decode(out_ids)
        # Stop-strings MUST be applied to the RAW body, BEFORE the fence re-wrap.
        #
        # THIS WAS THE PROXIMATE CAUSE OF pass@1 == 0.0. The old order re-wrapped
        # first -- text = "```python\n" + body + "\n```" -- which puts a newline at
        # index 9. HumanEval's stop strings are ("\ndef ", "\nclass ",
        # "\nif __name__"), so a body that opens by restating the signature (`def
        # ...`, the single most likely start for a function completion) made
        # "\ndef " match AT INDEX 9, truncating the whole thing to the 9 characters
        # "```python". The dumps confirm it byte-for-byte: raw completion was
        # '\n```python\n' with NO closing fence, i.e. the cut fired.
        #
        # Applying the stops to the body first means they can only match real
        # boundaries the model emitted, never the synthetic fence we added.
        if stop_strings:
            cut = len(text)
            for s in stop_strings:
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]
        if self.chat_frame and self.code_fence:
            # generation opened inside a ```python fence: close at the fence
            # and re-wrap so extract_code()'s fenced-block path fires.
            fence = text.find("```")
            body = text[:fence] if fence != -1 else text
            text = "```python\n" + body + "\n```"
        return text
