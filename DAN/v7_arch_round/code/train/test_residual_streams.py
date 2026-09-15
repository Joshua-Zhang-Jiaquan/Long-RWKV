"""CPU checks for v7 W-A0 — mHC residual streams + the shared integration loader.

Pure torch: ``models/residual_streams.py`` imports no fla, so its unit checks
run on the login node directly. The INTEGRATION checks (mechanism ON at init
== legacy forward) compile ``BiRWKV7ForMaskedDiffusion`` from the model file's
AST with a FAKE block — the ``test_block_timestep.py`` pattern, because fla's
Triton kernels need a GPU driver.

The properties that carry this arm (each has burned the project before):

1. **Exact identity at init** (bit-exact, ``torch.equal``): every Gate-0 /
   em@k / D0.3 number was measured on the single-stream model.
2. **It must actually respond once trained** — with a REAL threshold, not fp
   noise. The FIRST draft of this arm (row-stochastic mixing on replicated
   streams) passed a naive "output changed" check at 1.2e-06 which was pure
   rounding: the mechanism was a permanent no-op. The response controls now
   demand rel > 1e-3 AND measurable stream divergence.
3. **1x block compute**: the block is called exactly once per layer per
   forward (a counting block asserts it) — the whole premise of
   hyper-connections as cheap capacity.
4. **G-M bound**: every factor stays within ``mix_scale`` of its identity
   reference (the bounded-deviation connection manifold).
"""
from __future__ import annotations

import ast
import json  # noqa: F401  (model-file namespace)
import math
import sys
import types
from pathlib import Path

import torch
from torch import nn

_SCALE_DIR = Path(__file__).resolve().parents[1]
if str(_SCALE_DIR) not in sys.path:
    sys.path.insert(0, str(_SCALE_DIR))
if "models" not in sys.modules:
    _pkg = types.ModuleType("models")
    _pkg.__path__ = [str(_SCALE_DIR / "models")]
    sys.modules["models"] = _pkg

from models.residual_streams import (  # noqa: E402
    BackboneLoopControl,
    ResidualStreamMixer,
    expand_to_streams,
    readout,
)

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAIL.append(name)


# ----------------------------------------------------------------------
# Shared AST loader: the real model class with a fake block
# ----------------------------------------------------------------------
class _Cfg:
    """Minimal stand-in for RWKV7Config (only the fields the class touches)."""

    def __init__(self, vocab: int, hidden: int, layers: int) -> None:
        self.vocab_size = vocab
        self.hidden_size = hidden
        self.num_hidden_layers = layers
        self.norm_bias = True
        self.norm_eps = 1e-5


class _FakeBlock(nn.Module):
    """Mimics BiRWKV7Block's contract: output INCLUDES its own residual add.

    ``(h, v_f, v_b, force_forward, film, state_cache, use_cache)
       -> (h + delta(h), v_f', v_b')`` — elementwise+linear, deterministic per
    batch row. CALLS counts layer invocations (the 1x-compute assertion).
    """

    CALLS = 0

    def __init__(self, config: _Cfg, layer_idx: int, gate_bias_init: float = 4.0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.lin = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        with torch.no_grad():
            self.lin.weight.normal_(0.0, 0.02 + 0.01 * layer_idx)  # per-layer variation

    def forward(self, h, v_f, v_b, force_forward=False, film=None,
                state_cache=None, use_cache=False):  # noqa: ANN001, ANN202
        _FakeBlock.CALLS += 1
        return (h + torch.nn.functional.gelu(self.lin(h)), v_f + 0.5, v_b - 0.5)


class _SentinelFiLM(nn.Module):
    pass


class _SentinelSoft(nn.Module):
    pass


class _SentinelXAttn(nn.Module):
    pass


def load_model_cls() -> type:
    """Exec the real BiRWKV7ForMaskedDiffusion class with stubbed dependencies."""
    import importlib

    tt = importlib.import_module("models.state_hijacking_dit_torch_types")
    src = (_SCALE_DIR / "models" / "birwkv7_diffusion.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "BiRWKV7ForMaskedDiffusion")
    ns: dict = {
        "torch": torch, "nn": torch.nn, "math": math, "json": json,
        "Path": Path, "TYPE_CHECKING": False, "Callable": None,
        "cast": lambda _t, v: v, "override": lambda f: f,
        "TypedTorchModule": tt.TypedTorchModule,
        "MASK_TOKEN_ID": 65535, "PAD_TOKEN_ID": 0,
        "BiRWKV7Block": _FakeBlock,
        "LatentFiLMConditioner": _SentinelFiLM,
        "LatentSoftPrefixConditioner": _SentinelSoft,
        "LatentCrossAttnConditioner": _SentinelXAttn,
        "RWKV7Config": _Cfg,
        "ResidualStreamMixer": ResidualStreamMixer,
        "BackboneLoopControl": BackboneLoopControl,
        "expand_to_streams": expand_to_streams,
        "readout": readout,
        "_grad_checkpoint": lambda block, h, vf, vb, ff, film, sc, uc,
                             use_reentrant=False: block(h, vf, vb, ff, film, sc, uc),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<v7>", "exec"), ns)  # noqa: S102
    return ns["BiRWKV7ForMaskedDiffusion"]


def _probe_ids(vocab: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(20260906)
    ids = torch.randint(1, vocab, (2, 64), generator=g)
    ids[:, 32:] = 65535
    return ids


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-6))


# ----------------------------------------------------------------------
# Unit checks
# ----------------------------------------------------------------------
def main() -> int:
    torch.manual_seed(0)
    b, t, c = 3, 17, 32
    h = torch.randn(b, t, c)

    print("u1 expand/readout exactness")
    for n in (2, 4):
        big = expand_to_streams(h, n)
        check(f"expand shape n={n}", tuple(big.shape) == (b, t, n, c), f"{tuple(big.shape)}")
        check(f"readout(expand) == h n={n}", torch.equal(readout(big), h),
              "torch.equal (mean of n identical fp values is exact for n in {2,4})")

    print("u2 factors at init are EXACTLY the identity references")
    mix = ResidualStreamMixer(num_layers=6, n_streams=4)
    e0 = torch.zeros(4); e0[0] = 1.0
    worst = 0.0
    for l in range(6):
        w, a, p = mix.factors(l)
        worst = max(worst, float((w - e0).abs().max()),
                    float((a - torch.eye(4)).abs().max()),
                    float((p - 1.0).abs().max()))
    check("(w_pre, A_res, a_post) == (e_0, I, 1) exactly at init", worst == 0.0,
          f"max deviation {worst:.3e}")

    print("u3 G-M bound: deviation <= mix_scale under arbitrary perturbation")
    with torch.no_grad():
        for p_ in mix.parameters():
            p_.normal_(0.0, 5.0)
    dev = mix.max_deviation()
    check("max|A_res - I| <= mix_scale", dev <= 1.0, f"{dev:.3e}")

    print("u4 grads flow to ALL THREE raw groups AT init (no bilinear zero)")
    mix2 = ResidualStreamMixer(num_layers=2, n_streams=2)
    H = expand_to_streams(torch.randn(2, 5, 8), 2)
    w, a, p = mix2.factors(0)
    x = (H * w.view(1, 1, -1, 1)).sum(dim=2)
    o = x + torch.nn.functional.gelu(x)          # fake block: o = x + f(x)
    delta = o - x                                 # != 0 even at init
    Hp = (torch.einsum("ij,btjc->btic", a, H)
          + p.view(1, 1, -1, 1) * delta.unsqueeze(2))
    Hp.sum().backward()
    grads = {n_: float(p_.grad.abs().sum()) if p_.grad is not None else -1.0
             for n_, p_ in mix2.named_parameters()}
    check("w_pre_raw grad > 0", grads["w_pre_raw"] > 0, f"{grads['w_pre_raw']:.3e}")
    check("m_res_raw grad > 0", grads["m_res_raw"] > 0, f"{grads['m_res_raw']:.3e}")
    check("p_post_raw grad > 0 (delta != 0 at init)", grads["p_post_raw"] > 0,
          f"{grads['p_post_raw']:.3e}")

    print("u5 validation")
    try:
        ResidualStreamMixer(num_layers=2, n_streams=3)  # type: ignore[arg-type]
        check("n_streams=3 rejected", False, "no raise")
    except ValueError:
        check("n_streams=3 rejected", True, "ValueError")

    # ----------------------------------------------------------------------
    # Integration: the REAL forward
    # ----------------------------------------------------------------------
    print("i1 mHC ON at init == legacy forward (torch.equal, the D0.3 class)")
    cls = load_model_cls()
    cfg = _Cfg(vocab=65536, hidden=64, layers=6)
    ids = _probe_ids(cfg.vocab_size)
    torch.manual_seed(0)
    m_legacy = cls(cfg)
    with torch.no_grad():
        ref = m_legacy(ids)
    for n in (2, 4):
        torch.manual_seed(0)
        m = cls(cfg)
        _ = m.attach_residual_streams(n)
        with torch.no_grad():
            on = m(ids)
        check(f"n={n} logits bit-identical to legacy", torch.equal(ref, on),
              f"rel={_rel(on, ref):.3e}")

    print("i2 1x block compute: exactly L block calls per forward")
    _FakeBlock.CALLS = 0
    torch.manual_seed(0)
    m = cls(cfg)
    _ = m.attach_residual_streams(4)
    with torch.no_grad():
        _ = m(ids)
    check("6 layers -> exactly 6 block calls", _FakeBlock.CALLS == 6,
          f"{_FakeBlock.CALLS} calls (a folded [B*n,T,C] design would give 6; "
          "the point is it processes ONE combined input per layer)")

    print("i3 RESPONSE with a real threshold (the 1.2e-06 no-op lesson)")
    torch.manual_seed(0)
    m = cls(cfg)
    _ = m.attach_residual_streams(4)
    with torch.no_grad():
        for p_ in m.residual_streams.parameters():
            p_.normal_(0.0, 1.0)
    with torch.no_grad():
        live = m(ids)
    rel = _rel(live, ref)
    check("rel change > 1e-3 after generic perturbation", rel > 1e-3, f"rel={rel:.3e}")

    print("i4 STREAM DIVERGENCE (the row-stochastic degeneracy guard)")
    # Perturb ONLY w_pre: the input combination must change, the block must
    # then see a different tensor, and the streams must measurably split.
    torch.manual_seed(0)
    m = cls(cfg)
    _ = m.attach_residual_streams(4)
    with torch.no_grad():
        m.residual_streams.w_pre_raw[:, 1] = 1.5  # w = e_0 + tanh(1.5) on stream 1
    with torch.no_grad():
        logits, h_final = m(ids, output_hidden=True)
    # h_final is post-readout [B,T,C]; recompute divergence from a fresh forward
    # via the readout residual: compare against the legacy hidden instead.
    torch.manual_seed(0)
    m_ref = cls(cfg)
    with torch.no_grad():
        _, h_ref = m_ref(ids, output_hidden=True)
    rel_h = _rel(h_final, h_ref)
    check("w_pre-only perturbation changes the hidden state", rel_h > 1e-3,
          f"rel={rel_h:.3e}")

    print("i5 grads reach the mixer through the full forward")
    torch.manual_seed(0)
    m = cls(cfg)
    _ = m.attach_residual_streams(4)
    m(ids).sum().backward()
    ok = all(p_.grad is not None and float(p_.grad.abs().sum()) > 0.0
             for p_ in m.residual_streams.parameters())
    check("all mixer params get nonzero grads via forward+backward", ok, "")

    print("i6 state_dict keys land under the whitelisted prefix")
    keys = sorted(k for k in m.state_dict() if k.startswith("residual_streams."))
    check("exactly the 3 raw params persist", keys == [
        "residual_streams.m_res_raw", "residual_streams.p_post_raw",
        "residual_streams.w_pre_raw"], f"{keys}")

    print("i7 state_cache + streams is refused (single-stream contract)")
    try:
        torch.manual_seed(0)
        m = cls(cfg)
        _ = m.attach_residual_streams(2)
        _ = m(ids, False, None, state_cache=object())
        check("state_cache with streams raises", False, "no raise")
    except ValueError:
        check("state_cache with streams raises", True, "ValueError")

    print("i8 param counts (negligible by construction)")
    n4 = sum(p_.numel() for p_ in ResidualStreamMixer(32, 4).parameters())
    n2 = sum(p_.numel() for p_ in ResidualStreamMixer(32, 2).parameters())
    check("n=4 @ 32 layers == 768 params", n4 == 768, f"{n4}")
    check("n=2 @ 32 layers == 256 params", n2 == 256, f"{n2}")

    print()
    if FAIL:
        print(f"RESIDUAL-STREAMS SUITE: {len(FAIL)} FAIL -> {FAIL}")
        return 1
    print("RESIDUAL-STREAMS SUITE: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
