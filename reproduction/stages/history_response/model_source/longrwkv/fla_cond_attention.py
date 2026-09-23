"""The conditioned RWKV-7 mixer: two changed lines against the installed fla layer.

The paper requires the noise conditioner to act on the mixer's **native gate
preactivations** -- ``w = chi_w(g_w(h) + b^retention)`` -- while keeping chi, its
domains and all native normalization.  fla exposes no hook there, so this
subclasses ``RWKV7Attention`` and reproduces its ``forward`` with exactly two
insertions:

    w_pre = self.w_lora(xw)
    w_pre = w_pre + gate_bias[0]        # <-- added
    w     = -0.6065306597126334 * w_pre.sigmoid()

    a_pre = self.a_lora(xa)
    a_pre = a_pre + gate_bias[1]        # <-- added
    a     = a_pre.sigmoid()

Everything else is fla's own code, transcribed from ``fla/layers/rwkv7.py``
(lines 233-352 of version 0.5.0).

Because the body is copied, the copy could silently drift from fla.  It is not
guarded by a source hash -- that would fail on whitespace and pass on a semantic
change.  It is guarded instead by ``parity_with_native_mixer``, which runs this
class with ``gate_bias=None`` against a plain ``RWKV7Attention`` carrying the same
weights and requires bit-exact agreement.  A drift in the copied body, or a mistake
in the two inserted lines, changes the numbers and fails.

Adding a bias cannot escape the gates' domains: ``w`` stays in ``(-0.6065, 0)`` and
``a`` in ``(0, 1)`` for any finite bias, so ``chunk_rwkv7(..., safe_gate=True)``'s
assumption still holds.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from fla.layers.rwkv7 import RWKV7Attention
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules.l2norm import l2_norm
from fla.modules.token_shift import token_shift
from fla.ops.rwkv7 import chunk_rwkv7, fused_mul_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from fla.ops.rwkv7.gate_output_correction import gate_output_correction

from .refusal import Refusal

W_DECAY_SCALE = -0.6065306597126334


class CondRWKV7Attention(RWKV7Attention):
    """``RWKV7Attention`` whose two gate preactivations accept a per-position bias.

    ``__init__`` is inherited unchanged, so the parameter names, shapes and
    initialization are fla's, and a checkpoint written for the native layer loads
    here without a key translation.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        v_first: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        gate_bias: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        """Native forward with an optional ``(b_retention, b_write)`` preactivation bias.

        ``gate_bias`` entries are ``[N, key_dim]`` aligned with ``hidden_states`` in
        token order.  A caller passing the *reversed* input must pass correspondingly
        reversed biases; nothing here re-derives them.
        """
        if gate_bias is not None:
            if not isinstance(gate_bias, tuple) or len(gate_bias) != 2:
                raise Refusal("gate_bias must be a (retention, write) pair")
            for index, bias in enumerate(gate_bias):
                if bias.shape[-1] != self.key_dim:
                    raise Refusal(
                        f"gate_bias[{index}] has width {bias.shape[-1]}, expected "
                        f"the key dimension {self.key_dim}")
                if bias.shape[0] != hidden_states.shape[0] * hidden_states.shape[1]:
                    raise Refusal(
                        f"gate_bias[{index}] has {bias.shape[0]} rows, expected one "
                        f"per token ({hidden_states.shape[0] * hidden_states.shape[1]})")

        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
            am = attention_mask.narrow(1, attention_mask.size(1) - seq_len, seq_len).unsqueeze(-1)

        last_state = get_layer_cache(self, past_key_values)

        if attention_mask is not None:
            hidden_states = hidden_states.mul(am)

        if last_state is None:
            conv_cache = None
            recurrent_state = None
        else:
            conv_cache = last_state['conv_state']
            recurrent_state = last_state['recurrent_state']

        delta, conv_state = token_shift(
            hidden_states, cu_seqlens, output_cache=True, cache=conv_cache,
        )
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(hidden_states, delta, self.x_r, self.x_w,
                                                     self.x_k, self.x_v, self.x_a, self.x_g)

        r = self.r_proj(xr)
        # --- change 1 of 2: the retention preactivation carries the bias ---------
        w = self.w_lora(xw)
        if gate_bias is not None:
            w = w + gate_bias[0].to(w.dtype)
        w = W_DECAY_SCALE * w.sigmoid()

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        # --- change 2 of 2: the write preactivation carries the bias -------------
        a = self.a_lora(xa)
        if gate_bias is not None:
            a = a + gate_bias[1].to(a.dtype)
        a = a.sigmoid()
        g = self.g_lora(xg)

        if self.fuse_norm:
            kk = l2_norm(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim))
        else:
            kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim),
                             dim=-1, p=2.0)

        k = fused_k_rwkv7(k, a, self.k_a)

        if attention_mask is not None:
            v = v * am

        r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim),
                         (r, w, k, a))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        if self.training or seq_len >= 64:
            o, recurrent_state = chunk_rwkv7(
                r=r, w=w, k=k, v=v, a=-kk, b=kk * a, scale=1.,
                initial_state=recurrent_state, output_final_state=use_cache,
                cu_seqlens=cu_seqlens, safe_gate=True, chunk_size=64,
            )
        else:
            o, recurrent_state = fused_mul_recurrent_rwkv7(
                r=r, w=w, k=k, v=v, kk=kk, a=a, scale=1.,
                initial_state=recurrent_state, output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

        update_layer_cache(
            self, past_key_values, recurrent_state=recurrent_state,
            conv_state=conv_state, offset=r.shape[1],
        )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
        else:
            o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)

        o = gate_output_correction(o, r, k, self.r_k, v, g)
        o = self.o_proj(o)

        return o, None, past_key_values, v_first


def parity_with_native_mixer(native: RWKV7Attention,
                             conditioned: CondRWKV7Attention,
                             hidden_states: torch.Tensor,
                             v_first: torch.Tensor,
                             cu_seqlens: torch.LongTensor | None = None,
                             tolerance: float = 0.0) -> dict:
    """Prove the copied body still reproduces the native layer.

    Both mixers must be in the same mode, and the conditioned one must be called
    with ``gate_bias=None``.  ``tolerance=0`` demands bit-exact agreement, which is
    the right demand for a copied body: anything the copy does differently shows up
    immediately rather than being averaged into a small difference.

    Returns a record for the qualification log rather than only asserting, so a
    failure is diagnosable from the run output.
    """
    for name, parameter in native.named_parameters():
        counterpart = dict(conditioned.named_parameters()).get(name)
        if counterpart is None:
            raise Refusal(f"conditioned mixer is missing native parameter {name!r}")
        if counterpart.shape != parameter.shape:
            raise Refusal(
                f"{name}: native {tuple(parameter.shape)} vs conditioned "
                f"{tuple(counterpart.shape)}")

    native.eval()
    conditioned.eval()
    with torch.no_grad():
        native_out = native(hidden_states, v_first=v_first, cu_seqlens=cu_seqlens)[0]
        conditioned_out = conditioned(hidden_states, v_first=v_first,
                                      cu_seqlens=cu_seqlens, gate_bias=None)[0]

    difference = (native_out.float() - conditioned_out.float()).abs().max().item()
    record = {
        "tolerance": tolerance,
        "max_abs_difference": difference,
        "bit_exact": difference == 0.0,
        "shape": list(native_out.shape),
    }
    if tolerance == 0.0 and difference != 0.0:
        raise Refusal(
            f"conditioned mixer differs from the native layer by {difference} "
            f"with gate_bias=None; the copied forward body has drifted")
    if difference > tolerance:
        raise Refusal(
            f"conditioned mixer differs from the native layer by {difference} > "
            f"tolerance {tolerance}")
    return record


def zero_bias_parity(conditioned: CondRWKV7Attention,
                     hidden_states: torch.Tensor,
                     v_first: torch.Tensor,
                     cu_seqlens: torch.LongTensor | None = None) -> dict:
    """A3 with a zero-initialized modulator must equal A2 bit-for-bit.

    The modulator's output projection is zero-initialized in weight and bias, so at
    step 0 the bias it produces is the zero tensor -- and adding zero must not
    change the result.  This is the check that makes A3-minus-A2 a gate-modulation
    contrast rather than an initialization contrast.
    """
    tokens = hidden_states.shape[0] * hidden_states.shape[1]
    zeros = (torch.zeros(tokens, conditioned.key_dim, device=hidden_states.device,
                         dtype=hidden_states.dtype),
             torch.zeros(tokens, conditioned.key_dim, device=hidden_states.device,
                         dtype=hidden_states.dtype))
    conditioned.eval()
    with torch.no_grad():
        without = conditioned(hidden_states, v_first=v_first,
                              cu_seqlens=cu_seqlens, gate_bias=None)[0]
        with_zero = conditioned(hidden_states, v_first=v_first,
                                cu_seqlens=cu_seqlens, gate_bias=zeros)[0]
    difference = (without.float() - with_zero.float()).abs().max().item()
    if difference != 0.0:
        raise Refusal(
            f"a zero gate bias changed the output by {difference}; the mixer is not "
            f"additive in the bias and A3-minus-A2 would not isolate modulation")
    return {"zero_bias_max_abs_difference": difference, "bit_exact": True}
