# Marker-only fixture. This is not an actual-model or actual-snapshot proof.
SOURCE_MARKERS = (
    "self.attn_fwd: RWKV7Attention = _make_attn()",
    "self.attn_bwd: RWKV7Attention = _make_attn()",
    "h_pass, _, _ = self._run_layer_range(",
)
