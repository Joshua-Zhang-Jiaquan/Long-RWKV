"""Typed recurrent-cache mutation for state-hijacking RELAY models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

CacheField: TypeAlias = torch.Tensor | None
CacheLayerState: TypeAlias = dict[str, CacheField]
_SEEN_TOKENS: Final = "_seen_tokens"


class CacheLayer(Protocol):
    """A mutable RWKV cache layer that accepts an injected recurrent state."""

    state: CacheLayerState | None


class CacheState(Protocol):
    """A mutable RWKV cache with ordered per-layer state."""

    layers: Sequence[CacheLayer]


def inject_predicted_states(
    cache: CacheState,
    predicted_states: Sequence[torch.Tensor],
) -> CacheState:
    """Replace each cache recurrent state with its predicted state."""
    for layer_index, predicted_state in enumerate(predicted_states):
        layer = cache.layers[layer_index]
        state = _ensure_layer_state(layer)
        state["recurrent_state"] = predicted_state.to(torch.float32)
        _zero_transient_states(state)
        setattr(layer, _SEEN_TOKENS, 0)
    setattr(cache, _SEEN_TOKENS, 0)
    return cache


def blend_predicted_states(
    cache: CacheState,
    predicted_states: Sequence[torch.Tensor],
    blend: float,
) -> CacheState:
    """Blend each predicted state into the matching cache recurrent state."""
    for layer_index, predicted_state in enumerate(predicted_states):
        layer = cache.layers[layer_index]
        state = _ensure_layer_state(layer)
        current_state = state["recurrent_state"]
        planned_state = predicted_state.to(torch.float32)
        state["recurrent_state"] = _blended_state(current_state, planned_state, blend)
        _zero_transient_states(state)
        setattr(layer, _SEEN_TOKENS, 0)
    setattr(cache, _SEEN_TOKENS, 0)
    return cache


def _ensure_layer_state(layer: CacheLayer) -> CacheLayerState:
    """Return a layer state dictionary, creating the cache's canonical shape."""
    if layer.state is None:
        layer.state = {
            "recurrent_state": None,
            "attn_state": None,
            "conv_state": None,
            "ffn_state": None,
        }
    return layer.state


def _blended_state(
    current_state: CacheField,
    planned_state: torch.Tensor,
    blend: float,
) -> torch.Tensor:
    """Apply the established convex state blend without changing its edge cases."""
    if isinstance(current_state, torch.Tensor) and 0.0 < blend < 1.0:
        return current_state.to(torch.float32) * (1.0 - blend) + planned_state * blend
    if blend <= 0.0 and isinstance(current_state, torch.Tensor):
        return current_state.to(torch.float32)
    return planned_state


def _zero_transient_states(state: CacheLayerState) -> None:
    """Clear cache fields whose historic values cannot follow a state injection."""
    for state_key in ("conv_state", "ffn_state"):
        transient_state = state[state_key]
        if isinstance(transient_state, torch.Tensor):
            state[state_key] = torch.zeros_like(transient_state)
