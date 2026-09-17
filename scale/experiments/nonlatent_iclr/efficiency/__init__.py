"""Measured efficiency matrix: the honest seam over heterogeneous checkpoints.

Contribution 1 of the paper is an efficiency claim, so this package exists to
make the comparison measurable rather than asserted.  ``adapters`` is the one
interface every model is reached through; it keeps the cost of a single answer
(``nfe``) and the analytic running-state footprint explicit, so a 1-forward
autoregressive decode and a multi-forward diffusion answer can never be tabulated
as if they were the same unit of work.
"""

from . import adapters as adapters

__all__ = ["adapters"]
