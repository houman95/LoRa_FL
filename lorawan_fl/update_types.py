"""Data types shared by reconstruction and aggregation.

Communication loss is represented by ``valid_mask``.  A numerical zero in
``values`` remains an ordinary received value whenever its mask entry is true.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReconstructedUpdate:
    """A dense client update plus coordinate-level reception validity.

    ``values`` may contain any placeholder at invalid coordinates; aggregation
    must use ``valid_mask`` and must never interpret that placeholder as an
    observed client value.  ``received_fragments`` and ``total_fragments``
    preserve whole-update reception state for the discard-all baseline.
    """

    values: torch.Tensor
    valid_mask: torch.Tensor
    received_fragments: int | None = None
    total_fragments: int | None = None

    def __post_init__(self) -> None:
        values = self.values.detach().cpu().float()
        valid_mask = self.valid_mask.detach().cpu().bool()
        if values.shape != valid_mask.shape:
            raise ValueError("values and valid_mask must have identical shapes")
        if (self.received_fragments is None) != (self.total_fragments is None):
            raise ValueError(
                "received_fragments and total_fragments must be provided together"
            )
        if self.total_fragments is not None:
            if self.total_fragments <= 0:
                raise ValueError("total_fragments must be positive")
            if not 0 <= int(self.received_fragments) <= self.total_fragments:
                raise ValueError("received_fragments must be within fragment count")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid_mask)

    @property
    def observed(self) -> torch.Tensor:
        """Backward-compatible float mask used by pre-refactor result code."""

        return self.valid_mask.to(dtype=torch.float32)

    @property
    def complete_reception(self) -> bool:
        """Whether every physical fragment was received."""

        if self.total_fragments is not None:
            return self.received_fragments == self.total_fragments
        return bool(torch.all(self.valid_mask).item())

