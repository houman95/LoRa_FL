"""Pure aggregation policies for complete and partially received updates.

This module deliberately has no collision, packet-scheduling, or jitter
dependencies.  Its inputs are already reconstructed client tensors and their
validity masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import torch

from .update_types import ReconstructedUpdate


class AggregationPolicy(str, Enum):
    """Server handling of incomplete client fragment trains."""

    DISCARD_ALL = "discard_all"
    LEGACY_ZERO_FILL = "legacy_zero_fill"
    MASKED_PARTIAL = "masked_partial"
    MASKED_FIXED_DENOMINATOR = "masked_fixed_denominator"


@dataclass(frozen=True)
class AggregationCoverage:
    """Coordinate-level counts of valid client contributors."""

    contributors_per_coordinate: torch.Tensor

    @property
    def mean_contributors_per_coordinate(self) -> float:
        return float(self.contributors_per_coordinate.float().mean().item())

    @property
    def median_contributors_per_coordinate(self) -> float:
        return float(self.contributors_per_coordinate.float().median().item())

    @property
    def min_contributors_per_coordinate(self) -> int:
        return int(self.contributors_per_coordinate.min().item())

    @property
    def max_contributors_per_coordinate(self) -> int:
        return int(self.contributors_per_coordinate.max().item())

    @property
    def fraction_coordinates_with_zero_contributors(self) -> float:
        return float(
            (self.contributors_per_coordinate == 0).float().mean().item()
        )

    def fraction_coordinates_with_n_contributors(self, n: int) -> float:
        return float(
            (self.contributors_per_coordinate == int(n)).float().mean().item()
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_contributors_per_coordinate": (
                self.mean_contributors_per_coordinate
            ),
            "median_contributors_per_coordinate": (
                self.median_contributors_per_coordinate
            ),
            "min_contributors_per_coordinate": (
                self.min_contributors_per_coordinate
            ),
            "max_contributors_per_coordinate": (
                self.max_contributors_per_coordinate
            ),
            "fraction_coordinates_with_zero_contributors": (
                self.fraction_coordinates_with_zero_contributors
            ),
            "fraction_coordinates_with_one_contributor": (
                self.fraction_coordinates_with_n_contributors(1)
            ),
            "fraction_coordinates_with_two_contributors": (
                self.fraction_coordinates_with_n_contributors(2)
            ),
        }


@dataclass(frozen=True)
class AggregationResult:
    aggregated_delta: torch.Tensor
    coverage: AggregationCoverage
    contributing_clients: int


def parse_aggregation_policy(value: str | AggregationPolicy) -> AggregationPolicy:
    if isinstance(value, AggregationPolicy):
        return value
    try:
        return AggregationPolicy(str(value))
    except ValueError as exc:
        choices = ", ".join(policy.value for policy in AggregationPolicy)
        raise ValueError(f"aggregation_policy must be one of: {choices}") from exc


def policy_from_config(config: Mapping) -> AggregationPolicy:
    """Resolve the explicit policy, with exact legacy-flag compatibility."""

    if config.get("aggregation_policy") is not None:
        return parse_aggregation_policy(config["aggregation_policy"])
    if not bool(config.get("partial_fragment_aggregation", False)):
        return AggregationPolicy.DISCARD_ALL
    if bool(config.get("coordinate_wise_fragment_normalization", False)):
        return AggregationPolicy.MASKED_PARTIAL
    return AggregationPolicy.LEGACY_ZERO_FILL


def _stack_inputs(
    client_updates: Sequence[torch.Tensor],
    client_valid_masks: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not client_updates:
        raise ValueError("at least one client update is required")
    if len(client_updates) != len(client_valid_masks):
        raise ValueError("client_updates and client_valid_masks must align")
    values = torch.stack([
        update.detach().cpu().float() for update in client_updates
    ])
    valid = torch.stack([
        mask.detach().cpu().bool() for mask in client_valid_masks
    ])
    if values.shape != valid.shape:
        raise ValueError("client update and validity-mask shapes must match")
    return values, valid


def aggregate_partial_updates(
    client_updates: Sequence[torch.Tensor],
    client_valid_masks: Sequence[torch.Tensor],
    client_weights: Sequence[float] | None = None,
) -> AggregationResult:
    """Aggregate each coordinate over only clients that contributed it.

    If ``client_weights`` is omitted, equal client weighting is used.  With
    weights, both numerator and denominator are coordinate-wise and include
    only valid contributors.  A coordinate with no contributor receives a
    final aggregated delta of zero, meaning the server applies no change to
    that model coordinate; it is not treated as an observed zero gradient.
    """

    values, valid = _stack_inputs(client_updates, client_valid_masks)
    num_clients = values.shape[0]
    if client_weights is None:
        weights = torch.ones(num_clients, dtype=values.dtype)
    else:
        if len(client_weights) != num_clients:
            raise ValueError("client_weights must align with client_updates")
        weights = torch.as_tensor(client_weights, dtype=values.dtype)
        if torch.any(weights < 0):
            raise ValueError("client_weights must be nonnegative")
    broadcast_shape = (num_clients,) + (1,) * (values.ndim - 1)
    weighted_valid = valid.to(values.dtype) * weights.reshape(broadcast_shape)
    numerator = (values * weighted_valid).sum(dim=0)
    denominator = weighted_valid.sum(dim=0)
    aggregated = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(numerator),
    )
    counts = valid.sum(dim=0).to(torch.int64)
    return AggregationResult(
        aggregated_delta=aggregated,
        coverage=AggregationCoverage(counts),
        contributing_clients=int(torch.any(valid.reshape(num_clients, -1), dim=1).sum()),
    )


def aggregate_partial_updates_fixed_denominator(
    client_updates: Sequence[torch.Tensor],
    client_valid_masks: Sequence[torch.Tensor],
    client_weights: Sequence[float] | None = None,
) -> AggregationResult:
    """Zero missing coordinates but retain one selected-client denominator.

    This is the coverage-scaled mitigation requested after EXP010.  In
    contrast to observed-only normalization, the denominator is the total
    weight of every selected client represented in the input, including
    clients with zero received fragments.  It limits low-coverage magnitude
    inflation but does not guarantee that the result has one common effective
    client weighting across coordinates.
    """

    values, valid = _stack_inputs(client_updates, client_valid_masks)
    num_clients = values.shape[0]
    if client_weights is None:
        weights = torch.ones(num_clients, dtype=values.dtype)
    else:
        if len(client_weights) != num_clients:
            raise ValueError("client_weights must align with client_updates")
        weights = torch.as_tensor(client_weights, dtype=values.dtype)
        if torch.any(weights < 0):
            raise ValueError("client_weights must be nonnegative")
    denominator = weights.sum()
    if float(denominator.item()) <= 0.0:
        raise ValueError("fixed denominator requires positive total client weight")
    broadcast_shape = (num_clients,) + (1,) * (values.ndim - 1)
    weighted_valid = valid.to(values.dtype) * weights.reshape(broadcast_shape)
    numerator = (values * weighted_valid).sum(dim=0)
    aggregated = numerator / denominator
    counts = valid.sum(dim=0).to(torch.int64)
    return AggregationResult(
        aggregated_delta=aggregated,
        coverage=AggregationCoverage(counts),
        contributing_clients=int(
            torch.any(valid.reshape(num_clients, -1), dim=1).sum()
        ),
    )


def aggregate_reconstructed_updates(
    client_updates: Mapping[int, torch.Tensor | ReconstructedUpdate],
    policy: str | AggregationPolicy,
    client_weights: Mapping[int, float] | None = None,
) -> AggregationResult:
    """Apply one named baseline to already reconstructed client updates."""

    aggregation_policy = parse_aggregation_policy(policy)
    if not client_updates:
        raise ValueError("at least one client update is required")

    retained: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for client_id, update in client_updates.items():
        if isinstance(update, ReconstructedUpdate):
            if (
                aggregation_policy == AggregationPolicy.DISCARD_ALL
                and not update.complete_reception
            ):
                continue
            if (
                aggregation_policy == AggregationPolicy.MASKED_PARTIAL
                and not bool(torch.any(update.valid_mask).item())
            ):
                # A zero-reception client is mathematically absent from every
                # coordinate-wise numerator and denominator.  Removing it
                # before stacking also preserves exact equivalence with
                # discard_all when a realization has only full and zero
                # receptions.  Retaining all-false rows changes Torch's
                # floating-point reduction shape and can create artificial
                # policy differences even though the two formulas are equal.
                continue
            values = update.values
            if aggregation_policy in {
                AggregationPolicy.MASKED_PARTIAL,
                AggregationPolicy.MASKED_FIXED_DENOMINATOR,
            }:
                valid_mask = update.valid_mask
            else:
                # The diagnostic legacy baseline intentionally treats the
                # zero placeholders at missing coordinates as valid values.
                valid_mask = torch.ones_like(update.valid_mask, dtype=torch.bool)
        else:
            values = update.detach().cpu().float()
            valid_mask = torch.ones_like(values, dtype=torch.bool)
        retained.append((client_id, values, valid_mask))

    if not retained:
        exemplar = next(iter(client_updates.values()))
        values = exemplar.values if isinstance(exemplar, ReconstructedUpdate) else exemplar
        zeros = torch.zeros_like(values, dtype=torch.float32)
        counts = torch.zeros_like(values, dtype=torch.int64)
        return AggregationResult(zeros, AggregationCoverage(counts), 0)

    weights = None
    if client_weights is not None:
        weights = [float(client_weights[client_id]) for client_id, _, _ in retained]
    aggregation_function = (
        aggregate_partial_updates_fixed_denominator
        if aggregation_policy == AggregationPolicy.MASKED_FIXED_DENOMINATOR
        else aggregate_partial_updates
    )
    return aggregation_function(
        [values for _, values, _ in retained],
        [valid_mask for _, _, valid_mask in retained],
        weights,
    )
