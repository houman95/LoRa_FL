"""Deterministic update fragmentation and server-side reconstruction.

The communication layer carries fragment identity and reception status.  This
module alone maps a fragment index to sparse-update coordinates and rebuilds a
dense value tensor plus a validity mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .compression import decode_sparse_update
from .update_types import ReconstructedUpdate


COORDINATE_MAPPING = "sorted_sparse_records_even_shards_v1"


class FragmentStatus(str, Enum):
    RECEIVED = "received"
    COLLIDED = "collided"


@dataclass(frozen=True)
class FragmentIdentity:
    client_id: int
    round_index: int
    fragment_index: int
    total_fragments: int

    def __post_init__(self) -> None:
        if self.client_id < 0:
            raise ValueError("client_id must be nonnegative")
        if self.round_index < 0:
            raise ValueError("round_index must be nonnegative")
        if self.total_fragments <= 0:
            raise ValueError("total_fragments must be positive")
        if not 0 <= self.fragment_index < self.total_fragments:
            raise ValueError("fragment_index must be within total_fragments")


@dataclass(frozen=True)
class UpdateFragment:
    identity: FragmentIdentity
    coordinate_indices: torch.Tensor
    coordinate_values: torch.Tensor
    coordinate_mapping: str = COORDINATE_MAPPING


@dataclass(frozen=True)
class FragmentReception:
    identity: FragmentIdentity
    status: FragmentStatus

    @property
    def received(self) -> bool:
        return self.status == FragmentStatus.RECEIVED


def fragment_to_indices(
    encoded_coordinate_indices: Sequence[int] | np.ndarray,
    fragment_index: int,
    total_fragments: int,
) -> np.ndarray:
    """Return model coordinates represented by one deterministic shard."""

    if total_fragments <= 0:
        raise ValueError("total_fragments must be positive")
    if not 0 <= fragment_index < total_fragments:
        raise ValueError("fragment_index must be within total_fragments")
    indices = np.asarray(encoded_coordinate_indices, dtype=np.int64)
    encoded_positions = np.argsort(indices, kind="stable")
    shard_positions = np.array_split(encoded_positions, total_fragments)[
        fragment_index
    ]
    return indices[shard_positions].copy()


def fragment_update(
    blob: bytes,
    apply_zlib: bool,
    client_id: int,
    round_index: int,
    total_fragments: int,
) -> list[UpdateFragment]:
    """Materialize the deterministic sparse-record mapping for an update."""

    if total_fragments <= 0:
        raise ValueError("total_fragments must be positive")
    indices, values, _ = decode_sparse_update(blob, apply_zlib)
    encoded_positions = np.argsort(indices, kind="stable")
    shards = np.array_split(encoded_positions, total_fragments)
    fragments: list[UpdateFragment] = []
    for fragment_index, positions in enumerate(shards):
        fragments.append(UpdateFragment(
            identity=FragmentIdentity(
                client_id=client_id,
                round_index=round_index,
                fragment_index=fragment_index,
                total_fragments=total_fragments,
            ),
            coordinate_indices=torch.as_tensor(
                indices[positions].copy(), dtype=torch.int64
            ),
            coordinate_values=torch.as_tensor(
                values[positions].copy(), dtype=torch.float32
            ),
        ))
    return fragments


def _validate_receptions(
    receptions: Sequence[FragmentReception],
    client_id: int,
    round_index: int,
    total_fragments: int,
) -> dict[int, FragmentStatus]:
    statuses: dict[int, FragmentStatus] = {}
    for reception in receptions:
        identity = reception.identity
        if identity.client_id != client_id or identity.round_index != round_index:
            raise ValueError("fragment reception belongs to a different update")
        if identity.total_fragments != total_fragments:
            raise ValueError("inconsistent total_fragments in reception metadata")
        if identity.fragment_index in statuses:
            raise ValueError("duplicate fragment reception metadata")
        statuses[identity.fragment_index] = reception.status
    if set(statuses) != set(range(total_fragments)):
        raise ValueError("reception metadata must cover every fragment exactly once")
    return statuses


def reconstruct_partial_update(
    blob: bytes,
    apply_zlib: bool,
    receptions: Sequence[FragmentReception],
) -> ReconstructedUpdate:
    """Reconstruct received values without confusing loss with numeric zero."""

    if not receptions:
        raise ValueError("receptions must include every fragment status")
    first = receptions[0].identity
    statuses = _validate_receptions(
        receptions,
        first.client_id,
        first.round_index,
        first.total_fragments,
    )
    _, _, dimension = decode_sparse_update(blob, apply_zlib)
    values = torch.zeros(dimension, dtype=torch.float32)
    # Start with no observations.  If at least one independently decodable
    # sparse fragment arrives, coordinates omitted by top-k are known sparse
    # zeros, while encoded coordinates carried by lost fragments remain
    # missing.  If no fragment arrives, the client must contribute nothing.
    valid_mask = torch.zeros(dimension, dtype=torch.bool)
    received_fragments = 0
    fragments = fragment_update(
        blob,
        apply_zlib,
        first.client_id,
        first.round_index,
        first.total_fragments,
    )
    for fragment in fragments:
        fragment_index = fragment.identity.fragment_index
        if statuses[fragment_index] == FragmentStatus.RECEIVED:
            received_fragments += 1
            values[fragment.coordinate_indices] = fragment.coordinate_values
    if received_fragments > 0:
        valid_mask.fill_(True)
        for fragment in fragments:
            fragment_index = fragment.identity.fragment_index
            if statuses[fragment_index] == FragmentStatus.COLLIDED:
                valid_mask[fragment.coordinate_indices] = False
    else:
        # This explicit branch documents the no-reception invariant and guards
        # against future changes to sparse-zero handling.
        valid_mask.zero_()
    return ReconstructedUpdate(
        values=values,
        valid_mask=valid_mask,
        received_fragments=received_fragments,
        total_fragments=first.total_fragments,
    )


def reconstruct_partial_update_strict_fragment_local(
    blob: bytes,
    apply_zlib: bool,
    receptions: Sequence[FragmentReception],
) -> ReconstructedUpdate:
    """Reconstruct only coordinates physically carried by received shards.

    This diagnostic alternative deliberately does *not* infer that top-k
    omitted coordinates are known zeros after one fragment arrives.  It is not
    the historical EXP010 behavior and is provided only for the bounded
    sparse-zero decodability audit.
    """

    if not receptions:
        raise ValueError("receptions must include every fragment status")
    first = receptions[0].identity
    statuses = _validate_receptions(
        receptions,
        first.client_id,
        first.round_index,
        first.total_fragments,
    )
    _, _, dimension = decode_sparse_update(blob, apply_zlib)
    values = torch.zeros(dimension, dtype=torch.float32)
    valid_mask = torch.zeros(dimension, dtype=torch.bool)
    received_fragments = 0
    for fragment in fragment_update(
        blob,
        apply_zlib,
        first.client_id,
        first.round_index,
        first.total_fragments,
    ):
        if statuses[fragment.identity.fragment_index] == FragmentStatus.RECEIVED:
            received_fragments += 1
            values[fragment.coordinate_indices] = fragment.coordinate_values
            valid_mask[fragment.coordinate_indices] = True
    return ReconstructedUpdate(
        values=values,
        valid_mask=valid_mask,
        received_fragments=received_fragments,
        total_fragments=first.total_fragments,
    )


def receptions_from_delivered_indices(
    client_id: int,
    round_index: int,
    total_fragments: int,
    delivered_fragment_indices: Iterable[int],
) -> list[FragmentReception]:
    delivered = {int(index) for index in delivered_fragment_indices}
    if any(index < 0 or index >= total_fragments for index in delivered):
        raise ValueError(
            f"delivered fragment indices must be in [0, {total_fragments - 1}]"
        )
    return [
        FragmentReception(
            FragmentIdentity(client_id, round_index, index, total_fragments),
            FragmentStatus.RECEIVED if index in delivered else FragmentStatus.COLLIDED,
        )
        for index in range(total_fragments)
    ]


def receptions_from_transmission_log(
    transmission_rows: Iterable[Mapping],
) -> dict[int, list[FragmentReception]]:
    """Build complete per-client reception metadata from scheduler rows."""

    grouped: dict[int, list[FragmentReception]] = {}
    for row in transmission_rows:
        client_id = int(row["client_id"])
        status = (
            FragmentStatus.RECEIVED
            if row["status"] in {"success", "received"}
            else FragmentStatus.COLLIDED
        )
        grouped.setdefault(client_id, []).append(FragmentReception(
            FragmentIdentity(
                client_id=client_id,
                round_index=int(row.get("round_index", row.get("episode_idx", 0))),
                fragment_index=int(row["fragment_idx"]),
                total_fragments=int(row["num_fragments"]),
            ),
            status,
        ))
    for client_id in grouped:
        grouped[client_id].sort(key=lambda item: item.identity.fragment_index)
    return grouped


def fragment_reception_diagnostics(
    selected_clients: Sequence[int],
    receptions_by_client: Mapping[int, Sequence[FragmentReception]],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Return requested round-level and per-client reception diagnostics."""

    client_rows: list[dict[str, float | int]] = []
    full = partial = zero = 0
    attempted = successful = 0
    for client_id, receptions in sorted(receptions_by_client.items()):
        total = len(receptions)
        received = sum(reception.received for reception in receptions)
        collided = total - received
        attempted += total
        successful += received
        full += int(received == total)
        partial += int(0 < received < total)
        zero += int(received == 0)
        client_rows.append({
            "client_id": client_id,
            "num_fragments": total,
            "successful_fragments": received,
            "collided_fragments": collided,
            "fraction_received": received / max(1, total),
        })
    for client_id in sorted(set(selected_clients) - set(receptions_by_client)):
        zero += 1
        client_rows.append({
            "client_id": client_id,
            "num_fragments": 0,
            "successful_fragments": 0,
            "collided_fragments": 0,
            "fraction_received": 0.0,
        })
    diagnostics = {
        "selected_clients": len(selected_clients),
        "num_fragments_attempted": attempted,
        "num_fragments_successful": successful,
        "num_fragments_collided": attempted - successful,
        "fragment_success_rate": successful / max(1, attempted),
        "num_clients_full_success": full,
        "num_clients_partial_success": partial,
        "num_clients_zero_success": zero,
    }
    return diagnostics, client_rows
