"""Shared-channel outcome laws for uplink transmissions."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .config import NUM_UL_CHANNELS

class AlohaChannel:
    def __init__(self, num_channels: int = NUM_UL_CHANNELS):
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        self.num_channels = num_channels
        self._txs: Dict[int, List[Tuple[str, float, float]]] = {
            c: [] for c in range(num_channels)}

    def reset(self) -> None:
        for c in self._txs:
            self._txs[c] = []

    def add_transmission(self, packet_id: str, channel_idx: int,
                         start_s: float, end_s: float) -> None:
        self._txs[channel_idx].append((packet_id, start_s, end_s))

    def resolve(self) -> set:
        successful = set()
        for ch_txs in self._txs.values():
            for i, (packet_i, s_i, e_i) in enumerate(ch_txs):
                collided = any(
                    s_i < e_j and e_i > s_j
                    for j, (packet_j, s_j, e_j) in enumerate(ch_txs) if i != j)
                if not collided:
                    successful.add(packet_i)
        return successful

# ---------------------------------------------------------------------------
