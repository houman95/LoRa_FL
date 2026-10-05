"""LoRa physical-layer packet sizing and airtime helpers."""

from __future__ import annotations

import math

def LoRa_MTU(sf: int) -> int:
    if sf in (7, 8): return 222
    if sf == 9:      return 115
    return 51

def LoRa_frame_duration(sf: int, payload_bytes: int) -> float:
    n_pr = 8; h = 1; c = 1; bw = 125e3
    t_sym = (2 ** sf) / bw
    n_payload = 8 + max(
        math.ceil((2 * payload_bytes - sf - 5 * h + 11) /
                  (sf - 2 * (sf >= 11))) * (c + 4), 0)
    return (n_pr + 4.25) * t_sym + n_payload * t_sym

# ---------------------------------------------------------------------------
