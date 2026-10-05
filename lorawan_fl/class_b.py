"""LoRaWAN Class-B beacon, ping-slot, and downlink fragmentation helpers."""

from __future__ import annotations

import math
import struct
from typing import List, Optional, Tuple

from .config import BEACON_GUARD_S, BEACON_PERIOD_S, BEACON_RESERVED_S, SLOT_LEN_S
from .lora_phy import LoRa_frame_duration

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    def _aes128_ecb(key: bytes, pt: bytes) -> bytes:
        enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
        return enc.update(pt) + enc.finalize()
except ImportError:
    from Crypto.Cipher import AES

    def _aes128_ecb(key: bytes, pt: bytes) -> bytes:
        return AES.new(key, AES.MODE_ECB).encrypt(pt)

def _ping_offset(beacon_time_s: int, dev_addr: int, ping_period: int) -> int:
    key     = bytes(16)
    payload = struct.pack("<II", beacon_time_s & 0xFFFFFFFF, dev_addr & 0xFFFFFFFF)
    payload += bytes(16 - len(payload))
    rand    = _aes128_ecb(key, payload)
    return (rand[0] + rand[1] * 256) % ping_period

def ping_slot_times(beacon_start_s: float, beacon_time_int: int,
                    dev_addr: int, periodicity: int) -> List[float]:
    assert 0 <= periodicity <= 7
    ping_nb     = 1 << (7 - periodicity)
    ping_period = 1 << (5 + periodicity)
    offset      = _ping_offset(beacon_time_int, dev_addr, ping_period)
    return [
        beacon_start_s + BEACON_RESERVED_S + (offset + n * ping_period) * SLOT_LEN_S
        for n in range(ping_nb)
    ]

def _next_safe_dl_start(t: float) -> float:
    """Snap t into the usable ping-slot window of its beacon period."""
    idx          = math.floor(t / BEACON_PERIOD_S)
    beacon_start = idx * BEACON_PERIOD_S
    reserved_end = beacon_start + BEACON_RESERVED_S
    guard_start  = beacon_start + BEACON_PERIOD_S - BEACON_GUARD_S
    if t < reserved_end:
        return reserved_end
    if t >= guard_start:
        return (idx + 1) * BEACON_PERIOD_S + BEACON_RESERVED_S
    return t

def classb_ping_slot_delivery(dev_addr, ping_periodicity, beacon_start_s,
                               beacon_time_int, dl_start_s, dl_end_s) -> bool:
    slots = ping_slot_times(beacon_start_s, beacon_time_int, dev_addr, ping_periodicity)
    return any(t < dl_end_s and (t + SLOT_LEN_S) > dl_start_s for t in slots)

def classb_multicast_dl_start(
    round_ready_s: float,
    gw_dc: "DutyCycleTracker",
) -> Tuple[float, float]:
    """
    Return the next usable Class-B multicast start after both the FL server
    and gateway radio are ready.
    """
    raw_start    = max(round_ready_s, gw_dc.earliest_tx())
    dl_start     = _next_safe_dl_start(raw_start)
    beacon_start = math.floor(dl_start / BEACON_PERIOD_S) * BEACON_PERIOD_S
    return dl_start, beacon_start

def _downlink_fragment_sizes(
    payload_bytes: int,
    mtu: int,
    fec_rate: float,
) -> List[int]:
    """
    Packetize a global update into LoRaWAN downlink frames.
    Data fragments carry the serialized update; optional FEC fragments are
    modeled as full-sized repair packets, matching the previous airtime model.
    """
    if fec_rate <= 0.0:
        raise ValueError("FEC_rate must be > 0 for downlink packetization")
    data_fragments = max(1, math.ceil(payload_bytes / mtu))
    repair_fragments = max(0, math.ceil(data_fragments / fec_rate) - data_fragments)

    sizes = [
        min(mtu, max(1, payload_bytes - i * mtu))
        for i in range(data_fragments)
    ]
    sizes.extend([mtu] * repair_fragments)
    return sizes

def _schedule_gateway_downlink_fragments(
    first_ready_s: float,
    gw_dc: "DutyCycleTracker",
    fragment_sizes: List[int],
    spreading_factor: int,
) -> Tuple[float, float, float, List[Tuple[float, float]]]:
    """
    Serialize gateway downlink packets under EU868 duty-cycle constraints.
    Each frame consumes airtime and sets the gateway off-air interval before
    the next downlink frame can start.
    """
    first_start: Optional[float] = None
    final_end = first_ready_s
    total_airtime = 0.0
    ready_s = first_ready_s
    fragment_windows: List[Tuple[float, float]] = []

    for payload_bytes in fragment_sizes:
        frag_start, _ = classb_multicast_dl_start(ready_s, gw_dc)
        airtime = LoRa_frame_duration(spreading_factor, payload_bytes)
        slotted_airtime = math.ceil(airtime / SLOT_LEN_S) * SLOT_LEN_S
        frag_end = frag_start + slotted_airtime

        if first_start is None:
            first_start = frag_start
        total_airtime += airtime
        final_end = frag_end
        fragment_windows.append((frag_start, frag_end))

        gw_dc.record_tx(frag_start, slotted_airtime)
        ready_s = frag_end

    return (
        first_start if first_start is not None else first_ready_s,
        final_end,
        total_airtime,
        fragment_windows,
    )
