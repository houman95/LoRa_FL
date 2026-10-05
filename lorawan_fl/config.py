"""Shared constants and data-transfer objects for the LoRaWAN FL simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

BEACON_PERIOD_S = 128.0
BEACON_RESERVED_S = 2.120
BEACON_GUARD_S = 3.0
SLOT_LEN_S = 0.030
N_SLOTS = 4096

EU868_UL_CHANNELS = [868.1e6, 868.3e6, 868.5e6]
NUM_UL_CHANNELS = len(EU868_UL_CHANNELS)

DC_PERCENT = 1.0

INDEX_BYTES = 3
HEADER_BASE = 7
HEADER_QUANT_EXTRA = 8

class RoundPolicyType(str, Enum):
    FIXED_WINDOW   = "fixed_window"
    FIXED_K_TIMEOUT = "fixed_k_timeout"
    ADAPTIVE_COMPLETE = "adaptive_complete"

@dataclass
class RoundPolicyConfig:
    policy_type: RoundPolicyType
    window_s: float                      # max uplink collection window (W)
    target_successes_k: Optional[int] = None   # only for FIXED_K_TIMEOUT

    @property
    def name(self) -> str:
        if self.policy_type == RoundPolicyType.FIXED_K_TIMEOUT:
            return f"fixed_k{self.target_successes_k}_timeout"
        if self.policy_type == RoundPolicyType.ADAPTIVE_COMPLETE:
            return "adaptive_complete"
        return "fixed_window"

@dataclass
class CompressionConfig:
    use_sparsity:      bool  = True
    use_quantization:  bool  = True
    quantization_bits: int   = 4
    sparsity_threshold: float = 0.001
    apply_zlib:        bool  = False   # off by default for clean comparison
    max_payload_bytes: int   = 222     # LoRa_MTU(SF7)

@dataclass
class EnergyConfig:
    """Optional energy model and budgets used beyond regulatory duty cycle."""
    tx_power_w: float = 0.1
    max_client_packet_attempts: Optional[int] = None
    max_client_ul_airtime_s: Optional[float] = None
    max_client_tx_energy_j: Optional[float] = None
    max_round_packet_attempts: Optional[int] = None
    max_round_ul_airtime_s: Optional[float] = None

# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------
@dataclass
class UplinkTx:
    cid:          int
    packet_id:    str
    round_index:  int
    fragment_idx: int
    num_fragments: int
    start_s:      float
    end_s:        float
    channel_idx:  int
    airtime_s:    float
    payload_bytes: int
    update_blob:  bytes
    earliest_allowed_s: float = 0.0
    initial_jitter_s: float = 0.0
    fragment_jitter_s: float = 0.0
    preceding_fragment_jitter_s: float = 0.0

@dataclass
class UplinkCollectionResult:
    policy_name:        str
    collection_start_s: float
    collection_end_s:   float
    deadline_s:         float
    trigger_reason:     str          # window_expired | k_successes_reached | time_end_reached
    selected_clients:   List[int]
    attempted_clients:  List[int]
    successful_clients: List[int]
    collided_clients:   List[int]
    blocked_clients:    List[int]
    successful_updates: Dict[int, bytes]
    num_attempts:       int
    num_selected:       int
    num_successes:      int
    num_collisions:     int
    num_blocked:        int
    total_ul_airtime_s: float
    num_packet_attempts: int = 0
    num_packet_successes: int = 0
    num_packet_collisions: int = 0
    num_energy_blocked: int = 0

@dataclass
class DownlinkResult:
    start_s:                   float
    end_s:                     float
    airtime_s:                 float
    duration_s:                float
    num_fragments:             int
    payload_bytes:             int
    successful_clients:        List[int]
    missed_clients:            List[int]

@dataclass
class ClientMacState:
    num_dl_received:   int   = 0
    num_dl_missed:     int   = 0
    num_ul_attempts:   int   = 0
    num_ul_successes:  int   = 0
    num_ul_collisions: int   = 0
    num_ul_blocked:    int   = 0
    num_energy_blocked: int   = 0
    num_packet_attempts: int  = 0
    num_packet_successes: int = 0
    num_packet_collisions: int = 0
    total_ul_airtime_s: float = 0.0
    total_tx_energy_j: float = 0.0

# ---------------------------------------------------------------------------
