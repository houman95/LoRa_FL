"""Duty-cycle-aware transmitters and uplink collection policies."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .channel import AlohaChannel
from .config import (
    DC_PERCENT,
    ClientMacState,
    EnergyConfig,
    RoundPolicyConfig,
    RoundPolicyType,
    UplinkCollectionResult,
    UplinkTx,
)
from .energy import energy_budget_allows, record_packet_energy, txs_airtime_s
from .lora_phy import LoRa_frame_duration
from .timing import (
    UplinkRandomStreams,
    UplinkTimingConfig,
    schedule_fragment_times,
    timing_config_from_legacy,
)

class DutyCycleTracker:
    def __init__(self, dc_percent: float = DC_PERCENT):
        self.dc_percent  = dc_percent
        self.next_free_s = 0.0

    def can_transmit(self, now_s: float) -> bool:
        return now_s >= self.next_free_s

    def record_tx(self, now_s: float, airtime_s: float) -> None:
        toff = airtime_s * (100.0 / self.dc_percent - 1.0)
        self.next_free_s = now_s + airtime_s + toff

    def earliest_tx(self) -> float:
        return self.next_free_s

def _split_payload(blob: bytes, mtu: int) -> List[bytes]:
    return [blob[i:i + mtu] for i in range(0, len(blob), mtu)] or [b""]


def _tx_next_free_s(start_s: float, airtime_s: float, dc_percent: float) -> float:
    return start_s + airtime_s * (100.0 / dc_percent)


def _schedule_uplink_update(
    cid:              int,
    blob:             bytes,
    collection_start: float,
    deadline_s:       Optional[float],
    dev_dc:           List[DutyCycleTracker],
    spreading_factor: int,
    contention_jitter_s: float,
    num_uplink_channels: int,
    fragment_jitter_mode: str = "initial_only",
    fragment_jitter_rng: Optional[random.Random] = None,
    fragment_jitter_window_s: float = 0.0,
    timing_config: Optional[UplinkTimingConfig] = None,
    rng_streams: Optional[UplinkRandomStreams] = None,
    round_index: int = 0,
) -> Tuple[List[UplinkTx], str]:
    """
    Attempt to schedule all uplink fragments for one client update.

    The transmitter treats the payload as opaque bytes. It only knows the MTU,
    airtime law, channel set, duty-cycle clock, and collection deadline.
    """
    from .lora_phy import LoRa_MTU

    fragments = _split_payload(blob, LoRa_MTU(spreading_factor))
    airtimes = [LoRa_frame_duration(spreading_factor, len(fragment))
                for fragment in fragments]
    earliest = max(collection_start, dev_dc[cid].earliest_tx())
    timing = timing_config or timing_config_from_legacy(
        contention_jitter_s,
        fragment_jitter_mode,
        fragment_jitter_window_s,
    )
    initial_rng = rng_streams.initial_jitter if rng_streams else random
    if fragment_jitter_rng is not None:
        fragment_rng = fragment_jitter_rng
    elif rng_streams is not None:
        fragment_rng = rng_streams.fragment_jitter
    else:
        fragment_rng = random
    channel_rng = rng_streams.channel if rng_streams else random
    timings = schedule_fragment_times(
        earliest_start_s=earliest,
        airtimes_s=airtimes,
        duty_cycle_percent=dev_dc[cid].dc_percent,
        config=timing,
        initial_jitter_rng=initial_rng,
        fragment_jitter_rng=fragment_rng,
        deadline_s=deadline_s,
    )
    if timings is None:
        return [], "deadline_missed"
    if not dev_dc[cid].can_transmit(timings[0].start_s):
        return [], "blocked"

    txs: List[UplinkTx] = []
    num_fragments = len(fragments)
    for fragment_idx, (fragment, airtime_s, fragment_timing) in enumerate(
        zip(fragments, airtimes, timings)
    ):
        packet_id = f"{round_index}:{cid}:{fragment_idx}"
        txs.append(UplinkTx(
            cid=cid,
            packet_id=packet_id,
            round_index=round_index,
            fragment_idx=fragment_idx,
            num_fragments=num_fragments,
            start_s=fragment_timing.start_s,
            end_s=fragment_timing.end_s,
            channel_idx=channel_rng.randrange(num_uplink_channels),
            airtime_s=airtime_s,
            payload_bytes=len(fragment),
            update_blob=blob,
            earliest_allowed_s=fragment_timing.earliest_allowed_s,
            initial_jitter_s=fragment_timing.initial_jitter_s,
            fragment_jitter_s=fragment_timing.fragment_jitter_s,
            preceding_fragment_jitter_s=fragment_timing.fragment_jitter_s,
        ))

    return txs, None


def _select_bernoulli_participants(
    eligible: List[int],
    participation_probability: float,
    rng: random.Random | None = None,
) -> List[int]:
    p = min(1.0, max(0.0, participation_probability))
    participant_rng = rng or random
    return [cid for cid in eligible if participant_rng.random() < p]

# ---------------------------------------------------------------------------
# Phase 3a — Adaptive complete collection
# ---------------------------------------------------------------------------
def collect_adaptive_complete(
    collection_start_s: float,
    T_end_s:           float,
    policy:            RoundPolicyConfig,
    prepared_updates:  Dict[int, bytes],
    dev_dc:            List[DutyCycleTracker],
    client_mac:        List[ClientMacState],
    spreading_factor:  int,
    participation_probability: float,
    aloha:             AlohaChannel,
    contention_jitter_s: float,
    energy_cfg:        Optional[EnergyConfig] = None,
    collision_free_ablation: bool = False,
    fragment_jitter_mode: str = "initial_only",
    fragment_jitter_rng: Optional[random.Random] = None,
    fragment_jitter_window_s: float = 0.0,
    timing_config: Optional[UplinkTimingConfig] = None,
    rng_streams: Optional[UplinkRandomStreams] = None,
    round_index: int = 0,
) -> UplinkCollectionResult:
    """
    Let selected clients finish their fragment trains under duty-cycle limits.

    There is no fixed aggregation timeout here. Any fragmentation penalty comes
    from airtime, duty-cycle waiting, collisions/loss of any fragment, and the
    resulting wall-clock duration of the round.

    ``collision_free_ablation`` implements the collision-free,
    timing-preserving ablation: transmission offsets, channel assignment,
    airtime, duty-cycle bookkeeping, and round timing above this point are
    completely unaffected by this flag (they only depend on the Python
    ``random`` draws already consumed while scheduling ``txs``, not on the
    channel outcome). Only the *decoding* outcome changes. ``collided_clients``
    / ``num_collisions`` / ``num_packet_collisions`` always reflect the true
    physical overlap (so overlap is still logged even under the ablation),
    while ``successful_clients`` / ``num_successes`` / ``successful_updates`` /
    ``num_packet_successes`` reflect what is *accepted* for aggregation, which
    under the ablation includes clients that physically collided.
    """
    aloha.reset()
    eligible = list(prepared_updates.keys())
    to_tx = _select_bernoulli_participants(
        eligible,
        participation_probability,
        rng_streams.participation if rng_streams else None,
    )

    txs: List[UplinkTx] = []
    blocked: List[int] = []
    energy_blocked: List[int] = []
    round_packet_attempts = 0
    round_ul_airtime_s = 0.0

    for cid in to_tx:
        update_txs, reason = _schedule_uplink_update(
            cid, prepared_updates[cid], collection_start_s, None,
            dev_dc, spreading_factor, contention_jitter_s, aloha.num_channels,
            fragment_jitter_mode=fragment_jitter_mode,
            fragment_jitter_rng=fragment_jitter_rng,
            fragment_jitter_window_s=fragment_jitter_window_s,
            timing_config=timing_config,
            rng_streams=rng_streams,
            round_index=round_index)
        if not update_txs:
            blocked.append(cid)
            client_mac[cid].num_ul_blocked += 1
            continue
        if not energy_budget_allows(
            client_mac[cid], update_txs, energy_cfg,
            round_packet_attempts, round_ul_airtime_s):
            energy_blocked.append(cid)
            client_mac[cid].num_energy_blocked += 1
            continue
        txs.extend(update_txs)
        round_packet_attempts += len(update_txs)
        round_ul_airtime_s += txs_airtime_s(update_txs)
        for tx in update_txs:
            aloha.add_transmission(tx.packet_id, tx.channel_idx, tx.start_s, tx.end_s)
        dev_dc[cid].record_tx(update_txs[-1].start_s, update_txs[-1].airtime_s)
        client_mac[cid].num_ul_attempts += 1

    real_successful_packets = aloha.resolve()
    attempted = sorted({tx.cid for tx in txs})
    real_successful = sorted([
        cid for cid in attempted
        if all(tx.packet_id in real_successful_packets for tx in txs if tx.cid == cid)
    ])
    collided = sorted(set(attempted) - set(real_successful))

    if collision_free_ablation:
        # Ablation condition: the schedule/timing above is untouched; only the
        # decoding decision changes. Overlapping fragments are still declared
        # successfully received, so nothing is excluded from aggregation.
        accepted_packets = {tx.packet_id for tx in txs}
        successful = attempted
    else:
        accepted_packets = real_successful_packets
        successful = real_successful

    for cid in successful: client_mac[cid].num_ul_successes += 1
    for cid in collided:   client_mac[cid].num_ul_collisions += 1
    for cid in attempted:
        record_packet_energy(
            client_mac[cid],
            [tx for tx in txs if tx.cid == cid],
            real_successful_packets,
            energy_cfg,
        )

    collection_end_s = max([tx.end_s for tx in txs], default=collection_start_s)
    result = UplinkCollectionResult(
        policy_name        = "adaptive_complete",
        collection_start_s = collection_start_s,
        collection_end_s   = collection_end_s,
        deadline_s         = collection_end_s,
        trigger_reason     = "all_selected_finished" if txs else "no_selected_clients",
        selected_clients   = sorted(to_tx),
        attempted_clients  = attempted,
        successful_clients = successful,
        collided_clients   = collided,
        blocked_clients    = blocked,
        successful_updates = {
            cid: next(tx.update_blob for tx in txs if tx.cid == cid)
            for cid in successful
        },
        num_selected       = len(to_tx),
        num_attempts       = len(attempted),
        num_successes      = len(successful),
        num_collisions     = len(collided),
        num_blocked        = len(blocked),
        num_energy_blocked = len(energy_blocked),
        total_ul_airtime_s = sum(tx.airtime_s for tx in txs),
        num_packet_attempts = len(txs),
        num_packet_successes = sum(1 for tx in txs if tx.packet_id in accepted_packets),
        num_packet_collisions = sum(1 for tx in txs if tx.packet_id not in real_successful_packets),
    )
    result._txs_log = [
        {
            "client_id": tx.cid,
            "packet_id": tx.packet_id,
            "round_index": tx.round_index,
            "fragment_idx": tx.fragment_idx,
            "num_fragments": tx.num_fragments,
            "coordinate_mapping": "sorted_sparse_records_even_shards_v1",
            "start_s": tx.start_s,
            "end_s": tx.end_s,
            "earliest_allowed_s": tx.earliest_allowed_s,
            "airtime_s": tx.airtime_s,
            "channel_idx": tx.channel_idx,
            "payload_bytes": tx.payload_bytes,
            "initial_jitter_s": tx.initial_jitter_s,
            "fragment_jitter_s": tx.fragment_jitter_s,
            "preceding_fragment_jitter_s": tx.preceding_fragment_jitter_s,
            "status": "success" if tx.packet_id in real_successful_packets else "collision",
        }
        for tx in txs
    ]
    return result

# ---------------------------------------------------------------------------
# Phase 3b — Fixed-window collection
# ---------------------------------------------------------------------------
def collect_fixed_window(
    collection_start_s: float,
    T_end_s:           float,
    policy:            RoundPolicyConfig,
    prepared_updates:  Dict[int, bytes],
    dev_dc:            List[DutyCycleTracker],
    client_mac:        List[ClientMacState],
    spreading_factor:  int,
    participation_probability: float,
    aloha:             AlohaChannel,
    contention_jitter_s: float,
    energy_cfg:        Optional[EnergyConfig] = None,
    collision_free_ablation: bool = False,
    fragment_jitter_mode: str = "initial_only",
    fragment_jitter_rng: Optional[random.Random] = None,
    fragment_jitter_window_s: float = 0.0,
    timing_config: Optional[UplinkTimingConfig] = None,
    rng_streams: Optional[UplinkRandomStreams] = None,
    round_index: int = 0,
) -> UplinkCollectionResult:
    """Collect uplinks for the full window W; aggregate everyone who arrived.

    See ``collect_adaptive_complete`` for the ``collision_free_ablation``
    contract: it is timing-safe here too because the fixed deadline ``W``
    does not depend on the number of successes.
    """
    deadline_s = min(collection_start_s + policy.window_s, T_end_s)
    aloha.reset()

    eligible = list(prepared_updates.keys())
    to_tx = _select_bernoulli_participants(
        eligible,
        participation_probability,
        rng_streams.participation if rng_streams else None,
    )

    txs: List[UplinkTx] = []
    blocked: List[int]  = []
    energy_blocked: List[int] = []
    round_packet_attempts = 0
    round_ul_airtime_s = 0.0

    for cid in to_tx:
        update_txs, reason = _schedule_uplink_update(
            cid, prepared_updates[cid], collection_start_s, deadline_s,
            dev_dc, spreading_factor, contention_jitter_s, aloha.num_channels,
            fragment_jitter_mode=fragment_jitter_mode,
            fragment_jitter_rng=fragment_jitter_rng,
            fragment_jitter_window_s=fragment_jitter_window_s,
            timing_config=timing_config,
            rng_streams=rng_streams,
            round_index=round_index)
        if not update_txs:
            blocked.append(cid)
            client_mac[cid].num_ul_blocked += 1
            continue
        if not energy_budget_allows(
            client_mac[cid], update_txs, energy_cfg,
            round_packet_attempts, round_ul_airtime_s):
            energy_blocked.append(cid)
            client_mac[cid].num_energy_blocked += 1
            continue
        txs.extend(update_txs)
        round_packet_attempts += len(update_txs)
        round_ul_airtime_s += txs_airtime_s(update_txs)
        for tx in update_txs:
            aloha.add_transmission(tx.packet_id, tx.channel_idx, tx.start_s, tx.end_s)
        dev_dc[cid].record_tx(update_txs[-1].start_s, update_txs[-1].airtime_s)
        client_mac[cid].num_ul_attempts += 1

    real_successful_packets = aloha.resolve()
    attempted = sorted({tx.cid for tx in txs})
    real_successful = sorted([
        cid for cid in attempted
        if all(tx.packet_id in real_successful_packets for tx in txs if tx.cid == cid)
    ])
    collided = sorted(set(attempted) - set(real_successful))

    if collision_free_ablation:
        accepted_packets = {tx.packet_id for tx in txs}
        successful = attempted
    else:
        accepted_packets = real_successful_packets
        successful = real_successful

    for cid in successful: client_mac[cid].num_ul_successes  += 1
    for cid in collided:   client_mac[cid].num_ul_collisions += 1
    for cid in attempted:
        record_packet_energy(
            client_mac[cid],
            [tx for tx in txs if tx.cid == cid],
            real_successful_packets,
            energy_cfg,
        )

    result = UplinkCollectionResult(
        policy_name        = "fixed_window",
        collection_start_s = collection_start_s,
        collection_end_s   = deadline_s,
        deadline_s         = deadline_s,
        trigger_reason     = "window_expired" if deadline_s < T_end_s else "time_end_reached",
        selected_clients   = sorted(to_tx),
        attempted_clients  = attempted,
        successful_clients = successful,
        collided_clients   = collided,
        blocked_clients    = blocked,
        successful_updates = {
            cid: next(tx.update_blob for tx in txs if tx.cid == cid)
            for cid in successful
        },
        num_selected       = len(to_tx),
        num_attempts       = len(attempted),
        num_successes      = len(successful),
        num_collisions     = len(collided),
        num_blocked        = len(blocked),
        num_energy_blocked = len(energy_blocked),
        total_ul_airtime_s = sum(tx.airtime_s for tx in txs),
        num_packet_attempts = len(txs),
        num_packet_successes = sum(1 for tx in txs if tx.packet_id in accepted_packets),
        num_packet_collisions = sum(1 for tx in txs if tx.packet_id not in real_successful_packets),
    )
    result._txs_log = [
        {
            "client_id": tx.cid,
            "packet_id": tx.packet_id,
            "round_index": tx.round_index,
            "fragment_idx": tx.fragment_idx,
            "num_fragments": tx.num_fragments,
            "coordinate_mapping": "sorted_sparse_records_even_shards_v1",
            "start_s": tx.start_s,
            "end_s": tx.end_s,
            "earliest_allowed_s": tx.earliest_allowed_s,
            "airtime_s": tx.airtime_s,
            "channel_idx": tx.channel_idx,
            "payload_bytes": tx.payload_bytes,
            "initial_jitter_s": tx.initial_jitter_s,
            "fragment_jitter_s": tx.fragment_jitter_s,
            "preceding_fragment_jitter_s": tx.preceding_fragment_jitter_s,
            "status": "success" if tx.packet_id in real_successful_packets else "collision",
        }
        for tx in txs
    ]
    return result

# ---------------------------------------------------------------------------
# Phase 3c — Fixed-K-with-timeout collection
# ---------------------------------------------------------------------------
def collect_fixed_k_timeout(
    collection_start_s: float,
    T_end_s:           float,
    policy:            RoundPolicyConfig,
    prepared_updates:  Dict[int, bytes],
    dev_dc:            List[DutyCycleTracker],
    client_mac:        List[ClientMacState],
    spreading_factor:  int,
    participation_probability: float,
    aloha:             AlohaChannel,
    contention_jitter_s: float,
    energy_cfg:        Optional[EnergyConfig] = None,
    collision_free_ablation: bool = False,
    fragment_jitter_mode: str = "initial_only",
    fragment_jitter_rng: Optional[random.Random] = None,
    fragment_jitter_window_s: float = 0.0,
    timing_config: Optional[UplinkTimingConfig] = None,
    rng_streams: Optional[UplinkRandomStreams] = None,
    round_index: int = 0,
) -> UplinkCollectionResult:
    """
    Collect uplinks; close as soon as K successes arrive, or at deadline W.
    Only transmissions ending before the actual close time are counted.

    ``collision_free_ablation`` is not supported here: the early-close time
    ``collection_end_s`` is a function of how many transmissions were decoded
    successfully (the K-th success), so forgiving collisions would itself
    change the round's timing and violate the ablation's core guarantee.
    """
    if collision_free_ablation:
        raise ValueError(
            "collision_free_ablation is not timing-safe for fixed_k_timeout: "
            "the K-th-success stopping time depends on the collision outcome. "
            "Use adaptive_complete or fixed_window for this ablation."
        )
    K          = policy.target_successes_k
    deadline_s = min(collection_start_s + policy.window_s, T_end_s)
    aloha.reset()

    eligible = list(prepared_updates.keys())
    to_tx = _select_bernoulli_participants(
        eligible,
        participation_probability,
        rng_streams.participation if rng_streams else None,
    )

    txs: List[UplinkTx] = []
    blocked: List[int]  = []
    energy_blocked: List[int] = []
    round_packet_attempts = 0
    round_ul_airtime_s = 0.0

    for cid in to_tx:
        update_txs, reason = _schedule_uplink_update(
            cid, prepared_updates[cid], collection_start_s, deadline_s,
            dev_dc, spreading_factor, contention_jitter_s, aloha.num_channels,
            fragment_jitter_mode=fragment_jitter_mode,
            fragment_jitter_rng=fragment_jitter_rng,
            fragment_jitter_window_s=fragment_jitter_window_s,
            timing_config=timing_config,
            rng_streams=rng_streams,
            round_index=round_index)
        if not update_txs:
            blocked.append(cid)
            client_mac[cid].num_ul_blocked += 1
            continue
        if not energy_budget_allows(
            client_mac[cid], update_txs, energy_cfg,
            round_packet_attempts, round_ul_airtime_s):
            energy_blocked.append(cid)
            client_mac[cid].num_energy_blocked += 1
            continue
        txs.extend(update_txs)
        round_packet_attempts += len(update_txs)
        round_ul_airtime_s += txs_airtime_s(update_txs)
        for tx in update_txs:
            aloha.add_transmission(tx.packet_id, tx.channel_idx, tx.start_s, tx.end_s)

    successful_packets = aloha.resolve()

    attempted = sorted({tx.cid for tx in txs})
    successful_update_txs = []
    for cid in attempted:
        cid_txs = [tx for tx in txs if tx.cid == cid]
        if cid_txs and all(tx.packet_id in successful_packets for tx in cid_txs):
            successful_update_txs.append(max(cid_txs, key=lambda tx: tx.end_s))
    successful_update_txs = sorted(successful_update_txs, key=lambda tx: tx.end_s)

    if len(successful_update_txs) >= K:
        accepted_last_txs  = successful_update_txs[:K]
        collection_end_s   = accepted_last_txs[-1].end_s
        trigger_reason     = "k_successes_reached"
    else:
        accepted_last_txs  = successful_update_txs
        collection_end_s   = deadline_s
        trigger_reason     = "window_expired" if deadline_s < T_end_s else "time_end_reached"

    # Only count transmissions that ended before the actual close time
    counted_txs = [tx for tx in txs if tx.end_s <= collection_end_s]
    counted_ids = {tx.cid for tx in counted_txs}
    accepted_ids = {tx.cid for tx in accepted_last_txs}

    # Record DC and MAC state only for counted transmissions
    for cid in counted_ids:
        cid_txs = [tx for tx in counted_txs if tx.cid == cid]
        if cid_txs:
            last_tx = max(cid_txs, key=lambda tx: tx.end_s)
            dev_dc[cid].record_tx(last_tx.start_s, last_tx.airtime_s)
            client_mac[cid].num_ul_attempts += 1
    successful = sorted(accepted_ids)
    collided   = sorted([tx.cid for tx in counted_txs
                         if tx.packet_id not in successful_packets])
    collided = sorted(set(collided) - accepted_ids)

    for cid in successful: client_mac[cid].num_ul_successes  += 1
    for cid in collided:   client_mac[cid].num_ul_collisions += 1
    for cid in counted_ids:
        record_packet_energy(
            client_mac[cid],
            [tx for tx in counted_txs if tx.cid == cid],
            successful_packets,
            energy_cfg,
        )

    result = UplinkCollectionResult(
        policy_name        = f"fixed_k{K}_timeout",
        collection_start_s = collection_start_s,
        collection_end_s   = collection_end_s,
        deadline_s         = deadline_s,
        trigger_reason     = trigger_reason,
        selected_clients   = sorted(to_tx),
        attempted_clients  = sorted(counted_ids),
        successful_clients = successful,
        collided_clients   = collided,
        blocked_clients    = blocked,
        successful_updates = {
            cid: next(tx.update_blob for tx in txs if tx.cid == cid)
            for cid in successful
        },
        num_selected       = len(to_tx),
        num_attempts       = len(counted_ids),
        num_successes      = len(successful),
        num_collisions     = len(collided),
        num_blocked        = len(blocked),
        num_energy_blocked = len(energy_blocked),
        total_ul_airtime_s = sum(tx.airtime_s for tx in counted_txs),
        num_packet_attempts = len(counted_txs),
        num_packet_successes = sum(1 for tx in counted_txs if tx.packet_id in successful_packets),
        num_packet_collisions = sum(1 for tx in counted_txs if tx.packet_id not in successful_packets),
    )
    result._txs_log = [
        {
            "client_id": tx.cid,
            "packet_id": tx.packet_id,
            "round_index": tx.round_index,
            "fragment_idx": tx.fragment_idx,
            "num_fragments": tx.num_fragments,
            "coordinate_mapping": "sorted_sparse_records_even_shards_v1",
            "start_s": tx.start_s,
            "end_s": tx.end_s,
            "earliest_allowed_s": tx.earliest_allowed_s,
            "airtime_s": tx.airtime_s,
            "channel_idx": tx.channel_idx,
            "payload_bytes": tx.payload_bytes,
            "initial_jitter_s": tx.initial_jitter_s,
            "fragment_jitter_s": tx.fragment_jitter_s,
            "preceding_fragment_jitter_s": tx.preceding_fragment_jitter_s,
            "status": (
                "success" if tx.cid in accepted_ids
                else "collision" if tx.packet_id not in successful_packets
                else "not_accepted"
            ),
        }
        for tx in counted_txs
    ]
    return result

def collect_uplinks(
    collection_start_s: float,
    T_end_s:           float,
    policy:            RoundPolicyConfig,
    prepared_updates:  Dict[int, bytes],
    dev_dc:            List[DutyCycleTracker],
    client_mac:        List[ClientMacState],
    spreading_factor:  int,
    participation_probability: float,
    aloha:             AlohaChannel,
    contention_jitter_s: float,
    energy_cfg:        Optional[EnergyConfig] = None,
    collision_free_ablation: bool = False,
    fragment_jitter_mode: str = "initial_only",
    fragment_jitter_rng: Optional[random.Random] = None,
    fragment_jitter_window_s: float = 0.0,
    timing_config: Optional[UplinkTimingConfig] = None,
    rng_streams: Optional[UplinkRandomStreams] = None,
    round_index: int = 0,
) -> UplinkCollectionResult:
    if policy.policy_type == RoundPolicyType.ADAPTIVE_COMPLETE:
        return collect_adaptive_complete(
            collection_start_s, T_end_s, policy, prepared_updates,
            dev_dc, client_mac, spreading_factor, participation_probability,
            aloha, contention_jitter_s, energy_cfg, collision_free_ablation,
            fragment_jitter_mode, fragment_jitter_rng, fragment_jitter_window_s,
            timing_config, rng_streams, round_index)
    if policy.policy_type == RoundPolicyType.FIXED_WINDOW:
        return collect_fixed_window(
            collection_start_s, T_end_s, policy, prepared_updates,
            dev_dc, client_mac, spreading_factor, participation_probability,
            aloha, contention_jitter_s, energy_cfg, collision_free_ablation,
            fragment_jitter_mode, fragment_jitter_rng, fragment_jitter_window_s,
            timing_config, rng_streams, round_index)
    return collect_fixed_k_timeout(
        collection_start_s, T_end_s, policy, prepared_updates,
        dev_dc, client_mac, spreading_factor, participation_probability,
        aloha, contention_jitter_s, energy_cfg, collision_free_ablation,
        fragment_jitter_mode, fragment_jitter_rng, fragment_jitter_window_s,
        timing_config, rng_streams, round_index)

# ---------------------------------------------------------------------------
