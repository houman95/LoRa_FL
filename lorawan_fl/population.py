"""Communication-only background federations, independent of models/datasets.

The ordinary collector resolves one shared channel for both populations. Its
result is then projected onto the focal federation, retaining the *network*
collection time. Legacy learning/energy records therefore remain focal-only.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from .access_policy import DutyCycleTracker, collect_uplinks
from .channel import AlohaChannel
from .config import ClientMacState, RoundPolicyType


class SortedAlohaChannel(AlohaChannel):
    """Exact strict-overlap law in O(M log M), including unequal durations."""

    def resolve(self) -> set:
        successful = set()
        for transmissions in self._txs.values():
            ordered = sorted(transmissions, key=lambda tx: tx[1])
            furthest_end = float("-inf")
            for index, (packet, start, end) in enumerate(ordered):
                overlaps_previous = start < furthest_end
                overlaps_next = (index + 1 < len(ordered)
                                 and ordered[index + 1][1] < end)
                if not overlaps_previous and not overlaps_next:
                    successful.add(packet)
                furthest_end = max(furthest_end, end)
        return successful


class PopulationCollector:
    """Synchronized equal-traffic interferers with persistent per-device MACs.

IDs [0, focal_clients) belong to the learned federation. Remaining IDs have
only opaque, same-sized uplink payloads. They share p, SF, jitter, duty cycle,
channel law and collection boundaries, but never receive data or model state.
Other federations' downlink transmissions are deliberately not modeled.
"""

    def __init__(self, focal_clients, interferers, duty_cycle_percent,
                 result_dir, payload_bytes):
        if focal_clients < 1 or interferers < 0:
            raise ValueError("Population sizes must be positive/nonnegative")
        self.focal_clients = focal_clients
        self.interferers = interferers
        self.extra_dc = [DutyCycleTracker(duty_cycle_percent) for _ in range(interferers)]
        self.extra_mac = [ClientMacState() for _ in range(interferers)]
        self.payload = bytes(payload_bytes)
        self.path = Path(result_dir) / "network_rounds.csv"
        self.rows = []

    def __call__(self, start, horizon, policy, prepared, dev_dc, client_mac,
                 sf, probability, aloha, jitter, energy_cfg=None, **kwargs):
        if policy.policy_type != RoundPolicyType.ADAPTIVE_COMPLETE:
            raise ValueError("Background population currently requires adaptive_complete")
        if kwargs.get("collision_free_ablation", False):
            raise ValueError("Background population ablation is not implemented")
        if len(dev_dc) != self.focal_clients or len(client_mac) != self.focal_clients:
            raise ValueError("Focal MAC population does not match learning population")
        if any(cid < 0 or cid >= self.focal_clients for cid in prepared):
            raise ValueError("Prepared learning payload contains an interferer ID")
        if energy_cfg and (energy_cfg.max_round_packet_attempts is not None
                           or energy_cfg.max_round_ul_airtime_s is not None):
            raise ValueError("Shared round energy limits need a separate fairness design")
        network_prepared = dict(prepared)
        network_prepared.update({cid: self.payload for cid in range(
            self.focal_clients, self.focal_clients + self.interferers)})
        old_blocked = sum(mac.num_energy_blocked for mac in client_mac)
        full = collect_uplinks(
            start, horizon, policy, network_prepared,
            dev_dc + self.extra_dc, client_mac + self.extra_mac,
            sf, probability, SortedAlohaChannel(aloha.num_channels), jitter,
            energy_cfg, **kwargs)
        focal = self.project(full, lambda cid: cid < self.focal_clients)
        focal.num_energy_blocked = sum(mac.num_energy_blocked for mac in client_mac) - old_blocked
        # Both populations can be empty at low p; advance to avoid an infinite loop.
        if full.collection_end_s == start:
            full.collection_end_s = focal.collection_end_s = min(horizon, start + 1.0)
        for scope, predicate in (
            ("focal", lambda cid: cid < self.focal_clients),
            ("interferer", lambda cid: cid >= self.focal_clients),
            ("network", lambda cid: True),
        ):
            view = self.project(full, predicate)
            txs = view._txs_log
            received_clients = {tx["client_id"] for tx in txs if tx["status"] == "success"}
            by_client = {}
            for tx in txs:
                by_client.setdefault(tx["client_id"], []).append(tx)
            row = {
                "episode_idx": kwargs.get("round_index", 0), "scope": scope,
                "collection_start_s": start, "collection_end_s": full.collection_end_s,
                "horizon_s": horizon,
                "round_ends_after_horizon": int(full.collection_end_s > horizon),
                "selected_clients": view.num_selected,
                "attempted_updates": view.num_attempts,
                "successful_full_updates": view.num_successes,
                "partial_updates": len(received_clients - set(view.successful_clients)),
                "zero_fragment_updates": len(set(view.attempted_clients) - received_clients),
                "fragment_attempts": view.num_packet_attempts,
                "fragment_successes": view.num_packet_successes,
                "fragment_collisions": view.num_packet_collisions,
                "fragment_attempts_by_horizon": sum(tx["start_s"] < horizon for tx in txs),
                "fragment_successes_by_horizon": sum(tx["end_s"] <= horizon and tx["status"] == "success" for tx in txs),
                "fragment_collisions_by_horizon": sum(tx["end_s"] <= horizon and tx["status"] == "collision" for tx in txs),
                "fragments_in_flight_at_horizon": sum(tx["start_s"] < horizon < tx["end_s"] for tx in txs),
                "full_updates_by_horizon": sum(all(tx["end_s"] <= horizon and tx["status"] == "success" for tx in packets) for packets in by_client.values()),
                "ul_airtime_s": view.total_ul_airtime_s,
                "tx_energy_j": view.total_ul_airtime_s * (energy_cfg.tx_power_w if energy_cfg else 0.1),
            }
            self.rows.append(row)
            exists = self.path.exists()
            with self.path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        return focal

    @staticmethod
    def project(full, predicate):
        changes = {}
        for field in ("selected", "attempted", "successful", "collided", "blocked"):
            changes[field + "_clients"] = [cid for cid in getattr(full, field + "_clients")
                                            if predicate(cid)]
        for count, field in (("selected", "selected"), ("attempts", "attempted"),
                             ("successes", "successful"), ("collisions", "collided"),
                             ("blocked", "blocked")):
            changes["num_" + count] = len(changes[field + "_clients"])
        txs = [tx for tx in full._txs_log if predicate(tx["client_id"])]
        changes.update(
            successful_updates={cid: blob for cid, blob in full.successful_updates.items()
                                if predicate(cid)},
            total_ul_airtime_s=sum(tx["airtime_s"] for tx in txs),
            num_packet_attempts=len(txs),
            num_packet_successes=sum(tx["status"] == "success" for tx in txs),
            num_packet_collisions=sum(tx["status"] == "collision" for tx in txs),
            num_energy_blocked=0,
        )
        projected = replace(full, **changes)
        projected._txs_log = txs
        return projected

    def summary(self):
        result = {"num_learning_clients": self.focal_clients,
                  "num_interferers": self.interferers,
                  "num_network_clients": self.focal_clients + self.interferers,
                  "legacy_metric_scope": "focal_federation",
                  "horizon_semantics": "finish_started_round_like_EXP018"}
        for scope in ("focal", "interferer", "network"):
            rows = [row for row in self.rows if row["scope"] == scope]
            for field in ("attempted_updates", "successful_full_updates", "fragment_attempts",
                          "fragment_successes", "fragment_collisions", "ul_airtime_s", "tx_energy_j",
                          "fragment_attempts_by_horizon", "fragment_successes_by_horizon",
                          "fragment_collisions_by_horizon", "fragments_in_flight_at_horizon",
                          "full_updates_by_horizon"):
                result[f"{scope}_{field}"] = sum(row[field] for row in rows)
        return result
