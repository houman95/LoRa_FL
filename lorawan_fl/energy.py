"""Energy accounting and optional transmission budgets."""

from __future__ import annotations

from typing import Iterable, Optional

from .config import ClientMacState, EnergyConfig, UplinkTx


def tx_energy_j(airtime_s: float, cfg: EnergyConfig) -> float:
    """Transmit-energy proxy from radio-on airtime."""
    return cfg.tx_power_w * airtime_s


def txs_airtime_s(txs: Iterable[UplinkTx]) -> float:
    return sum(tx.airtime_s for tx in txs)


def txs_energy_j(txs: Iterable[UplinkTx], cfg: EnergyConfig) -> float:
    return sum(tx_energy_j(tx.airtime_s, cfg) for tx in txs)


def energy_budget_allows(
    client_state: ClientMacState,
    txs: list[UplinkTx],
    cfg: Optional[EnergyConfig],
    round_packet_attempts: int = 0,
    round_ul_airtime_s: float = 0.0,
) -> bool:
    """Return whether a candidate fragment train fits all configured budgets."""
    if cfg is None:
        return True

    packet_attempts = len(txs)
    airtime_s = txs_airtime_s(txs)
    energy_j = txs_energy_j(txs, cfg)

    if (
        cfg.max_client_packet_attempts is not None
        and client_state.num_packet_attempts + packet_attempts
        > cfg.max_client_packet_attempts
    ):
        return False
    if (
        cfg.max_client_ul_airtime_s is not None
        and client_state.total_ul_airtime_s + airtime_s
        > cfg.max_client_ul_airtime_s
    ):
        return False
    if (
        cfg.max_client_tx_energy_j is not None
        and client_state.total_tx_energy_j + energy_j
        > cfg.max_client_tx_energy_j
    ):
        return False
    if (
        cfg.max_round_packet_attempts is not None
        and round_packet_attempts + packet_attempts
        > cfg.max_round_packet_attempts
    ):
        return False
    if (
        cfg.max_round_ul_airtime_s is not None
        and round_ul_airtime_s + airtime_s
        > cfg.max_round_ul_airtime_s
    ):
        return False

    return True


def record_packet_energy(
    client_state: ClientMacState,
    txs: list[UplinkTx],
    successful_packet_ids: set[str],
    cfg: Optional[EnergyConfig],
) -> None:
    """Update per-client packet and energy counters for counted transmissions."""
    effective_cfg = cfg if cfg is not None else EnergyConfig()
    client_state.num_packet_attempts += len(txs)
    successes = sum(1 for tx in txs if tx.packet_id in successful_packet_ids)
    client_state.num_packet_successes += successes
    client_state.num_packet_collisions += len(txs) - successes
    client_state.total_ul_airtime_s += txs_airtime_s(txs)
    client_state.total_tx_energy_j += txs_energy_j(txs, effective_cfg)
