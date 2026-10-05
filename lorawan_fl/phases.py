"""Protocol phases that connect learning, compression, access, and aggregation."""

from __future__ import annotations

import copy
import hashlib
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .access_policy import DutyCycleTracker
from .aggregation import (
    AggregationPolicy,
    aggregate_reconstructed_updates,
    parse_aggregation_policy,
)
from .class_b import (
    _downlink_fragment_sizes,
    _schedule_gateway_downlink_fragments,
)
from .compression import (
    compress_update,
    compress_update_with_information,
    decompress_update,
    split_flat_gradient,
)
from .config import ClientMacState, CompressionConfig, DownlinkResult
from .fragmentation import (
    FragmentReception,
    fragment_update,
    receptions_from_delivered_indices,
    reconstruct_partial_update,
)
from .lora_phy import LoRa_MTU
from .model import set_parameters, train
from .update_types import ReconstructedUpdate

def run_downlink_phase(
    sim_time_s:        float,
    global_model:      nn.Module,
    dev_addrs:         List[int],
    gw_dc:             DutyCycleTracker,
    round_ready_time:  float,
    spreading_factor:  int,
    FEC_rate:          float,
    ping_periodicity:  int,
    compression_cfg:   CompressionConfig,
    num_clients:       int,
    client_mac:        List[ClientMacState],
    ideal_downlink:    bool = False,
) -> Tuple[DownlinkResult, float, nn.Module]:
    """
    Schedule one gateway downlink dissemination phase.

    Downlink is modeled as a reliable broadcast: no uplink happens during this
    phase, no downlink collisions/losses are applied, and all clients receive
    the global model after the final gateway fragment is transmitted. The only
    downlink communication constraint modeled here is gateway duty cycle.

    With ``ideal_downlink=True``, the same reliable, complete global-model
    delivery occurs immediately at round readiness.  Gateway packetization,
    airtime, duty-cycle waiting, and downlink delay are removed; no loss model
    is removed because this simulator does not model downlink loss.

    Returns (DownlinkResult, new_sim_time_s, global_model_snapshot).
    """
    frag_size = LoRa_MTU(spreading_factor)
    blob, dl_bytes, _, _ = compress_update(global_model, None, compression_cfg)
    dl_frag_sizes = _downlink_fragment_sizes(dl_bytes, frag_size, FEC_rate)
    ready_time = max(sim_time_s, round_ready_time)
    if ideal_downlink:
        dl_start = ready_time
        dl_end = ready_time
        dl_airtime = 0.0
        actual_num_fragments = 0
    else:
        dl_start, dl_end, dl_airtime, _ = _schedule_gateway_downlink_fragments(
            ready_time,
            gw_dc,
            dl_frag_sizes,
            spreading_factor,
        )
        actual_num_fragments = len(dl_frag_sizes)

    successful = list(range(num_clients))
    missed: List[int] = []
    for cid in range(num_clients):
        client_mac[cid].num_dl_received += 1

    model_snapshot = copy.deepcopy(global_model)

    result = DownlinkResult(
        start_s=dl_start, end_s=dl_end, airtime_s=dl_airtime,
        duration_s=dl_end - dl_start, num_fragments=actual_num_fragments,
        payload_bytes=dl_bytes,
        successful_clients=successful, missed_clients=missed)
    return result, dl_end, model_snapshot

# ---------------------------------------------------------------------------
# Phase 2 — Local training + compression
# ---------------------------------------------------------------------------
def run_local_training_phase(
    client_ids:   List[int],
    client_model: nn.Module,
    reference:    nn.Module,
    trainloaders,
    compression_cfg: CompressionConfig,
    spreading_factor: int,
    local_epochs: int = 1,
    local_lr: float = 0.01,
    delivered_fragment_indices: Dict[int, List[int]] | None = None,
    num_fragments: int | None = None,
    coordinate_wise_fragment_normalization: bool = False,
    fragment_receptions_by_client: Dict[int, List[FragmentReception]] | None = None,
    aggregation_policy: str | AggregationPolicy | None = None,
    training_seed_base: int | None = None,
    episode_idx: int = 0,
) -> Dict[int, bytes | torch.Tensor | ReconstructedUpdate]:
    """Train clients and return complete blobs or fragment-masked tensors."""
    if fragment_receptions_by_client is not None and delivered_fragment_indices is not None:
        raise ValueError(
            "use fragment_receptions_by_client or delivered_fragment_indices, not both"
        )
    if (delivered_fragment_indices is None) != (num_fragments is None):
        raise ValueError(
            "delivered_fragment_indices and num_fragments must be provided together"
        )
    if (
        coordinate_wise_fragment_normalization
        and delivered_fragment_indices is None
        and fragment_receptions_by_client is None
    ):
        raise ValueError(
            "coordinate-wise fragment normalization requires fragment delivery masks"
        )
    mtu = LoRa_MTU(spreading_factor)
    resolved_policy = (
        parse_aggregation_policy(aggregation_policy)
        if aggregation_policy is not None
        else AggregationPolicy.MASKED_PARTIAL
        if coordinate_wise_fragment_normalization
        else AggregationPolicy.LEGACY_ZERO_FILL
    )
    prepared: Dict[int, bytes | torch.Tensor | ReconstructedUpdate] = {}
    for cid in client_ids:
        if training_seed_base is not None:
            client_seed = int(training_seed_base + episode_idx * 100_003 + cid)
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(client_seed)
        set_parameters(client_model, reference)
        train(client_model, trainloaders[cid], epochs=local_epochs, lr=local_lr)
        blob, blob_bytes, _, _ = compress_update(
            client_model, reference, compression_cfg)
        if blob_bytes > compression_cfg.max_payload_bytes:
            print(
                f"    [WARN] cid={cid}: payload {blob_bytes}B exceeds "
                f"fragmentation budget {compression_cfg.max_payload_bytes}B, skipping")
            continue
        if (
            delivered_fragment_indices is None
            and fragment_receptions_by_client is None
        ):
            prepared[cid] = blob
        else:
            if fragment_receptions_by_client is not None:
                receptions = fragment_receptions_by_client.get(cid)
                if receptions is None:
                    raise ValueError(f"missing fragment metadata for client {cid}")
            else:
                receptions = receptions_from_delivered_indices(
                    client_id=cid,
                    round_index=0,
                    total_fragments=int(num_fragments),
                    delivered_fragment_indices=delivered_fragment_indices.get(cid, []),
                )
            reconstructed = reconstruct_partial_update(
                blob, compression_cfg.apply_zlib, receptions
            )
            # Both partial baselines use the same reconstruction.  The legacy
            # policy ignores its mask only inside the aggregation module.
            if resolved_policy == AggregationPolicy.DISCARD_ALL:
                if reconstructed.complete_reception:
                    prepared[cid] = reconstructed
            else:
                prepared[cid] = reconstructed
    return prepared


def _cosine_to_oracle(
    actual: torch.Tensor, oracle: torch.Tensor
) -> float | None:
    denominator = float(
        torch.linalg.vector_norm(actual).item()
        * torch.linalg.vector_norm(oracle).item()
    )
    if denominator <= torch.finfo(torch.float32).tiny:
        return None
    return float(torch.clamp(torch.dot(actual, oracle) / denominator, -1.0, 1.0).item())


def _aggregate_quality_metrics(
    actual: torch.Tensor,
    oracle: torch.Tensor,
) -> Dict[str, float | None]:
    oracle_norm = float(torch.linalg.vector_norm(oracle).item())
    actual_norm = float(torch.linalg.vector_norm(actual).item())
    error = float(torch.linalg.vector_norm(actual - oracle).item())
    oracle_nonzero = oracle != 0
    sign_disagreement = (
        float(
            (torch.sign(actual[oracle_nonzero]) != torch.sign(oracle[oracle_nonzero]))
            .float()
            .mean()
            .item()
        )
        if bool(torch.any(oracle_nonzero).item())
        else None
    )
    denominator = max(oracle_norm, torch.finfo(torch.float32).tiny)
    return {
        "aggregate_cosine_to_oracle": _cosine_to_oracle(actual, oracle),
        "aggregate_norm_ratio_to_oracle": actual_norm / denominator,
        "aggregate_relative_l2_error": error / denominator,
        "aggregate_sign_disagreement_fraction": sign_disagreement,
        "aggregate_norm": actual_norm,
        "oracle_norm": oracle_norm,
    }


def run_oracle_diagnostic_local_training_phase(
    client_ids:   List[int],
    client_model: nn.Module,
    reference:    nn.Module,
    trainloaders,
    compression_cfg: CompressionConfig,
    spreading_factor: int,
    local_epochs: int,
    local_lr: float,
    fragment_receptions_by_client: Dict[int, List[FragmentReception]],
    aggregation_policy: str | AggregationPolicy,
    training_seed_base: int,
    episode_idx: int,
) -> tuple[
    Dict[int, ReconstructedUpdate],
    Dict[str, float | int | str],
    List[Dict[str, float | int]],
    List[Dict[str, float | str]],
]:
    """Train selected clients once and derive read-only oracle diagnostics.

    The full-information oracle is the equal-client average of the complete
    compressed updates already produced for this round.  It is never applied
    to the model.  All diagnostics below are deterministic tensor reductions;
    an exact Torch RNG-state check guards the non-invasive requirement.
    """

    resolved_policy = parse_aggregation_policy(aggregation_policy)
    reconstructed: Dict[int, ReconstructedUpdate] = {}
    complete: Dict[int, torch.Tensor] = {}
    client_rows: List[Dict[str, float | int]] = []
    for cid in sorted(client_ids):
        client_seed = int(training_seed_base + episode_idx * 100_003 + cid)
        torch.manual_seed(client_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(client_seed)
        set_parameters(client_model, reference)
        train(
            client_model, trainloaders[cid],
            epochs=local_epochs, lr=local_lr,
        )
        blob, blob_bytes, _, _ = compress_update(
            client_model, reference, compression_cfg
        )
        if blob_bytes > compression_cfg.max_payload_bytes:
            raise RuntimeError(
                f"client {cid} payload {blob_bytes} exceeds configured budget "
                f"{compression_cfg.max_payload_bytes}"
            )
        receptions = fragment_receptions_by_client.get(cid)
        if receptions is None:
            raise ValueError(f"missing fragment metadata for client {cid}")
        partial = reconstruct_partial_update(
            blob, compression_cfg.apply_zlib, receptions
        )
        full = decompress_update(blob, compression_cfg.apply_zlib)
        reconstructed[cid] = partial
        complete[cid] = full

        received_indices = {
            item.identity.fragment_index for item in receptions if item.received
        }
        fragments = fragment_update(
            blob, compression_cfg.apply_zlib, cid, episode_idx, len(receptions)
        )
        full_nonzero = int(torch.count_nonzero(full).item())
        received_nonzero = 0
        received_energy = 0.0
        for fragment in fragments:
            if fragment.identity.fragment_index not in received_indices:
                continue
            nonzero = fragment.coordinate_values != 0
            received_nonzero += int(torch.count_nonzero(nonzero).item())
            received_energy += float(
                torch.sum(fragment.coordinate_values.float() ** 2).item()
            )
        full_energy = float(torch.sum(full.float() ** 2).item())
        client_rows.append({
            "client_id": cid,
            "fraction_fragments_received": (
                len(received_indices) / max(1, len(receptions))
            ),
            "fraction_retained_nonzero_entries_received": (
                received_nonzero / max(1, full_nonzero)
            ),
            "fraction_update_squared_norm_received": (
                received_energy / max(full_energy, torch.finfo(torch.float32).tiny)
            ),
        })

    rng_before = torch.random.get_rng_state().clone()
    oracle = torch.stack([complete[cid] for cid in sorted(complete)]).mean(dim=0)
    actual_result = aggregate_reconstructed_updates(
        reconstructed, resolved_policy
    )
    discard_result = aggregate_reconstructed_updates(
        reconstructed, AggregationPolicy.DISCARD_ALL
    )
    masked_result = aggregate_reconstructed_updates(
        reconstructed, AggregationPolicy.MASKED_PARTIAL
    )

    partial_clients = sum(
        0 < update.received_fragments < update.total_fragments
        for update in reconstructed.values()
    )
    exact_gate_checked = int(partial_clients == 0)
    exact_gate_passed = int(
        not exact_gate_checked
        or (
            torch.equal(discard_result.aggregated_delta, masked_result.aggregated_delta)
            and torch.equal(
                discard_result.coverage.contributors_per_coordinate,
                masked_result.coverage.contributors_per_coordinate,
            )
        )
    )
    if exact_gate_checked and not exact_gate_passed:
        raise RuntimeError(
            "exact full/zero-reception equivalence gate failed during simulation"
        )

    values = list(client_rows)
    partial_values = [
        row for row, update in zip(values, reconstructed.values())
        if 0 < update.received_fragments < update.total_fragments
    ]
    valid_fractions = [
        float(update.valid_mask.float().mean().item())
        for update in reconstructed.values()
    ]
    update_hasher = hashlib.sha256()
    seed_hasher = hashlib.sha256()
    for cid in sorted(complete):
        update_hasher.update(int(cid).to_bytes(8, "little", signed=False))
        update_hasher.update(complete[cid].contiguous().numpy().tobytes())
        client_seed = int(training_seed_base + episode_idx * 100_003 + cid)
        seed_hasher.update(int(cid).to_bytes(8, "little", signed=False))
        seed_hasher.update(client_seed.to_bytes(8, "little", signed=False))

    diagnostics: Dict[str, float | int | str] = {
        **actual_result.coverage.as_dict(),
        **_aggregate_quality_metrics(actual_result.aggregated_delta, oracle),
        "mean_received_fragment_fraction_per_client": (
            sum(float(row["fraction_fragments_received"]) for row in values)
            / max(1, len(values))
        ),
        "mean_received_fraction_among_partial_clients": (
            sum(float(row["fraction_fragments_received"]) for row in partial_values)
            / len(partial_values)
            if partial_values else None
        ),
        "mean_valid_coordinate_fraction": (
            sum(valid_fractions) / max(1, len(valid_fractions))
        ),
        "exact_full_zero_equivalence_checked": exact_gate_checked,
        "exact_full_zero_equivalence_passed": exact_gate_passed,
        "local_update_set_sha256": update_hasher.hexdigest(),
        "local_training_seed_fingerprint": seed_hasher.hexdigest(),
    }

    layer_rows: List[Dict[str, float | str]] = []
    offset = 0
    for name, parameter in reference.named_parameters():
        count = parameter.numel()
        layer_rows.append({
            "layer": name,
            **_aggregate_quality_metrics(
                actual_result.aggregated_delta[offset:offset + count],
                oracle[offset:offset + count],
            ),
        })
        offset += count

    rng_after = torch.random.get_rng_state()
    diagnostics["oracle_rng_state_unchanged"] = int(
        torch.equal(rng_before, rng_after)
    )
    if not diagnostics["oracle_rng_state_unchanged"]:
        raise RuntimeError("oracle diagnostics altered Torch RNG state")

    actual_updates = {
        cid: update for cid, update in reconstructed.items()
        if resolved_policy != AggregationPolicy.DISCARD_ALL
        or update.complete_reception
    }
    return actual_updates, diagnostics, client_rows, layer_rows


def _population_heterogeneity(
    vectors: List[torch.Tensor],
) -> tuple[torch.Tensor, float]:
    """Return a population mean and mean squared distance from that mean."""
    if not vectors:
        return torch.zeros(0, dtype=torch.float32), float("nan")
    stacked = torch.stack([vector.detach().cpu().float() for vector in vectors])
    mean_vector = stacked.mean(dim=0)
    heterogeneity = torch.mean(
        torch.sum((stacked - mean_vector) ** 2, dim=1)
    ).item()
    return mean_vector, float(heterogeneity)


def summarize_update_information(
    compressed_vectors: Dict[int, torch.Tensor],
    uncompressed_vectors: Dict[int, torch.Tensor],
    successful_client_ids: List[int],
    retained_entries: List[int],
    update_dimensions: List[int],
    retained_energy_ratios: List[float],
) -> Dict[str, float | int]:
    """Summarize participant and successful-subset update information."""
    participant_ids = sorted(compressed_vectors)
    compressed = [compressed_vectors[cid] for cid in participant_ids]
    uncompressed = [uncompressed_vectors[cid] for cid in participant_ids]
    compressed_mean, compressed_hx = _population_heterogeneity(compressed)
    uncompressed_mean, uncompressed_hx = _population_heterogeneity(uncompressed)
    compression_distortion = (
        float(torch.sum((compressed_mean - uncompressed_mean) ** 2).item())
        if compressed and uncompressed else float("nan")
    )

    accepted_ids = [
        cid for cid in successful_client_ids if cid in compressed_vectors
    ]
    accepted = [compressed_vectors[cid] for cid in accepted_ids]
    if accepted:
        successful_mean = torch.stack(accepted).mean(dim=0)
        successful_mean_norm = float(
            torch.linalg.vector_norm(successful_mean).item()
        )
        sampling_error = float(
            torch.sum((successful_mean - compressed_mean) ** 2).item()
        )
        total_error = float(
            torch.sum((successful_mean - uncompressed_mean) ** 2).item()
        )
    else:
        successful_mean_norm = float("nan")
        sampling_error = float("nan")
        total_error = float("nan")

    retained_k = (
        int(round(sum(retained_entries) / len(retained_entries)))
        if retained_entries else 0
    )
    update_dimension = (
        int(round(sum(update_dimensions) / len(update_dimensions)))
        if update_dimensions else 0
    )
    retained_fraction = (
        retained_k / update_dimension if update_dimension > 0 else float("nan")
    )
    retained_energy_mean = (
        sum(retained_energy_ratios) / len(retained_energy_ratios)
        if retained_energy_ratios else float("nan")
    )

    return {
        "information_population_size": len(participant_ids),
        "retained_entries": retained_k,
        "update_dimension": update_dimension,
        "retained_fraction": retained_fraction,
        "topk_retained_energy_ratio_mean": retained_energy_mean,
        "compressed_update_heterogeneity": compressed_hx,
        "uncompressed_update_heterogeneity": uncompressed_hx,
        "attempted_compressed_update_heterogeneity": compressed_hx,
        "attempted_uncompressed_update_heterogeneity": uncompressed_hx,
        "aggregate_compression_distortion_sq": compression_distortion,
        "participating_compressed_mean_norm": (
            float(torch.linalg.vector_norm(compressed_mean).item())
            if compressed else float("nan")
        ),
        "participating_uncompressed_mean_norm": (
            float(torch.linalg.vector_norm(uncompressed_mean).item())
            if uncompressed else float("nan")
        ),
        "successful_compressed_mean_norm": successful_mean_norm,
        "aggregate_sampling_error_sq": sampling_error,
        "communication_subset_error_sq": sampling_error,
        "total_aggregate_error_sq": total_error,
    }


def run_instrumented_local_training_phase(
    client_ids,
    successful_client_ids,
    client_model,
    reference,
    trainloaders,
    compression_cfg: CompressionConfig,
    spreading_factor: int,
    local_epochs: int,
    local_lr: float,
    training_seed_base: int,
    episode_idx: int,
):
    """
    Train every attempted participant for information instrumentation.

    Only complete successful clients are returned for aggregation. Stable
    per-client/per-episode Torch seeds make matched runs independent of client
    execution order.
    """
    compressed_updates: Dict[int, bytes] = {}
    compressed_vectors: Dict[int, torch.Tensor] = {}
    uncompressed_vectors: Dict[int, torch.Tensor] = {}
    retained_entries: List[int] = []
    update_dimensions: List[int] = []
    retained_energy_ratios: List[float] = []

    for cid in sorted(client_ids):
        client_seed = int(training_seed_base + episode_idx * 100_003 + cid)
        torch.manual_seed(client_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(client_seed)
        set_parameters(client_model, reference)
        train(
            client_model,
            trainloaders[cid],
            epochs=local_epochs,
            lr=local_lr,
        )
        (
            blob,
            blob_bytes,
            retained_k,
            update_dimension,
            full_vector,
            retained_energy_ratio,
        ) = compress_update_with_information(
            client_model, reference, compression_cfg
        )
        if blob_bytes > compression_cfg.max_payload_bytes:
            print(
                f"    [WARN] cid={cid}: payload {blob_bytes}B exceeds "
                f"fragmentation budget {compression_cfg.max_payload_bytes}B, skipping"
            )
            continue
        compressed_updates[cid] = blob
        compressed_vectors[cid] = decompress_update(
            blob, compression_cfg.apply_zlib
        )
        uncompressed_vectors[cid] = full_vector
        retained_entries.append(retained_k)
        update_dimensions.append(update_dimension)
        retained_energy_ratios.append(retained_energy_ratio)

    successful_updates = {
        cid: compressed_updates[cid]
        for cid in successful_client_ids
        if cid in compressed_updates
    }
    information = summarize_update_information(
        compressed_vectors,
        uncompressed_vectors,
        successful_client_ids,
        retained_entries,
        update_dimensions,
        retained_energy_ratios,
    )
    return successful_updates, information

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_successful_updates(
    global_model:     nn.Module,
    successful_updates: Dict[int, bytes | torch.Tensor | ReconstructedUpdate],
    apply_zlib:       bool,
    server_lr:         float = 1.0,
    coordinate_wise_fragment_normalization: bool = False,
    aggregation_policy: str | AggregationPolicy | None = None,
    client_weights: Dict[int, float] | None = None,
    coverage_diagnostics: Dict[str, float | int] | None = None,
) -> bool:
    if not successful_updates:
        return False
    resolved_policy = (
        parse_aggregation_policy(aggregation_policy)
        if aggregation_policy is not None
        else AggregationPolicy.MASKED_PARTIAL
        if coordinate_wise_fragment_normalization
        else AggregationPolicy.DISCARD_ALL
    )
    decoded = {
        client_id: (
            update
            if isinstance(update, (torch.Tensor, ReconstructedUpdate))
            else decompress_update(update, apply_zlib)
        )
        for client_id, update in successful_updates.items()
    }
    result = aggregate_reconstructed_updates(
        decoded,
        resolved_policy,
        client_weights=client_weights,
    )
    if result.contributing_clients == 0:
        return False
    avg_flat = result.aggregated_delta
    if coverage_diagnostics is not None:
        coverage_diagnostics.update(result.coverage.as_dict())
    layers   = split_flat_gradient(avg_flat, global_model)
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if name in layers:
                param.data.add_(server_lr * layers[name].to(param.device))
    return True
