"""Experiment orchestration for one realization and policy comparisons."""

from __future__ import annotations

import csv
import json
import math
import os
import time

import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from .access_policy import DutyCycleTracker, collect_uplinks
from .aggregation import AggregationPolicy, policy_from_config
from .channel import AlohaChannel
from .population import PopulationCollector
from .compression import encoded_update_payload_bytes
from .config import (
    ClientMacState,
    CompressionConfig,
    DEVICE,
    EnergyConfig,
    RoundPolicyConfig,
    RoundPolicyType,
)
from .lora_phy import LoRa_MTU
from .fragmentation import (
    fragment_reception_diagnostics,
    receptions_from_transmission_log,
)
from .model import create_model, load_datasets, test, warm_start_to_accuracy
from .phases import (
    aggregate_successful_updates,
    run_downlink_phase,
    run_instrumented_local_training_phase,
    run_local_training_phase,
    run_oracle_diagnostic_local_training_phase,
)
from .results import (
    _open_csv,
    _plot_comparison_results,
    _plot_episode_results,
    _write_experiment_purpose,
    write_directional_summary,
    write_target_metrics,
)
from .utils import gini, set_all_seeds
from .timing import (
    INITIAL_ONLY,
    UplinkRandomStreams,
    UplinkTimingConfig,
    timing_config_from_legacy,
)

UPDATE_INFORMATION_FIELDS = [
    "information_population_size",
    "retained_entries",
    "update_dimension",
    "retained_fraction",
    "topk_retained_energy_ratio_mean",
    "compressed_update_heterogeneity",
    "uncompressed_update_heterogeneity",
    "attempted_compressed_update_heterogeneity",
    "attempted_uncompressed_update_heterogeneity",
    "aggregate_compression_distortion_sq",
    "participating_compressed_mean_norm",
    "participating_uncompressed_mean_norm",
    "successful_compressed_mean_norm",
    "aggregate_sampling_error_sq",
    "communication_subset_error_sq",
    "total_aggregate_error_sq",
]
LABEL_REPRESENTATION_FIELDS = [
    "label_representation_error_sq",
    "successful_label_entropy",
    "successful_classes_represented",
    *[f"attempted_client_mean_label_proportion_{label}" for label in range(10)],
    *[f"successful_client_mean_label_proportion_{label}" for label in range(10)],
]


def _mean_label_proportions(client_ids, client_label_proportions):
    if not client_ids or client_label_proportions is None:
        return None
    vectors = torch.tensor(
        [client_label_proportions[cid] for cid in client_ids],
        dtype=torch.float64,
    )
    return vectors.mean(dim=0)


def _label_representation_metrics(
    attempted_client_ids,
    successful_client_ids,
    client_label_proportions,
):
    attempted_mean = _mean_label_proportions(
        attempted_client_ids, client_label_proportions
    )
    successful_mean = _mean_label_proportions(
        successful_client_ids, client_label_proportions
    )
    if attempted_mean is None:
        return {}
    result = {
        f"attempted_client_mean_label_proportion_{label}": float(
            attempted_mean[label].item()
        )
        for label in range(10)
    }
    if successful_mean is None:
        result.update(
            {
                "label_representation_error_sq": float("nan"),
                "successful_label_entropy": float("nan"),
                "successful_classes_represented": 0,
            }
        )
        result.update(
            {
                f"successful_client_mean_label_proportion_{label}": float("nan")
                for label in range(10)
            }
        )
        return result
    result["label_representation_error_sq"] = float(
        torch.sum((successful_mean - attempted_mean) ** 2).item()
    )
    positive = successful_mean[successful_mean > 0]
    result["successful_label_entropy"] = float(
        -torch.sum(positive * torch.log(positive)).item()
    )
    result["successful_classes_represented"] = int(
        torch.sum(successful_mean > 0).item()
    )
    result.update(
        {
            f"successful_client_mean_label_proportion_{label}": float(
                successful_mean[label].item()
            )
            for label in range(10)
        }
    )
    return result

def _dummy_update_payloads(client_ids: list[int], payload_bytes: int) -> dict[int, bytes]:
    """Create opaque same-size payloads for communication-only pre-passes."""
    payload = bytes(payload_bytes)
    return {cid: payload for cid in client_ids}


def _delivered_fragment_indices(collection) -> dict[int, list[int]]:
    """Return physically delivered fragment indices for attempted clients."""
    delivered = {cid: [] for cid in collection.attempted_clients}
    for record in getattr(collection, "_txs_log", []):
        if record.get("status") == "success":
            delivered[int(record["client_id"])].append(int(record["fragment_idx"]))
    return {cid: sorted(indices) for cid, indices in delivered.items()}


def _fragment_receptions(collection):
    return receptions_from_transmission_log(getattr(collection, "_txs_log", []))


def _reception_metrics(collection):
    receptions = _fragment_receptions(collection)
    diagnostics, client_rows = fragment_reception_diagnostics(
        collection.selected_clients,
        receptions,
    )
    total_fragment_jitter_s = sum(
        float(row.get("fragment_jitter_s", 0.0))
        for row in getattr(collection, "_txs_log", [])
    )
    diagnostics["total_fragment_jitter_s"] = total_fragment_jitter_s
    diagnostics["mean_fragment_jitter_s"] = (
        total_fragment_jitter_s
        / max(1, int(diagnostics["num_fragments_attempted"]))
    )
    return receptions, diagnostics, client_rows


def _partial_reception_counts(collection) -> tuple[int, int]:
    delivered = _delivered_fragment_indices(collection)
    return (
        len(collection.attempted_clients),
        sum(bool(indices) for indices in delivered.values()),
    )

def run_simulation_time_based(
    config_dict:     dict,
    policy:          RoundPolicyConfig,
    seed:            int,
    result_dir:      str,
    trainloaders,
    testloader,
    initial_state_dict=None,
    client_label_proportions=None,
) -> dict:
    set_all_seeds(seed)

    T_end_s          = config_dict["T_end_s"]
    num_clients      = config_dict["num_clients"]
    num_interferers = int(config_dict.get("num_interferers", 0))
    if num_interferers < 0:
        raise ValueError("num_interferers must be nonnegative")
    if num_interferers and len(trainloaders) != num_clients:
        raise ValueError("Only num_clients focal data partitions may be supplied")
    num_uplink_channels = config_dict.get("num_uplink_channels", 3)
    if num_uplink_channels <= 0:
        raise ValueError("num_uplink_channels must be positive")
    spreading_factor = config_dict["spreading_factor"]
    # ``fragmentation_level`` remains the legacy UL-facing label.  The two
    # directional levels are independent compressed-payload budgets in MTUs.
    uplink_fragmentation_level = config_dict.get(
        "uplink_fragmentation_level", config_dict["fragmentation_level"])
    downlink_fragmentation_level = config_dict.get(
        "downlink_fragmentation_level", config_dict["fragmentation_level"])
    fragmentation_level = uplink_fragmentation_level
    ideal_downlink = config_dict.get("ideal_downlink", False)
    collision_free_ablation = bool(config_dict.get("collision_free_ablation", False))
    aggregation_policy = policy_from_config(config_dict)
    partial_fragment_aggregation = (
        aggregation_policy != AggregationPolicy.DISCARD_ALL
    )
    coordinate_wise_fragment_normalization = aggregation_policy in {
        AggregationPolicy.MASKED_PARTIAL,
        AggregationPolicy.MASKED_FIXED_DENOMINATOR,
    }
    if collision_free_ablation and partial_fragment_aggregation:
        raise ValueError(
            "collision_free_ablation and partial_fragment_aggregation are mutually exclusive"
        )
    if partial_fragment_aggregation and policy.policy_type != RoundPolicyType.ADAPTIVE_COMPLETE:
        raise ValueError(
            "partial_fragment_aggregation currently requires adaptive_complete"
        )
    FEC_rate         = config_dict["FEC_rate"]
    participation_probability = config_dict["participation_probability"]
    duty_cycle = float(config_dict.get("duty_cycle", 0.01))
    if not 0.0 < duty_cycle <= 1.0:
        raise ValueError("duty_cycle must be a fraction in (0, 1]")
    duty_cycle_percent = duty_cycle * 100.0
    contention_jitter_s = float(config_dict.get(
        "initial_jitter_window_s", config_dict.get("contention_jitter_s", 34.0)
    ))
    fragment_jitter_mode = config_dict.get(
        "fragment_jitter_mode", "initial_only"
    )
    fragment_jitter_window_s = float(
        config_dict.get("fragment_jitter_window_s", 0.0)
    )
    if (
        "enable_fragment_jitter" in config_dict
        or "initial_jitter_window_s" in config_dict
    ):
        enable_fragment_jitter = bool(
            config_dict.get("enable_fragment_jitter", False)
        )
        timing_config = UplinkTimingConfig(
            initial_jitter_window_s=contention_jitter_s,
            fragment_jitter_window_s=fragment_jitter_window_s,
            enable_fragment_jitter=enable_fragment_jitter,
            mode=(
                fragment_jitter_mode
                if enable_fragment_jitter
                else INITIAL_ONLY
            ),
        )
    else:
        timing_config = timing_config_from_legacy(
            contention_jitter_s,
            fragment_jitter_mode,
            fragment_jitter_window_s,
        )
        enable_fragment_jitter = timing_config.enable_fragment_jitter
    fragment_jitter_mode = timing_config.mode
    rng_streams = UplinkRandomStreams.from_seed(seed)
    aggregation_time_s = config_dict["aggregation_time_s"]
    ping_periodicity = config_dict["ping_periodicity"]
    dataset_name     = config_dict.get("dataset_name", "mnist")
    model_name       = config_dict.get("model_name", "tinycnn")
    num_classes      = int(config_dict.get("num_classes", 10))
    local_epochs     = config_dict.get("local_epochs", 1)
    local_lr         = config_dict.get("local_lr", 0.01)
    server_lr        = config_dict.get("server_lr", 1.0)
    target_accuracies = config_dict.get("target_accuracies", [])
    stop_at_target_accuracy = config_dict.get("stop_at_target_accuracy")
    log_update_information = bool(
        config_dict.get("log_update_information", False)
    )
    deterministic_client_training = bool(
        config_dict.get("deterministic_client_training", False)
    )
    enable_oracle_diagnostics = bool(
        config_dict.get("enable_oracle_diagnostics", False)
    )
    if enable_oracle_diagnostics and not deterministic_client_training:
        raise ValueError(
            "Oracle diagnostics require deterministic client training"
        )
    if log_update_information and not deterministic_client_training:
        raise ValueError(
            "Instrumented update logging requires deterministic client training"
        )
    if partial_fragment_aggregation and log_update_information:
        raise ValueError(
            "partial_fragment_aggregation does not yet support instrumented update logging"
        )
    mtu_bytes = LoRa_MTU(spreading_factor)

    uplink_compression_cfg = CompressionConfig(
        use_sparsity      = config_dict["use_sparsity"],
        use_quantization  = config_dict["use_quantization"],
        quantization_bits = config_dict["quantization_bits"],
        sparsity_threshold= config_dict["sparsity_threshold"],
        apply_zlib        = config_dict["apply_zlib"],
        max_payload_bytes = mtu_bytes * uplink_fragmentation_level,
    )
    downlink_compression_cfg = CompressionConfig(
        use_sparsity      = config_dict["use_sparsity"],
        use_quantization  = config_dict["use_quantization"],
        quantization_bits = config_dict["quantization_bits"],
        sparsity_threshold= config_dict["sparsity_threshold"],
        apply_zlib        = config_dict["apply_zlib"],
        max_payload_bytes = mtu_bytes * downlink_fragmentation_level,
    )
    energy_cfg = EnergyConfig(
        tx_power_w=config_dict.get("tx_power_w", 0.1),
        max_client_packet_attempts=config_dict.get("max_client_packet_attempts"),
        max_client_ul_airtime_s=config_dict.get("max_client_ul_airtime_s"),
        max_client_tx_energy_j=config_dict.get("max_client_tx_energy_j"),
        max_round_packet_attempts=config_dict.get("max_round_packet_attempts"),
        max_round_ul_airtime_s=config_dict.get("max_round_ul_airtime_s"),
    )

    # --- State initialisation ---
    dev_addrs    = [0x26010000 + cid for cid in range(num_clients)]
    dev_dc       = [DutyCycleTracker(duty_cycle_percent) for _ in range(num_clients)]
    gw_dc        = DutyCycleTracker(duty_cycle_percent)
    client_mac   = [ClientMacState() for _ in range(num_clients)]
    aloha        = AlohaChannel(num_uplink_channels)
    model_kwargs = {}
    if config_dict.get("base_filters") is not None:
        model_kwargs["base_filters"] = int(config_dict["base_filters"])
    if config_dict.get("model_norm") is not None:
        model_kwargs["norm"] = str(config_dict["model_norm"])
    client_model = create_model(model_name, dataset_name, num_classes, **model_kwargs).to(DEVICE)
    global_model = create_model(model_name, dataset_name, num_classes, **model_kwargs).to(DEVICE)
    if initial_state_dict is not None:
        global_model.load_state_dict(initial_state_dict)
    sim_time_s   = 0.0
    round_ready  = 0.0
    episode_idx  = 0

    # --- CSV writers ---
    ep_path   = os.path.join(result_dir, "episode_metrics.csv")
    tx_path   = os.path.join(result_dir, "transmission_log.csv")
    client_rx_path = os.path.join(result_dir, "client_fragment_reception.csv")
    layer_quality_path = os.path.join(result_dir, "aggregate_quality_layers.csv")
    ep_fields = [
        "seed","policy","num_clients","num_uplink_channels","dirichlet_alpha",
        "partition_seed","participation_probability",
        "fragmentation_level","fragment_count","uplink_fragmentation_level",
        "downlink_fragmentation_level","ideal_downlink","collision_free_ablation",
        "aggregation_policy","initial_jitter_window_s","enable_fragment_jitter",
        "fragment_jitter_mode","fragment_jitter_window_s",
        "duty_cycle","seed_learning","seed_participation",
        "seed_initial_jitter","seed_fragment_jitter","seed_channel",
        "partial_fragment_aggregation",
        "coordinate_wise_fragment_normalization",
        "episode_idx","sim_time_s",
        "collection_start_s","collection_end_s","collection_duration_s",
        "aggregation_time_s","round_duration_s","communication_time_s",
        "window_s","target_k","trigger_reason",
        "num_dl_recipients","num_dl_missed",
        "downlink_start_s","downlink_end_s","downlink_airtime_s",
        "downlink_duration_s","downlink_num_fragments","downlink_payload_bytes",
        "num_prepared_updates","num_selected","num_attempts","num_successes",
        "num_collisions","num_blocked","num_packet_attempts",
        "num_packet_successes","num_packet_collisions",
        "num_energy_blocked","num_aggregation_clients",
        "num_fragment_contributing_clients","fragment_delivery_rate",
        "selected_clients","num_fragments_attempted",
        "num_fragments_successful","num_fragments_collided",
        "fragment_success_rate","successful_fragments_per_hour",
        "num_clients_full_success",
        "num_clients_partial_success","num_clients_zero_success",
        "total_fragment_jitter_s","mean_fragment_jitter_s",
        "mean_received_fragment_fraction_per_client",
        "mean_received_fraction_among_partial_clients",
        "mean_valid_coordinate_fraction",
        "mean_contributors_per_coordinate","median_contributors_per_coordinate",
        "min_contributors_per_coordinate",
        "max_contributors_per_coordinate",
        "fraction_coordinates_with_zero_contributors",
        "fraction_coordinates_with_one_contributor",
        "fraction_coordinates_with_two_contributors",
        "aggregate_cosine_to_oracle","aggregate_norm_ratio_to_oracle",
        "aggregate_relative_l2_error","aggregate_sign_disagreement_fraction",
        "aggregate_norm","oracle_norm",
        "exact_full_zero_equivalence_checked",
        "exact_full_zero_equivalence_passed","oracle_rng_state_unchanged",
        "local_update_set_sha256","local_training_seed_fingerprint",
        "complete_update_failures","successful_update_fraction_conditioned_on_attempt",
        "empty_round",
        "total_ul_airtime_s","total_airtime_s",
        "round_tx_energy_j",
        "cumulative_packet_attempts","cumulative_packet_successes",
        "cumulative_packet_collisions","cumulative_ul_airtime_s",
        "cumulative_downlink_airtime_s","cumulative_total_airtime_s",
        "cumulative_tx_energy_j",
        "aggregated","test_loss","test_accuracy",
        "precision_macro","recall_macro","f1_macro",
        *UPDATE_INFORMATION_FIELDS,
        *LABEL_REPRESENTATION_FIELDS,
    ]
    tx_fields = [
        "seed","policy","num_clients","num_uplink_channels","dirichlet_alpha",
        "partition_seed","participation_probability",
        "fragmentation_level","fragment_count","uplink_fragmentation_level",
        "downlink_fragmentation_level","ideal_downlink","collision_free_ablation",
        "aggregation_policy","initial_jitter_window_s","enable_fragment_jitter",
        "fragment_jitter_mode","fragment_jitter_window_s",
        "duty_cycle","seed_learning","seed_participation",
        "seed_initial_jitter","seed_fragment_jitter","seed_channel",
        "partial_fragment_aggregation",
        "coordinate_wise_fragment_normalization",
        "episode_idx","client_id","round_index",
        "packet_id","fragment_idx","num_fragments","coordinate_mapping",
        "start_s","end_s","earliest_allowed_s","airtime_s","channel_idx",
        "payload_bytes","initial_jitter_s","fragment_jitter_s",
        "preceding_fragment_jitter_s","status",
    ]
    ep_fh, ep_w = _open_csv(ep_path, ep_fields)
    tx_fh, tx_w = _open_csv(tx_path, tx_fields)
    client_rx_fields = [
        "seed", "aggregation_policy", "episode_idx", "client_id",
        "num_fragments", "successful_fragments", "collided_fragments",
        "fraction_received", "fraction_fragments_received",
        "fraction_retained_nonzero_entries_received",
        "fraction_update_squared_norm_received",
    ]
    client_rx_fh, client_rx_w = _open_csv(client_rx_path, client_rx_fields)
    layer_quality_fields = [
        "seed", "aggregation_policy", "episode_idx", "layer",
        "aggregate_cosine_to_oracle", "aggregate_norm_ratio_to_oracle",
        "aggregate_relative_l2_error", "aggregate_sign_disagreement_fraction",
        "aggregate_norm", "oracle_norm",
    ]
    layer_quality_fh, layer_quality_w = _open_csv(
        layer_quality_path, layer_quality_fields
    )

    print(f"\n{'='*70}")
    print(
        f"  Policy: {policy.name}  |  seed={seed}  |  "
        f"UL budget={uplink_fragmentation_level} MTU | "
        f"DL budget={downlink_fragmentation_level} MTU | "
        f"ideal DL={ideal_downlink} | T_end={T_end_s}s")
    print(f"{'='*70}")

    uplink_payload_bytes = encoded_update_payload_bytes(global_model, uplink_compression_cfg)
    population = (PopulationCollector(num_clients, num_interferers, duty_cycle_percent,
                                      result_dir, uplink_payload_bytes)
                  if num_interferers else None)
    uplink_collector = population if population is not None else collect_uplinks

    run_metadata = {
        "seed": seed,
        "policy": policy.name,
        "num_clients": num_clients,
        "num_uplink_channels": num_uplink_channels,
        "dirichlet_alpha": config_dict.get("dirichlet_alpha"),
        "partition_seed": config_dict.get("partition_seed"),
        "participation_probability": participation_probability,
        "fragmentation_level": fragmentation_level,
        "fragment_count": uplink_fragmentation_level,
        "uplink_fragmentation_level": uplink_fragmentation_level,
        "downlink_fragmentation_level": downlink_fragmentation_level,
        "ideal_downlink": int(ideal_downlink),
        "collision_free_ablation": int(collision_free_ablation),
        "aggregation_policy": aggregation_policy.value,
        "initial_jitter_window_s": contention_jitter_s,
        "enable_fragment_jitter": int(enable_fragment_jitter),
        "fragment_jitter_mode": fragment_jitter_mode,
        "fragment_jitter_window_s": fragment_jitter_window_s,
        "duty_cycle": duty_cycle,
        "seed_learning": seed,
        "seed_participation": rng_streams.seeds["participation"],
        "seed_initial_jitter": rng_streams.seeds["initial_jitter"],
        "seed_fragment_jitter": rng_streams.seeds["fragment_jitter"],
        "seed_channel": rng_streams.seeds["channel"],
        "partial_fragment_aggregation": int(partial_fragment_aggregation),
        "coordinate_wise_fragment_normalization": int(
            coordinate_wise_fragment_normalization
        ),
    }

    # --- Episode 0: bootstrap (no real downlink, everyone starts from global) ---
    # Communication is resolved before local training to avoid training updates
    # that the channel would later discard.
    eligible0 = list(range(num_clients))
    prepared0 = _dummy_update_payloads(eligible0, uplink_payload_bytes)
    collection0 = uplink_collector(
        0.0, T_end_s, policy, prepared0, dev_dc, client_mac,
        spreading_factor, participation_probability, aloha, contention_jitter_s,
        energy_cfg, collision_free_ablation=collision_free_ablation,
        fragment_jitter_mode=fragment_jitter_mode,
        fragment_jitter_window_s=fragment_jitter_window_s,
        timing_config=timing_config,
        rng_streams=rng_streams,
        round_index=0)
    aggregation_client_ids0 = (
        collection0.attempted_clients
        if partial_fragment_aggregation
        else collection0.successful_clients
    )
    fragment_receptions0, reception_metrics0, client_reception_rows0 = (
        _reception_metrics(collection0)
    )
    num_aggregation_clients0, num_fragment_contributing_clients0 = (
        _partial_reception_counts(collection0)
        if partial_fragment_aggregation
        else (collection0.num_successes, collection0.num_successes)
    )
    label_information_rows = []
    label_information0 = _label_representation_metrics(
        collection0.attempted_clients,
        aggregation_client_ids0,
        client_label_proportions,
    )
    if label_information0:
        label_information_rows.append(label_information0)
    information_rows = []
    layer_information0 = []
    client_information0 = []
    if enable_oracle_diagnostics and collection0.attempted_clients:
        successful0, information0, client_information0, layer_information0 = (
            run_oracle_diagnostic_local_training_phase(
                collection0.attempted_clients,
                client_model,
                global_model,
                trainloaders,
                uplink_compression_cfg,
                spreading_factor,
                local_epochs,
                local_lr,
                fragment_receptions0,
                aggregation_policy,
                training_seed_base=seed * 10_000_000,
                episode_idx=0,
            )
        )
        information_rows.append(information0)
    elif log_update_information:
        successful0, information0 = run_instrumented_local_training_phase(
            collection0.attempted_clients,
            collection0.successful_clients,
            client_model,
            global_model,
            trainloaders,
            uplink_compression_cfg,
            spreading_factor,
            local_epochs,
            local_lr,
            training_seed_base=seed * 10_000_000,
            episode_idx=0,
        )
        information_rows.append(information0)
    else:
        successful0 = run_local_training_phase(
            aggregation_client_ids0, client_model, global_model,
            trainloaders, uplink_compression_cfg, spreading_factor, local_epochs, local_lr,
            delivered_fragment_indices=None,
            num_fragments=None,
            coordinate_wise_fragment_normalization=(
                coordinate_wise_fragment_normalization
            ),
            fragment_receptions_by_client=(
                fragment_receptions0 if partial_fragment_aggregation else None
            ),
            aggregation_policy=aggregation_policy,
            training_seed_base=(seed * 10_000_000 if deterministic_client_training else None),
            episode_idx=0)
        information0 = {}
    collection0.successful_updates = successful0
    sim_time_s = collection0.collection_end_s
    if collection0.num_attempts > 0:
        sim_time_s += aggregation_time_s
    round_ready = sim_time_s
    coverage0 = {}
    aggregated0 = aggregate_successful_updates(
        global_model, collection0.successful_updates, uplink_compression_cfg.apply_zlib,
        server_lr,
        coordinate_wise_fragment_normalization=coordinate_wise_fragment_normalization,
        aggregation_policy=aggregation_policy,
        coverage_diagnostics=coverage0)
    test_loss0, test_acc0, yt0, yp0 = test(global_model, testloader)
    prec0 = precision_score(yt0, yp0, average="macro", zero_division=0)
    rec0  = recall_score (yt0, yp0, average="macro", zero_division=0)
    f10   = f1_score     (yt0, yp0, average="macro", zero_division=0)
    latest_test_loss = test_loss0
    latest_test_acc = test_acc0
    target_stop_reached = (
        stop_at_target_accuracy is not None
        and latest_test_acc >= stop_at_target_accuracy
    )

    total_prepared_updates = len(prepared0)
    total_selected_updates = collection0.num_selected
    bootstrap_round_duration_s = collection0.collection_end_s + (
        aggregation_time_s if collection0.num_attempts > 0 else 0.0)
    total_round_duration_s = bootstrap_round_duration_s
    total_communication_time_s = collection0.collection_end_s
    total_downlink_airtime_s = 0.0
    total_downlink_duration_s = 0.0
    total_packet_attempts = collection0.num_packet_attempts
    total_packet_successes = collection0.num_packet_successes
    total_packet_collisions = collection0.num_packet_collisions
    total_successful_updates = collection0.num_successes
    total_aggregation_clients = num_aggregation_clients0
    total_fragment_contributing_clients = num_fragment_contributing_clients0
    total_clients_full_success = int(reception_metrics0["num_clients_full_success"])
    total_clients_partial_success = int(reception_metrics0["num_clients_partial_success"])
    total_clients_zero_success = int(reception_metrics0["num_clients_zero_success"])
    total_fragment_jitter_s = float(reception_metrics0["total_fragment_jitter_s"])
    total_tx_energy_j = collection0.total_ul_airtime_s * energy_cfg.tx_power_w

    ep_w.writerow({
        **run_metadata,
        "episode_idx": 0,
        "sim_time_s": sim_time_s,
        "collection_start_s": 0.0,
        "collection_end_s": collection0.collection_end_s,
        "collection_duration_s": collection0.collection_end_s,
        "aggregation_time_s": aggregation_time_s if collection0.num_attempts > 0 else 0.0,
        "round_duration_s": bootstrap_round_duration_s,
        "communication_time_s": collection0.collection_end_s,
        "window_s": policy.window_s,
        "target_k": policy.target_successes_k,
        "trigger_reason": collection0.trigger_reason,
        "num_dl_recipients": num_clients, "num_dl_missed": 0,
        "downlink_start_s": 0.0, "downlink_end_s": 0.0,
        "downlink_airtime_s": 0.0,
        "downlink_duration_s": 0.0,
        "downlink_num_fragments": 0,
        "downlink_payload_bytes": 0,
        "num_prepared_updates": len(prepared0),
        "num_selected": collection0.num_selected,
        "num_attempts": collection0.num_attempts,
        "num_successes": collection0.num_successes,
        "num_collisions": collection0.num_collisions,
        "num_blocked": collection0.num_blocked,
        "num_packet_attempts": collection0.num_packet_attempts,
        "num_packet_successes": collection0.num_packet_successes,
        "num_packet_collisions": collection0.num_packet_collisions,
        "num_energy_blocked": collection0.num_energy_blocked,
        "num_aggregation_clients": num_aggregation_clients0,
        "num_fragment_contributing_clients": num_fragment_contributing_clients0,
        "fragment_delivery_rate": (
            collection0.num_packet_successes / max(1, collection0.num_packet_attempts)
        ),
        **reception_metrics0,
        "successful_fragments_per_hour": (
            collection0.num_packet_successes
            / max(bootstrap_round_duration_s / 3600.0, 1e-12)
        ),
        **coverage0,
        "complete_update_failures": (
            collection0.num_attempts - collection0.num_successes
        ),
        "successful_update_fraction_conditioned_on_attempt": (
            collection0.num_successes / max(1, collection0.num_attempts)
        ),
        "empty_round": int(num_fragment_contributing_clients0 == 0),
        "total_ul_airtime_s": collection0.total_ul_airtime_s,
        "total_airtime_s": collection0.total_ul_airtime_s,
        "round_tx_energy_j": collection0.total_ul_airtime_s * energy_cfg.tx_power_w,
        "cumulative_packet_attempts": total_packet_attempts,
        "cumulative_packet_successes": total_packet_successes,
        "cumulative_packet_collisions": total_packet_collisions,
        "cumulative_ul_airtime_s": sum(mac.total_ul_airtime_s for mac in client_mac),
        "cumulative_downlink_airtime_s": total_downlink_airtime_s,
        "cumulative_total_airtime_s": (
            sum(mac.total_ul_airtime_s for mac in client_mac)
            + total_downlink_airtime_s
        ),
        "cumulative_tx_energy_j": total_tx_energy_j,
        "aggregated": int(aggregated0),
        "test_loss": f"{test_loss0:.6f}", "test_accuracy": f"{test_acc0:.6f}",
        "precision_macro": f"{prec0:.6f}", "recall_macro": f"{rec0:.6f}",
        "f1_macro": f"{f10:.6f}",
        **information0,
        **label_information0,
    })
    ep_fh.flush()
    for tx in getattr(collection0, '_txs_log', []):
        tx_w.writerow({
            **run_metadata,
            "episode_idx": 0,
            **tx,
        })
    tx_fh.flush()
    for row in client_reception_rows0:
        client_info = next(
            (item for item in client_information0 if item["client_id"] == row["client_id"]),
            {},
        )
        client_rx_w.writerow({
            "seed": seed,
            "aggregation_policy": aggregation_policy.value,
            "episode_idx": 0,
            **row,
            **client_info,
        })
    client_rx_fh.flush()
    for row in layer_information0:
        layer_quality_w.writerow({
            "seed": seed,
            "aggregation_policy": aggregation_policy.value,
            "episode_idx": 0,
            **row,
        })
    layer_quality_fh.flush()
    episode_idx = 1

    print(f"  Ep 0 | t={sim_time_s:8.1f}s | "
          f"succ={collection0.num_successes} | "
          f"acc={test_acc0:.4f} | {collection0.trigger_reason}")

    # --- Main simulation loop ---
    while sim_time_s < T_end_s and not target_stop_reached:
        # -- Downlink --
        dl_result, sim_time_s, model_snapshot = run_downlink_phase(
            sim_time_s, global_model, dev_addrs, gw_dc, round_ready,
            spreading_factor, FEC_rate, ping_periodicity,
            downlink_compression_cfg, num_clients, client_mac, ideal_downlink)

        if sim_time_s >= T_end_s:
            total_round_duration_s += dl_result.duration_s
            total_communication_time_s += dl_result.duration_s
            total_downlink_airtime_s += dl_result.airtime_s
            total_downlink_duration_s += dl_result.duration_s
            ep_w.writerow({
                **run_metadata,
                "episode_idx": episode_idx,
                "sim_time_s": sim_time_s,
                "collection_start_s": sim_time_s,
                "collection_end_s": sim_time_s,
                "collection_duration_s": 0.0,
                "aggregation_time_s": 0.0,
                "round_duration_s": dl_result.duration_s,
                "communication_time_s": dl_result.duration_s,
                "window_s": policy.window_s,
                "target_k": policy.target_successes_k,
                "trigger_reason": "time_end_reached_after_downlink",
                "num_dl_recipients": len(dl_result.successful_clients),
                "num_dl_missed": len(dl_result.missed_clients),
                "downlink_start_s": dl_result.start_s,
                "downlink_end_s": dl_result.end_s,
                "downlink_airtime_s": dl_result.airtime_s,
                "downlink_duration_s": dl_result.duration_s,
                "downlink_num_fragments": dl_result.num_fragments,
                "downlink_payload_bytes": dl_result.payload_bytes,
                "num_prepared_updates": 0,
                "num_selected": 0,
                "num_attempts": 0,
                "num_successes": 0,
                "num_collisions": 0,
                "num_blocked": 0,
                "num_packet_attempts": 0,
                "num_packet_successes": 0,
                "num_packet_collisions": 0,
                "num_energy_blocked": 0,
                "total_ul_airtime_s": 0.0,
                "total_airtime_s": dl_result.airtime_s,
                "round_tx_energy_j": 0.0,
                "cumulative_packet_attempts": total_packet_attempts,
                "cumulative_packet_successes": total_packet_successes,
                "cumulative_packet_collisions": total_packet_collisions,
                "cumulative_ul_airtime_s": sum(mac.total_ul_airtime_s for mac in client_mac),
                "cumulative_downlink_airtime_s": total_downlink_airtime_s,
                "cumulative_total_airtime_s": (
                    sum(mac.total_ul_airtime_s for mac in client_mac)
                    + total_downlink_airtime_s
                ),
                "cumulative_tx_energy_j": total_tx_energy_j,
                "aggregated": 0,
                "test_loss": f"{latest_test_loss:.6f}",
                "test_accuracy": f"{latest_test_acc:.6f}",
                "precision_macro": "",
                "recall_macro": "",
                "f1_macro": "",
            })
            ep_fh.flush()
            print(f"  Ep {episode_idx:3d} | t={sim_time_s:8.1f}s | "
                  f"DL={len(dl_result.successful_clients):2d}/{num_clients} | "
                  "succ= 0 | no_uplink | time_end_reached_after_downlink")
            episode_idx += 1
            break

        # -- Communication pre-pass --
        communication_candidates = dl_result.successful_clients
        total_prepared_updates += len(communication_candidates)

        if not communication_candidates:
            total_round_duration_s += dl_result.duration_s
            total_communication_time_s += dl_result.duration_s
            total_downlink_airtime_s += dl_result.airtime_s
            total_downlink_duration_s += dl_result.duration_s
            trigger_reason = "no_downlink_recipients"
            ep_w.writerow({
                **run_metadata,
                "episode_idx": episode_idx,
                "sim_time_s": sim_time_s,
                "collection_start_s": sim_time_s,
                "collection_end_s": sim_time_s,
                "collection_duration_s": 0.0,
                "aggregation_time_s": 0.0,
                "round_duration_s": dl_result.duration_s,
                "communication_time_s": dl_result.duration_s,
                "window_s": policy.window_s,
                "target_k": policy.target_successes_k,
                "trigger_reason": trigger_reason,
                "num_dl_recipients": len(dl_result.successful_clients),
                "num_dl_missed": len(dl_result.missed_clients),
                "downlink_start_s": dl_result.start_s,
                "downlink_end_s": dl_result.end_s,
                "downlink_airtime_s": dl_result.airtime_s,
                "downlink_duration_s": dl_result.duration_s,
                "downlink_num_fragments": dl_result.num_fragments,
                "downlink_payload_bytes": dl_result.payload_bytes,
                "num_prepared_updates": len(communication_candidates),
                "num_selected": 0,
                "num_attempts": 0,
                "num_successes": 0,
                "num_collisions": 0,
                "num_blocked": 0,
                "num_packet_attempts": 0,
                "num_packet_successes": 0,
                "num_packet_collisions": 0,
                "num_energy_blocked": 0,
                "total_ul_airtime_s": 0.0,
                "total_airtime_s": dl_result.airtime_s,
                "round_tx_energy_j": 0.0,
                "cumulative_packet_attempts": total_packet_attempts,
                "cumulative_packet_successes": total_packet_successes,
                "cumulative_packet_collisions": total_packet_collisions,
                "cumulative_ul_airtime_s": sum(mac.total_ul_airtime_s for mac in client_mac),
                "cumulative_downlink_airtime_s": total_downlink_airtime_s,
                "cumulative_total_airtime_s": (
                    sum(mac.total_ul_airtime_s for mac in client_mac)
                    + total_downlink_airtime_s
                ),
                "cumulative_tx_energy_j": total_tx_energy_j,
                "aggregated": 0,
                "test_loss": f"{latest_test_loss:.6f}",
                "test_accuracy": f"{latest_test_acc:.6f}",
                "precision_macro": "",
                "recall_macro": "",
                "f1_macro": "",
            })
            ep_fh.flush()
            print(f"  Ep {episode_idx:3d} | t={sim_time_s:8.1f}s | "
                  f"DL={len(dl_result.successful_clients):2d}/{num_clients} | "
                  f"succ= 0 | no_uplink | {trigger_reason}")
            episode_idx += 1
            round_ready = sim_time_s
            continue

        # -- Uplink collection --
        collection_start = sim_time_s
        prepared = _dummy_update_payloads(communication_candidates, uplink_payload_bytes)
        collection = uplink_collector(
            collection_start, T_end_s, policy, prepared,
            dev_dc, client_mac, spreading_factor, participation_probability,
            aloha, contention_jitter_s, energy_cfg,
            collision_free_ablation=collision_free_ablation,
            fragment_jitter_mode=fragment_jitter_mode,
            fragment_jitter_window_s=fragment_jitter_window_s,
            timing_config=timing_config,
            rng_streams=rng_streams,
            round_index=episode_idx)
        aggregation_client_ids = (
            collection.attempted_clients
            if partial_fragment_aggregation
            else collection.successful_clients
        )
        fragment_receptions, reception_metrics, client_reception_rows = (
            _reception_metrics(collection)
        )
        num_aggregation_clients, num_fragment_contributing_clients = (
            _partial_reception_counts(collection)
            if partial_fragment_aggregation
            else (collection.num_successes, collection.num_successes)
        )
        label_information = _label_representation_metrics(
            collection.attempted_clients,
            aggregation_client_ids,
            client_label_proportions,
        )
        if label_information:
            label_information_rows.append(label_information)
        sim_time_s  = collection.collection_end_s
        if collection.num_attempts > 0:
            sim_time_s += aggregation_time_s
        round_ready = sim_time_s

        # -- Local training for complete successes or all attempted partial updates --
        layer_information = []
        client_information = []
        if enable_oracle_diagnostics and collection.attempted_clients:
            successful_updates, information, client_information, layer_information = (
                run_oracle_diagnostic_local_training_phase(
                    collection.attempted_clients,
                    client_model,
                    model_snapshot,
                    trainloaders,
                    uplink_compression_cfg,
                    spreading_factor,
                    local_epochs,
                    local_lr,
                    fragment_receptions,
                    aggregation_policy,
                    training_seed_base=seed * 10_000_000,
                    episode_idx=episode_idx,
                )
            )
            information_rows.append(information)
        elif log_update_information:
            successful_updates, information = run_instrumented_local_training_phase(
                collection.attempted_clients,
                collection.successful_clients,
                client_model,
                model_snapshot,
                trainloaders,
                uplink_compression_cfg,
                spreading_factor,
                local_epochs,
                local_lr,
                training_seed_base=seed * 10_000_000,
                episode_idx=episode_idx,
            )
            information_rows.append(information)
        else:
            successful_updates = run_local_training_phase(
                aggregation_client_ids, client_model, model_snapshot,
                trainloaders, uplink_compression_cfg, spreading_factor, local_epochs, local_lr,
                delivered_fragment_indices=None,
                num_fragments=None,
                coordinate_wise_fragment_normalization=(
                    coordinate_wise_fragment_normalization
                ),
                fragment_receptions_by_client=(
                    fragment_receptions if partial_fragment_aggregation else None
                ),
                aggregation_policy=aggregation_policy,
                training_seed_base=(seed * 10_000_000 if deterministic_client_training else None),
                episode_idx=episode_idx)
            information = {}
        collection.successful_updates = successful_updates

        # -- Aggregation --
        coverage = {}
        aggregated = aggregate_successful_updates(
            global_model, collection.successful_updates, uplink_compression_cfg.apply_zlib,
            server_lr,
            coordinate_wise_fragment_normalization=coordinate_wise_fragment_normalization,
            aggregation_policy=aggregation_policy,
            coverage_diagnostics=coverage)

        # -- Evaluation --
        test_loss, test_acc, yt, yp = test(global_model, testloader)
        latest_test_loss = test_loss
        latest_test_acc = test_acc
        prec = precision_score(yt, yp, average="macro", zero_division=0)
        rec  = recall_score  (yt, yp, average="macro", zero_division=0)
        f1v  = f1_score      (yt, yp, average="macro", zero_division=0)

        # -- Log episode --
        collection_duration_s = collection.collection_end_s - collection.collection_start_s
        round_duration_s = dl_result.duration_s + collection_duration_s + (
            aggregation_time_s if collection.num_attempts > 0 else 0.0)
        total_airtime_s = dl_result.airtime_s + collection.total_ul_airtime_s
        total_round_duration_s += round_duration_s
        total_communication_time_s += dl_result.duration_s + collection_duration_s
        total_downlink_airtime_s += dl_result.airtime_s
        total_downlink_duration_s += dl_result.duration_s
        total_packet_attempts += collection.num_packet_attempts
        total_packet_successes += collection.num_packet_successes
        total_packet_collisions += collection.num_packet_collisions
        total_successful_updates += collection.num_successes
        total_aggregation_clients += num_aggregation_clients
        total_fragment_contributing_clients += num_fragment_contributing_clients
        total_clients_full_success += int(reception_metrics["num_clients_full_success"])
        total_clients_partial_success += int(reception_metrics["num_clients_partial_success"])
        total_clients_zero_success += int(reception_metrics["num_clients_zero_success"])
        total_fragment_jitter_s += float(reception_metrics["total_fragment_jitter_s"])
        total_selected_updates += collection.num_selected
        round_tx_energy_j = collection.total_ul_airtime_s * energy_cfg.tx_power_w
        total_tx_energy_j += round_tx_energy_j
        cumulative_ul_airtime_s = sum(mac.total_ul_airtime_s for mac in client_mac)
        ep_w.writerow({
            **run_metadata,
            "episode_idx": episode_idx,
            "sim_time_s": sim_time_s,
            "collection_start_s": collection.collection_start_s,
            "collection_end_s": collection.collection_end_s,
            "collection_duration_s": collection_duration_s,
            "aggregation_time_s": aggregation_time_s if collection.num_attempts > 0 else 0.0,
            "round_duration_s": round_duration_s,
            "communication_time_s": dl_result.duration_s + collection_duration_s,
            "window_s": policy.window_s,
            "target_k": policy.target_successes_k,
            "trigger_reason": collection.trigger_reason,
            "num_dl_recipients": len(dl_result.successful_clients),
            "num_dl_missed": len(dl_result.missed_clients),
            "downlink_start_s": dl_result.start_s,
            "downlink_end_s": dl_result.end_s,
            "downlink_airtime_s": dl_result.airtime_s,
            "downlink_duration_s": dl_result.duration_s,
            "downlink_num_fragments": dl_result.num_fragments,
            "downlink_payload_bytes": dl_result.payload_bytes,
            "num_prepared_updates": len(communication_candidates),
            "num_selected": collection.num_selected,
            "num_attempts": collection.num_attempts,
            "num_successes": collection.num_successes,
            "num_collisions": collection.num_collisions,
            "num_blocked": collection.num_blocked,
            "num_packet_attempts": collection.num_packet_attempts,
            "num_packet_successes": collection.num_packet_successes,
            "num_packet_collisions": collection.num_packet_collisions,
            "num_energy_blocked": collection.num_energy_blocked,
            "num_aggregation_clients": num_aggregation_clients,
            "num_fragment_contributing_clients": num_fragment_contributing_clients,
            "fragment_delivery_rate": (
                collection.num_packet_successes / max(1, collection.num_packet_attempts)
            ),
            **reception_metrics,
            "successful_fragments_per_hour": (
                collection.num_packet_successes
                / max(round_duration_s / 3600.0, 1e-12)
            ),
            **coverage,
            "complete_update_failures": (
                collection.num_attempts - collection.num_successes
            ),
            "successful_update_fraction_conditioned_on_attempt": (
                collection.num_successes / max(1, collection.num_attempts)
            ),
            "empty_round": int(num_fragment_contributing_clients == 0),
            "total_ul_airtime_s": collection.total_ul_airtime_s,
            "total_airtime_s": total_airtime_s,
            "round_tx_energy_j": round_tx_energy_j,
            "cumulative_packet_attempts": total_packet_attempts,
            "cumulative_packet_successes": total_packet_successes,
            "cumulative_packet_collisions": total_packet_collisions,
            "cumulative_ul_airtime_s": cumulative_ul_airtime_s,
            "cumulative_downlink_airtime_s": total_downlink_airtime_s,
            "cumulative_total_airtime_s": (
                cumulative_ul_airtime_s + total_downlink_airtime_s
            ),
            "cumulative_tx_energy_j": total_tx_energy_j,
            "aggregated": int(aggregated),
            "test_loss": f"{test_loss:.6f}",
            "test_accuracy": f"{test_acc:.6f}",
            "precision_macro": f"{prec:.6f}",
            "recall_macro": f"{rec:.6f}",
            "f1_macro": f"{f1v:.6f}",
            **information,
            **label_information,
        })
        ep_fh.flush()

        # -- Log transmissions --
        for tx in getattr(collection, '_txs_log', []):
            tx_w.writerow({
                **run_metadata,
                "episode_idx": episode_idx,
                **tx,
            })
            tx_fh.flush()
        for row in client_reception_rows:
            client_info = next(
                (item for item in client_information if item["client_id"] == row["client_id"]),
                {},
            )
            client_rx_w.writerow({
                "seed": seed,
                "aggregation_policy": aggregation_policy.value,
                "episode_idx": episode_idx,
                **row,
                **client_info,
            })
        client_rx_fh.flush()
        for row in layer_information:
            layer_quality_w.writerow({
                "seed": seed,
                "aggregation_policy": aggregation_policy.value,
                "episode_idx": episode_idx,
                **row,
            })
        layer_quality_fh.flush()

        print(f"  Ep {episode_idx:3d} | t={sim_time_s:8.1f}s | "
              f"DL={len(dl_result.successful_clients):2d}/{num_clients} | "
              f"succ={collection.num_successes:2d} | "
              f"acc={test_acc:.4f} | {collection.trigger_reason}")

        target_stop_reached = (
            stop_at_target_accuracy is not None
            and latest_test_acc >= stop_at_target_accuracy
        )
        episode_idx += 1

    ep_fh.close(); tx_fh.close(); client_rx_fh.close(); layer_quality_fh.close()

    # --- Client summary CSV ---
    cli_path   = os.path.join(result_dir, "client_metrics.csv")
    cli_fields = [
        "seed","policy","num_clients","num_uplink_channels","dirichlet_alpha",
        "partition_seed","participation_probability",
        "fragmentation_level","fragment_count","uplink_fragmentation_level","downlink_fragmentation_level",
        "ideal_downlink","collision_free_ablation","aggregation_policy",
        "initial_jitter_window_s","enable_fragment_jitter",
        "fragment_jitter_mode","fragment_jitter_window_s","duty_cycle",
        "seed_learning","seed_participation","seed_initial_jitter",
        "seed_fragment_jitter","seed_channel","partial_fragment_aggregation",
        "coordinate_wise_fragment_normalization",
        "client_id",
        "num_dl_received","num_dl_missed",
        "num_ul_attempts","num_ul_successes","num_ul_collisions","num_ul_blocked",
        "num_energy_blocked",
        "num_packet_attempts","num_packet_successes","num_packet_collisions",
        "total_ul_airtime_s","total_tx_energy_j",
        "participation_rate","success_rate","collision_rate",
    ]
    with open(cli_path, "w", newline="") as cf:
        cw = csv.DictWriter(cf, fieldnames=cli_fields)
        cw.writeheader()
        for cid, mac in enumerate(client_mac):
            total_ep = episode_idx
            cw.writerow({
                **run_metadata, "client_id": cid,
                "num_dl_received": mac.num_dl_received,
                "num_dl_missed": mac.num_dl_missed,
                "num_ul_attempts": mac.num_ul_attempts,
                "num_ul_successes": mac.num_ul_successes,
                "num_ul_collisions": mac.num_ul_collisions,
                "num_ul_blocked": mac.num_ul_blocked,
                "num_energy_blocked": mac.num_energy_blocked,
                "num_packet_attempts": mac.num_packet_attempts,
                "num_packet_successes": mac.num_packet_successes,
                "num_packet_collisions": mac.num_packet_collisions,
                "total_ul_airtime_s": f"{mac.total_ul_airtime_s:.4f}",
                "total_tx_energy_j": f"{mac.total_tx_energy_j:.6f}",
                "participation_rate": (f"{mac.num_ul_attempts / max(1, total_ep):.4f}"),
                "success_rate": (f"{mac.num_ul_successes / max(1, mac.num_ul_attempts):.4f}"
                                 if mac.num_ul_attempts > 0 else "0.0000"),
                "collision_rate": (f"{mac.num_ul_collisions / max(1, mac.num_ul_attempts):.4f}"
                                   if mac.num_ul_attempts > 0 else "0.0000"),
            })

    # --- Final summary JSON ---
    success_counts = [mac.num_ul_successes for mac in client_mac]
    summary = {
        "seed": seed,
        "policy": policy.name,
        "num_clients": num_clients,
        "num_uplink_channels": num_uplink_channels,
        "fragmentation_level": fragmentation_level,
        "fragment_count": uplink_fragmentation_level,
        "uplink_fragmentation_level": uplink_fragmentation_level,
        "downlink_fragmentation_level": downlink_fragmentation_level,
        "ideal_downlink": ideal_downlink,
        "collision_free_ablation": collision_free_ablation,
        "aggregation_policy": aggregation_policy.value,
        "initial_jitter_window_s": contention_jitter_s,
        "enable_fragment_jitter": enable_fragment_jitter,
        "fragment_jitter_mode": fragment_jitter_mode,
        "fragment_jitter_window_s": fragment_jitter_window_s,
        "duty_cycle": duty_cycle,
        "seed_learning": seed,
        "seed_participation": rng_streams.seeds["participation"],
        "seed_initial_jitter": rng_streams.seeds["initial_jitter"],
        "seed_fragment_jitter": rng_streams.seeds["fragment_jitter"],
        "seed_channel": rng_streams.seeds["channel"],
        "partial_fragment_aggregation": partial_fragment_aggregation,
        "coordinate_wise_fragment_normalization": (
            coordinate_wise_fragment_normalization
        ),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "partition": config_dict.get("partition"),
        "dirichlet_alpha": config_dict.get("dirichlet_alpha"),
        "partition_seed": config_dict.get("partition_seed"),
        "participation_probability": participation_probability,
        "log_update_information": log_update_information,
        "enable_oracle_diagnostics": enable_oracle_diagnostics,
        "deterministic_client_training": deterministic_client_training,
        "local_epochs": local_epochs,
        "local_lr": local_lr,
        "server_lr": server_lr,
        "warm_start_target_accuracy": config_dict.get("warm_start_target_accuracy", 0.0),
        "warm_start_achieved_accuracy": config_dict.get("warm_start_achieved_accuracy"),
        "warm_start_epochs_run": config_dict.get("warm_start_epochs_run"),
        "target_accuracies": target_accuracies,
        "stop_at_target_accuracy": stop_at_target_accuracy,
        "target_stop_reached": target_stop_reached,
        "T_end_s": T_end_s,
        "window_s": policy.window_s,
        "target_k": policy.target_successes_k,
        "num_episodes": episode_idx,
        "final_sim_time_s": sim_time_s,
        "final_test_loss": float(latest_test_loss) if episode_idx > 0 else None,
        "final_test_accuracy": float(latest_test_acc) if episode_idx > 0 else None,
        "total_ul_attempts": sum(mac.num_ul_attempts for mac in client_mac),
        "total_ul_successes": sum(mac.num_ul_successes for mac in client_mac),
        "total_ul_collisions": sum(mac.num_ul_collisions for mac in client_mac),
        "total_ul_blocked": sum(mac.num_ul_blocked for mac in client_mac),
        "total_energy_blocked": sum(mac.num_energy_blocked for mac in client_mac),
        "total_ul_airtime_s": sum(mac.total_ul_airtime_s for mac in client_mac),
        "total_tx_energy_j": sum(mac.total_tx_energy_j for mac in client_mac),
        "total_downlink_airtime_s": total_downlink_airtime_s,
        "total_downlink_duration_s": total_downlink_duration_s,
        "total_round_duration_s": total_round_duration_s,
        "total_communication_time_s": total_communication_time_s,
        "total_packet_attempts": total_packet_attempts,
        "total_packet_successes": total_packet_successes,
        "total_packet_collisions": total_packet_collisions,
        "total_successful_updates": total_successful_updates,
        "total_aggregation_clients": total_aggregation_clients,
        "total_fragment_contributing_clients": total_fragment_contributing_clients,
        "total_clients_full_success": total_clients_full_success,
        "total_clients_partial_success": total_clients_partial_success,
        "total_clients_zero_success": total_clients_zero_success,
        "total_fragment_jitter_s": total_fragment_jitter_s,
        "mean_fragment_jitter_s": (
            total_fragment_jitter_s / max(1, total_packet_attempts)
        ),
        "fragment_delivery_rate": (
            total_packet_successes / max(1, total_packet_attempts)
        ),
        "total_selected_updates": total_selected_updates,
        "total_prepared_updates": total_prepared_updates,
        "actual_participation_level": (
            total_selected_updates / max(1, total_prepared_updates)
        ),
        "actual_attempt_level": (
            sum(mac.num_ul_attempts for mac in client_mac) / max(1, total_prepared_updates)
        ),
        "participation_gini": gini(success_counts),
        "successful_updates_per_simulated_hour": (
            total_successful_updates / max(T_end_s / 3600.0, 1e-12)
        ),
        "information_episode_count": len(information_rows),
        "label_representation_episode_count": len(label_information_rows),
        "unique_successful_clients": sum(count > 0 for count in success_counts),
        "successful_client_coverage_fraction": (
            sum(count > 0 for count in success_counts) / max(1, num_clients)
        ),
    }
    for field in UPDATE_INFORMATION_FIELDS:
        values = [
            float(row[field])
            for row in information_rows
            if field in row
            and isinstance(row[field], (int, float))
            and math.isfinite(float(row[field]))
        ]
        summary[f"mean_{field}"] = (
            sum(values) / len(values) if values else None
        )
    label_errors = [
        float(row["label_representation_error_sq"])
        for row in label_information_rows
        if math.isfinite(float(row["label_representation_error_sq"]))
    ]
    summary["mean_label_representation_error_sq"] = (
        sum(label_errors) / len(label_errors) if label_errors else None
    )
    total_successes = sum(success_counts)
    if client_label_proportions is not None and total_successes > 0:
        for label in range(10):
            value = sum(
                success_counts[cid] * client_label_proportions[cid][label]
                for cid in range(num_clients)
            ) / total_successes
            summary[
                f"run_successful_client_mean_label_proportion_{label}"
            ] = value
    if population is not None:
        summary.update(population.summary())
    with open(os.path.join(result_dir, "final_summary.json"), "w") as jf:
        json.dump(summary, jf, indent=2)

    if config_dict.get("generate_run_plots", True):
        _plot_episode_results(result_dir)
    if target_accuracies:
        write_target_metrics(
            result_dir,
            target_accuracies,
            {
                "seed": seed,
                "policy": policy.name,
                "num_clients": num_clients,
                "num_uplink_channels": num_uplink_channels,
                "fragmentation_level": fragmentation_level,
                "uplink_fragmentation_level": uplink_fragmentation_level,
                "downlink_fragmentation_level": downlink_fragmentation_level,
                "ideal_downlink": ideal_downlink,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "local_epochs": local_epochs,
                "local_lr": local_lr,
                "server_lr": server_lr,
            },
        )

    return summary

# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------
def _link_fragmentation_configs(args) -> list[dict]:
    """Return directional payload-budget configurations from CLI input.

    Legacy ``--fragmentation_levels`` keeps its prior behavior by setting the
    same budget on both links.  Explicit pairs make a directional study
    reproducible without creating a Cartesian product by accident.
    """
    if not args.link_fragmentation_pairs:
        return [
            {
                "uplink_fragmentation_level": level,
                "downlink_fragmentation_level": level,
                "ideal_downlink": False,
            }
            for level in args.fragmentation_levels
        ]

    configs = []
    seen = set()
    for raw_pair in args.link_fragmentation_pairs:
        parts = raw_pair.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid link-fragmentation pair {raw_pair!r}; use UL:DL or UL:DL:ideal")
        try:
            uplink_level, downlink_level = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid link-fragmentation pair {raw_pair!r}; levels must be integers") from exc
        ideal_downlink = len(parts) == 3 and parts[2].lower() == "ideal"
        if len(parts) == 3 and not ideal_downlink:
            raise ValueError(
                f"Invalid link-fragmentation pair {raw_pair!r}; third component must be 'ideal'")
        if uplink_level <= 0 or downlink_level <= 0:
            raise ValueError("Link fragmentation levels must be positive")
        key = (uplink_level, downlink_level, ideal_downlink)
        if key not in seen:
            seen.add(key)
            configs.append({
                "uplink_fragmentation_level": uplink_level,
                "downlink_fragmentation_level": downlink_level,
                "ideal_downlink": ideal_downlink,
            })
    return configs


def run_policy_comparison(args):
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    base_dir   = os.path.join("results", f"{timestamp}_{args.output_label}")
    os.makedirs(base_dir, exist_ok=True)

    policy_options = {
        "adaptive_complete": RoundPolicyConfig(
            policy_type = RoundPolicyType.ADAPTIVE_COMPLETE,
            window_s    = args.window_s,
        ),
        "fixed_window": RoundPolicyConfig(
            policy_type = RoundPolicyType.FIXED_WINDOW,
            window_s    = args.window_s,
        ),
        "fixed_k_timeout": RoundPolicyConfig(
            policy_type          = RoundPolicyType.FIXED_K_TIMEOUT,
            window_s             = args.window_s,
            target_successes_k   = args.target_k,
        ),
    }
    if args.uplink_policy == "all":
        policies = list(policy_options.values())
    else:
        policies = [policy_options[args.uplink_policy]]

    config_dict = {
        "T_end_s":          args.T_end_s,
        "window_s":         args.window_s,
        "target_k":         args.target_k,
        "num_clients":      args.num_clients,
        "num_uplink_channels": args.num_uplink_channels,
        "dataset_name":     args.dataset,
        "model_name":       args.model,
        "base_filters":     getattr(args, "base_filters", None),
        "model_norm":       getattr(args, "model_norm", None),
        "class_subset":     args.classes,
        "num_classes":      len(args.classes) if args.classes else 10,
        "partition":        args.partition,
        "dirichlet_alpha":  args.dirichlet_alpha,
        "partition_seed":   args.partition_seed,
        "local_epochs":     args.local_epochs,
        "local_lr":         args.local_lr,
        "server_lr":        args.server_lr,
        "warm_start_target_accuracy": args.warm_start_target_accuracy,
        "warm_start_max_epochs": args.warm_start_max_epochs,
        "warm_start_max_batches_per_epoch": args.warm_start_max_batches_per_epoch,
        "warm_start_eval_interval_batches": args.warm_start_eval_interval_batches,
        "warm_start_lr": args.warm_start_lr,
        "target_accuracies": args.target_accuracies,
        "stop_at_target_accuracy": args.stop_at_target_accuracy,
        "spreading_factor": args.sf,
        "fragmentation_levels": args.fragmentation_levels,
        "fragment_count": (
            args.fragment_count
            if args.fragment_count is not None
            else args.fragmentation_levels
        ),
        "link_fragmentation_pairs": args.link_fragmentation_pairs,
        "FEC_rate":         args.FEC_rate,
        "uplink_policy":    args.uplink_policy,
        "contention_jitter_s": args.contention_jitter_s,
        "initial_jitter_window_s": args.initial_jitter_window_s,
        "enable_fragment_jitter": args.enable_fragment_jitter,
        "fragment_jitter_mode": args.fragment_jitter_mode,
        "fragment_jitter_window_s": args.fragment_jitter_window_s,
        "duty_cycle": args.duty_cycle,
        "aggregation_time_s": args.aggregation_time_s,
        "sampling_ratio":   args.participation_probability,
        "participation_probability": args.participation_probability,
        "ping_periodicity": args.ping_periodicity,
        "use_sparsity":     True,
        "use_quantization": True,
        "quantization_bits": args.quantization_bits,
        "sparsity_threshold": 0.001,
        "apply_zlib":       args.apply_zlib,
        "tx_power_w":       args.tx_power_w,
        "max_client_packet_attempts": args.max_client_packet_attempts,
        "max_client_ul_airtime_s": args.max_client_ul_airtime_s,
        "max_client_tx_energy_j": args.max_client_tx_energy_j,
        "max_round_packet_attempts": args.max_round_packet_attempts,
        "max_round_ul_airtime_s": args.max_round_ul_airtime_s,
        "collision_free_ablation": bool(getattr(args, "collision_free_ablation", False)),
        "aggregation_policy": args.aggregation_policy,
        "partial_fragment_aggregation": args.aggregation_policy != "discard_all",
        "coordinate_wise_fragment_normalization": (
            args.aggregation_policy == "masked_partial"
        ),
        "num_monte_carlo_runs": args.num_monte_carlo_runs,
    }

    # Save config
    with open(os.path.join(base_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    _write_experiment_purpose(base_dir, config_dict, policies)

    # Load datasets once; share across all runs
    print("Loading datasets …")
    trainloaders, _, testloader = load_datasets(
        args.num_clients,
        dataset_name=args.dataset,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        partition_seed=args.partition_seed,
        class_subset=args.classes,
    )

    initial_state_dict = None
    if args.warm_start_target_accuracy > 0.0:
        print(
            "Warm-starting global model to "
            f"{args.warm_start_target_accuracy:.3f} accuracy …")
        set_all_seeds(args.partition_seed)
        warm_model = create_model(
            args.model,
            args.dataset,
            len(args.classes) if args.classes else 10,
            **(
                {"base_filters": args.base_filters}
                if getattr(args, "base_filters", None) is not None else {}
            ),
            **(
                {"norm": args.model_norm}
                if getattr(args, "model_norm", None) is not None else {}
            ),
        ).to(DEVICE)
        warm_history = warm_start_to_accuracy(
            warm_model,
            trainloaders,
            testloader,
            target_accuracy=args.warm_start_target_accuracy,
            max_epochs=args.warm_start_max_epochs,
            max_batches_per_epoch=args.warm_start_max_batches_per_epoch,
            eval_interval_batches=args.warm_start_eval_interval_batches,
            lr=args.warm_start_lr,
        )
        warm_path = os.path.join(base_dir, "warm_start_model.pt")
        torch.save(warm_model.state_dict(), warm_path)
        initial_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in warm_model.state_dict().items()
        }

        warm_csv_path = os.path.join(base_dir, "warm_start_history.csv")
        with open(warm_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["epoch", "test_loss", "test_accuracy", "batches_trained"],
            )
            writer.writeheader()
            writer.writerows(warm_history)

        warm_summary = {
            "target_accuracy": args.warm_start_target_accuracy,
            "achieved_accuracy": warm_history[-1]["test_accuracy"],
            "achieved_loss": warm_history[-1]["test_loss"],
            "epochs_run": warm_history[-1]["epoch"],
            "checkpoint": warm_path,
        }
        with open(os.path.join(base_dir, "warm_start_summary.json"), "w") as fh:
            json.dump(warm_summary, fh, indent=2)
        config_dict["warm_start_achieved_accuracy"] = warm_summary["achieved_accuracy"]
        config_dict["warm_start_epochs_run"] = warm_summary["epochs_run"]
        with open(os.path.join(base_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        print(
            "Warm-start complete: "
            f"acc={warm_summary['achieved_accuracy']:.4f}, "
            f"epochs={warm_summary['epochs_run']}")

    all_summaries = []

    link_configs = _link_fragmentation_configs(args)
    for link_config in link_configs:
        uplink_level = link_config["uplink_fragmentation_level"]
        downlink_level = link_config["downlink_fragmentation_level"]
        ideal_downlink = link_config["ideal_downlink"]
        frag_config = {
            **config_dict,
            **link_config,
            # Preserve historical tools that expect this field as the UL level.
            "fragmentation_level": uplink_level,
        }
        for policy in policies:
            for seed in args.seeds:
                link_label = f"ul_{uplink_level}_dl_{downlink_level}"
                if ideal_downlink:
                    link_label += "_ideal"
                result_dir = os.path.join(
                    base_dir, link_label, policy.name, f"seed_{seed}")
                os.makedirs(result_dir, exist_ok=True)
                summary = run_simulation_time_based(
                    frag_config, policy, seed, result_dir,
                    trainloaders, testloader, initial_state_dict)
                all_summaries.append(summary)

    # Write comparison summary CSV
    if all_summaries:
        cmp_path   = os.path.join(base_dir, "comparison_summary.csv")
        cmp_fields = list(all_summaries[0].keys())
        with open(cmp_path, "w", newline="") as cf:
            cw = csv.DictWriter(cf, fieldnames=cmp_fields)
            cw.writeheader()
            for row in all_summaries:
                cw.writerow(row)
        _plot_comparison_results(base_dir)
        if args.link_fragmentation_pairs:
            write_directional_summary(base_dir, all_summaries)
        else:
            from .results import _plot_fragmentation_results, _plot_target_results
            _plot_fragmentation_results(base_dir)
            _plot_target_results(base_dir)

    print(f"\n{'='*70}")
    print(f"  Results saved to: {base_dir}")
    print(f"{'='*70}\n")
    return base_dir

# ---------------------------------------------------------------------------
