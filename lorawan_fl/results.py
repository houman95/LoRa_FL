"""CSV logging, experiment descriptions, and result plotting."""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Tuple

import numpy as np

from .config import RoundPolicyConfig

def _open_csv(path: str, fieldnames: List[str]) -> Tuple[object, csv.DictWriter]:
    fh = open(path, "w", newline="")
    w  = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    return fh, w


def _safe_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def write_directional_summary(base_dir: str, summaries: List[Dict[str, object]]) -> None:
    """Aggregate paired directional configurations without collapsing UL and DL."""
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    for row in summaries:
        key = (
            row.get("policy"),
            row.get("uplink_fragmentation_level"),
            row.get("downlink_fragmentation_level"),
            row.get("ideal_downlink"),
        )
        groups.setdefault(key, []).append(row)

    fields = [
        "policy", "num_clients", "num_uplink_channels", "uplink_fragmentation_level", "downlink_fragmentation_level",
        "ideal_downlink", "n_seeds", "seeds", "final_test_accuracy_mean",
        "final_test_accuracy_std", "final_test_accuracy_ci95", "final_sim_time_s_mean",
        "total_ul_successes_mean", "total_ul_collisions_mean",
        "total_packet_attempts_mean", "total_packet_successes_mean",
        "total_packet_collisions_mean", "packet_collision_rate_mean",
        "total_ul_airtime_s_mean", "total_downlink_airtime_s_mean",
        "total_downlink_duration_s_mean", "total_tx_energy_j_mean",
        "target_stop_reached_count",
    ]
    with open(os.path.join(base_dir, "directional_summary.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key, rows in sorted(groups.items(), key=lambda item: item[0]):
            n = len(rows)
            def stats(name: str) -> tuple[float, float, float]:
                values = [float(row.get(name) or 0.0) for row in rows]
                mean = sum(values) / n
                std = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1)) if n > 1 else 0.0
                ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
                return mean, std, ci95

            accuracy_mean, accuracy_std, accuracy_ci95 = stats("final_test_accuracy")
            writer.writerow({
                "policy": key[0],
                "num_clients": rows[0].get("num_clients"),
                "num_uplink_channels": rows[0].get("num_uplink_channels"),
                "uplink_fragmentation_level": key[1],
                "downlink_fragmentation_level": key[2],
                "ideal_downlink": key[3],
                "n_seeds": n,
                "seeds": " ".join(str(row.get("seed")) for row in rows),
                "final_test_accuracy_mean": accuracy_mean,
                "final_test_accuracy_std": accuracy_std,
                "final_test_accuracy_ci95": accuracy_ci95,
                "final_sim_time_s_mean": stats("final_sim_time_s")[0],
                "total_ul_successes_mean": stats("total_ul_successes")[0],
                "total_ul_collisions_mean": stats("total_ul_collisions")[0],
                "total_packet_attempts_mean": stats("total_packet_attempts")[0],
                "total_packet_successes_mean": stats("total_packet_successes")[0],
                "total_packet_collisions_mean": stats("total_packet_collisions")[0],
                "packet_collision_rate_mean": (
                    sum(
                        float(row.get("total_packet_collisions") or 0.0)
                        / max(1.0, float(row.get("total_packet_attempts") or 0.0))
                        for row in rows
                    ) / n
                ),
                "total_ul_airtime_s_mean": stats("total_ul_airtime_s")[0],
                "total_downlink_airtime_s_mean": stats("total_downlink_airtime_s")[0],
                "total_downlink_duration_s_mean": stats("total_downlink_duration_s")[0],
                "total_tx_energy_j_mean": stats("total_tx_energy_j")[0],
                "target_stop_reached_count": sum(bool(row.get("target_stop_reached")) for row in rows),
            })


def _write_experiment_purpose(
    base_dir: str,
    config_dict: dict,
    policies: List[RoundPolicyConfig],
) -> None:
    policy_names = ", ".join(policy.name for policy in policies)
    purpose = (
        "Experiment purpose\n"
        "==================\n\n"
        "This run studies federated learning over a simulated EU868 LoRaWAN "
        "Class-B network. It compares the configured uplink collection policies "
        "while accounting for LoRa airtime, end-device duty-cycle limits, "
        "gateway downlink packetization, and gateway duty-cycle waiting time.\n\n"
        "The goal is to measure the tradeoff between learning quality and "
        "communication cost, especially when larger or more fragmented global "
        "updates make the downlink dissemination phase longer.\n\n"
        "Run configuration\n"
        "-----------------\n"
        f"Policies: {policy_names}\n"
        f"Simulation horizon: {config_dict['T_end_s']} s\n"
        f"Uplink window: {config_dict['window_s']} s\n"
        f"Target K: {config_dict['target_k']}\n"
        f"Clients: {config_dict['num_clients']}\n"
        f"Modeled uplink channels: {config_dict.get('num_uplink_channels', 3)}\n"
        f"Dataset: {config_dict.get('dataset_name', 'mnist')}\n"
        f"Model: {config_dict.get('model_name', 'tinycnn')}\n"
        f"Partition: {config_dict.get('partition', 'iid')}\n"
        f"Dirichlet alpha: {config_dict.get('dirichlet_alpha')}\n"
        f"Local epochs: {config_dict.get('local_epochs', 1)}\n"
        f"Target accuracies: {config_dict.get('target_accuracies', [])}\n"
        f"Spreading factor: SF{config_dict['spreading_factor']}\n"
        f"Legacy shared fragmentation levels: {config_dict['fragmentation_levels']}\n"
        f"Directional link pairs: {config_dict.get('link_fragmentation_pairs')}\n"
        f"FEC rate: {config_dict['FEC_rate']}\n"
        f"Uplink policy: {config_dict['uplink_policy']}\n"
        f"Contention jitter: {config_dict['contention_jitter_s']} s\n"
        f"Aggregation time: {config_dict['aggregation_time_s']} s\n"
        f"Participation probability: {config_dict['participation_probability']}\n"
        f"Quantization bits: {config_dict['quantization_bits']}\n"
        f"TX power proxy: {config_dict.get('tx_power_w', 0.1)} W\n"
        f"Max client packet attempts: {config_dict.get('max_client_packet_attempts')}\n"
        f"Max client uplink airtime: {config_dict.get('max_client_ul_airtime_s')} s\n"
        f"Max client TX energy: {config_dict.get('max_client_tx_energy_j')} J\n"
        f"Max round packet attempts: {config_dict.get('max_round_packet_attempts')}\n"
        f"Max round uplink airtime: {config_dict.get('max_round_ul_airtime_s')} s\n"
    )
    with open(os.path.join(base_dir, "experiment_purpose.txt"), "w", encoding="utf-8") as fh:
        fh.write(purpose)


def _plot_episode_results(result_dir: str) -> None:
    ep_path = os.path.join(result_dir, "episode_metrics.csv")
    if not os.path.exists(ep_path):
        return

    with open(ep_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"    [WARN] Could not generate figures: {exc}")
        return

    times = [_safe_float(row, "sim_time_s") for row in rows]
    losses = [_safe_float(row, "test_loss") for row in rows]
    accs = [_safe_float(row, "test_accuracy") for row in rows]
    dl_durations = [_safe_float(row, "downlink_duration_s") for row in rows]
    ul_durations = [_safe_float(row, "collection_duration_s") for row in rows]
    dl_fragments = [_safe_float(row, "downlink_num_fragments") for row in rows]
    packet_attempts = [_safe_float(row, "cumulative_packet_attempts") for row in rows]
    tx_energy = [_safe_float(row, "cumulative_tx_energy_j") for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(times, losses, marker="o", linewidth=2, color="tab:red")
    axes[0].set_xlabel("Simulation time (s)")
    axes[0].set_ylabel("Test loss")
    axes[0].set_title("Loss vs. wall-clock time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, accs, marker="o", linewidth=2, color="tab:blue")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Accuracy vs. wall-clock time")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "learning_curve.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    episodes = [_safe_float(row, "episode_idx") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(episodes, ul_durations, marker="o", label="Uplink collection")
    axes[0].plot(episodes, dl_durations, marker="s", label="Downlink dissemination")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Duration (s)")
    axes[0].set_title("Round communication timing")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar(episodes, dl_fragments, color="tab:green")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Downlink packets")
    axes[1].set_title("Global update packetization")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "communication_timing.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(packet_attempts, accs, marker="o", linewidth=2, color="tab:purple")
    axes[0].set_xlabel("Cumulative packet attempts")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Accuracy vs. packet energy proxy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tx_energy, accs, marker="o", linewidth=2, color="tab:olive")
    axes[1].set_xlabel("Cumulative TX energy proxy (J)")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Accuracy vs. TX energy proxy")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, "accuracy_vs_energy.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_target_metrics(
    result_dir: str,
    target_accuracies: List[float],
    metadata: Dict[str, object],
) -> None:
    """Write first-hit wall-clock and energy-to-target metrics for one run."""
    ep_path = os.path.join(result_dir, "episode_metrics.csv")
    if not os.path.exists(ep_path):
        return
    with open(ep_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    fields = [
        "seed","policy","fragmentation_level","uplink_fragmentation_level",
        "downlink_fragmentation_level","ideal_downlink","dataset_name","model_name",
        "local_epochs","target_accuracy","reached","first_episode_reached",
        "wall_clock_s_to_target","packet_attempts_to_target",
        "packet_successes_to_target","packet_collisions_to_target",
        "ul_airtime_s_to_target","downlink_airtime_s_to_target",
        "total_airtime_s_to_target","tx_energy_j_to_target",
    ]
    target_path = os.path.join(result_dir, "target_metrics.csv")
    with open(target_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for target in target_accuracies:
            first = None
            for row in rows:
                if _safe_float(row, "test_accuracy", -1.0) >= target:
                    first = row
                    break
            out = {
                "seed": metadata.get("seed"),
                "policy": metadata.get("policy"),
                "fragmentation_level": metadata.get("fragmentation_level"),
                "uplink_fragmentation_level": metadata.get("uplink_fragmentation_level"),
                "downlink_fragmentation_level": metadata.get("downlink_fragmentation_level"),
                "ideal_downlink": metadata.get("ideal_downlink"),
                "dataset_name": metadata.get("dataset_name"),
                "model_name": metadata.get("model_name"),
                "local_epochs": metadata.get("local_epochs"),
                "target_accuracy": target,
                "reached": int(first is not None),
                "first_episode_reached": "",
                "wall_clock_s_to_target": "",
                "packet_attempts_to_target": "",
                "packet_successes_to_target": "",
                "packet_collisions_to_target": "",
                "ul_airtime_s_to_target": "",
                "downlink_airtime_s_to_target": "",
                "total_airtime_s_to_target": "",
                "tx_energy_j_to_target": "",
            }
            if first is not None:
                out.update({
                    "first_episode_reached": first.get("episode_idx", ""),
                    "wall_clock_s_to_target": first.get("sim_time_s", ""),
                    "packet_attempts_to_target": first.get("cumulative_packet_attempts", ""),
                    "packet_successes_to_target": first.get("cumulative_packet_successes", ""),
                    "packet_collisions_to_target": first.get("cumulative_packet_collisions", ""),
                    "ul_airtime_s_to_target": first.get("cumulative_ul_airtime_s", ""),
                    "downlink_airtime_s_to_target": first.get("cumulative_downlink_airtime_s", ""),
                    "total_airtime_s_to_target": first.get("cumulative_total_airtime_s", ""),
                    "tx_energy_j_to_target": first.get("cumulative_tx_energy_j", ""),
                })
            writer.writerow(out)


def _plot_comparison_results(base_dir: str) -> None:
    series_by_policy: Dict[str, List[List[Dict[str, str]]]] = {}
    for root, _, files in os.walk(base_dir):
        if "episode_metrics.csv" not in files:
            continue
        ep_path = os.path.join(root, "episode_metrics.csv")
        with open(ep_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            rel_parts = os.path.relpath(root, base_dir).split(os.sep)
            if rel_parts[0].startswith("frag_") and len(rel_parts) >= 2:
                policy_name = f"{rel_parts[1]} / {rel_parts[0]}"
            else:
                policy_name = rel_parts[0]
            series_by_policy.setdefault(policy_name, []).append(rows)
    if not series_by_policy:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not generate comparison figure: {exc}")
        return

    def average_metric(seed_runs: List[List[Dict[str, str]]], metric: str):
        cleaned_runs = []
        all_times = []
        for rows in seed_runs:
            rows = sorted(rows, key=lambda row: _safe_float(row, "sim_time_s"))
            times = np.array([_safe_float(row, "sim_time_s") for row in rows], dtype=float)
            values = np.array([_safe_float(row, metric) for row in rows], dtype=float)
            keep = np.isfinite(times) & np.isfinite(values)
            times, values = times[keep], values[keep]
            if len(times) == 0:
                continue

            unique_times, unique_indices = np.unique(times, return_index=True)
            values = values[unique_indices]
            cleaned_runs.append((unique_times, values))
            all_times.extend(unique_times.tolist())

        if not cleaned_runs:
            return np.array([]), np.array([]), np.array([])

        time_grid = np.array(sorted(set(all_times)), dtype=float)
        interpolated = []
        for times, values in cleaned_runs:
            interp_values = np.interp(time_grid, times, values)
            interp_values[time_grid < times[0]] = np.nan
            interp_values[time_grid > times[-1]] = np.nan
            interpolated.append(interp_values)

        stacked = np.vstack(interpolated)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        return time_grid, mean, std

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for policy_name, seed_runs in sorted(series_by_policy.items()):
        loss_times, mean_losses, std_losses = average_metric(seed_runs, "test_loss")
        acc_times, mean_accs, std_accs = average_metric(seed_runs, "test_accuracy")
        label = f"{policy_name} (mean of {len(seed_runs)} seeds)"

        if len(loss_times):
            loss_line = axes[0].plot(
                loss_times, mean_losses, marker="o", linewidth=2.2, label=label)
            color = loss_line[0].get_color()
            axes[0].fill_between(
                loss_times, mean_losses - std_losses, mean_losses + std_losses,
                color=color, alpha=0.14, linewidth=0)

        if len(acc_times):
            acc_line = axes[1].plot(
                acc_times, mean_accs, marker="o", linewidth=2.2, label=label)
            color = acc_line[0].get_color()
            axes[1].fill_between(
                acc_times, mean_accs - std_accs, mean_accs + std_accs,
                color=color, alpha=0.14, linewidth=0)

    axes[0].set_xlabel("Simulation time (s)")
    axes[0].set_ylabel("Test loss")
    axes[0].set_title("Policy comparison: mean loss over seeds")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Policy comparison: mean accuracy over seeds")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "comparison_learning_curves.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_fragmentation_results(base_dir: str) -> None:
    summary_path = os.path.join(base_dir, "comparison_summary.csv")
    if not os.path.exists(summary_path):
        return

    with open(summary_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "fragmentation_level" not in rows[0]:
        return

    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    for row in rows:
        try:
            frag = int(float(row["fragmentation_level"]))
        except (TypeError, ValueError):
            continue
        grouped.setdefault((row["policy"], frag), []).append(row)

    if not grouped:
        return

    aggregate_rows = []
    for (policy, frag), policy_rows in sorted(grouped.items()):
        def mean(key: str) -> float:
            values = [_safe_float(row, key, np.nan) for row in policy_rows]
            values = [value for value in values if np.isfinite(value)]
            return float(np.mean(values)) if values else 0.0

        aggregate_rows.append({
            "policy": policy,
            "fragmentation_level": frag,
            "num_seeds": len(policy_rows),
            "mean_final_test_accuracy": mean("final_test_accuracy"),
            "mean_final_test_loss": mean("final_test_loss"),
            "mean_num_episodes": mean("num_episodes"),
            "mean_total_packet_successes": mean("total_packet_successes"),
            "mean_total_packet_attempts": mean("total_packet_attempts"),
            "mean_total_successful_updates": mean("total_successful_updates"),
            "mean_total_selected_updates": mean("total_selected_updates"),
            "mean_total_round_duration_s": mean("total_round_duration_s"),
            "mean_total_ul_airtime_s": mean("total_ul_airtime_s"),
            "mean_total_downlink_airtime_s": mean("total_downlink_airtime_s"),
            "mean_total_tx_energy_j": mean("total_tx_energy_j"),
            "mean_total_communication_time_s": mean("total_communication_time_s"),
            "mean_actual_participation_level": mean("actual_participation_level"),
            "mean_actual_attempt_level": mean("actual_attempt_level"),
        })

    tradeoff_path = os.path.join(base_dir, "fragmentation_tradeoff_summary.csv")
    with open(tradeoff_path, "w", newline="") as fh:
        fieldnames = list(aggregate_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not generate fragmentation figures: {exc}")
        return

    by_policy: Dict[str, List[Dict[str, float]]] = {}
    for row in aggregate_rows:
        by_policy.setdefault(row["policy"], []).append(row)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metric_specs = [
        ("mean_final_test_accuracy", "Final test accuracy"),
        ("mean_final_test_loss", "Final test loss"),
        ("mean_num_episodes", "Completed FL rounds"),
        ("mean_total_communication_time_s", "Total communication time (s)"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metric_specs):
        for policy, policy_rows in sorted(by_policy.items()):
            policy_rows = sorted(policy_rows, key=lambda row: row["fragmentation_level"])
            frags = [row["fragmentation_level"] for row in policy_rows]
            values = [row[metric] for row in policy_rows]
            ax.plot(frags, values, marker="o", linewidth=2.0, label=policy)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Fragmentation level")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 1].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Rounds")
    axes[1, 1].set_ylabel("Seconds")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "fragmentation_performance.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cost_specs = [
        ("mean_total_ul_airtime_s", "Uplink airtime (s)"),
        ("mean_total_downlink_airtime_s", "Downlink airtime (s)"),
        ("mean_total_packet_attempts", "Packet attempts"),
    ]
    for ax, (metric, title) in zip(axes, cost_specs):
        for policy, policy_rows in sorted(by_policy.items()):
            policy_rows = sorted(policy_rows, key=lambda row: row["fragmentation_level"])
            frags = [row["fragmentation_level"] for row in policy_rows]
            values = [row[metric] for row in policy_rows]
            ax.plot(frags, values, marker="o", linewidth=2.0, label=policy)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Fragmentation level")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "fragmentation_communication_costs.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for policy, policy_rows in sorted(by_policy.items()):
        policy_rows = sorted(policy_rows, key=lambda row: row["fragmentation_level"])
        frags = [row["fragmentation_level"] for row in policy_rows]
        values = [row["mean_total_tx_energy_j"] for row in policy_rows]
        ax.plot(frags, values, marker="o", linewidth=2.0, label=policy)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Fragmentation level")
    ax.set_ylabel("TX energy proxy (J)")
    ax.set_title("Fragmentation energy cost")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "fragmentation_energy_costs.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        "Fragmentation turning-point analysis",
        "====================================",
        "",
    ]
    target_participation = None
    config_path = os.path.join(base_dir, "config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, encoding="utf-8") as fh:
                target_participation = json.load(fh).get("participation_probability")
        except Exception:
            target_participation = None

    active_counts = []
    for root, _, files in os.walk(base_dir):
        if "episode_metrics.csv" not in files:
            continue
        with open(os.path.join(root, "episode_metrics.csv"), newline="") as fh:
            for row in csv.DictReader(fh):
                active_counts.append(_safe_float(row, "num_selected", np.nan))

    if active_counts:
        active_counts = [value for value in active_counts if np.isfinite(value)]
        report_lines.append("Validation")
        report_lines.append("----------")
        report_lines.append(
            f"Bernoulli-selected users per round: mean={np.mean(active_counts):.3f}, "
            f"std={np.std(active_counts):.3f}, min={np.min(active_counts):.0f}, "
            f"max={np.max(active_counts):.0f}")
        report_lines.append(
            "Selected-user fluctuation observed: "
            f"{'yes' if np.std(active_counts) > 0 else 'no'}")
        if target_participation is not None:
            achieved_values = [
                row["mean_actual_participation_level"] for row in aggregate_rows
            ]
            achieved = float(np.mean(achieved_values)) if achieved_values else 0.0
            report_lines.append(
                f"Target participation probability: {target_participation:.4f}")
            report_lines.append(
                f"Mean achieved participation level: {achieved:.4f}")

        overhead_increase_checks = []
        for policy, policy_rows in sorted(by_policy.items()):
            policy_rows = sorted(policy_rows, key=lambda row: row["fragmentation_level"])
            overhead = [row["mean_total_communication_time_s"] for row in policy_rows]
            if len(overhead) >= 2:
                overhead_increase_checks.append(
                    all(curr >= prev for prev, curr in zip(overhead, overhead[1:])))
        if overhead_increase_checks:
            report_lines.append(
                "Larger fragmentation increases communication overhead: "
                f"{'yes' if all(overhead_increase_checks) else 'not monotonically in this run'}")
        report_lines.append("")

    for policy, policy_rows in sorted(by_policy.items()):
        policy_rows = sorted(policy_rows, key=lambda row: row["fragmentation_level"])
        best = max(policy_rows, key=lambda row: row["mean_final_test_accuracy"])
        best_idx = policy_rows.index(best)
        later_rows = policy_rows[best_idx + 1:]
        later_decreases = [
            row for row in later_rows
            if row["mean_final_test_accuracy"] < best["mean_final_test_accuracy"]
        ]
        report_lines.append(f"Policy: {policy}")
        report_lines.append(
            f"  N_frag*: {best['fragmentation_level']} "
            f"(mean final accuracy={best['mean_final_test_accuracy']:.4f})")
        if later_decreases:
            report_lines.append("  Turning point observed: yes")
        else:
            report_lines.append(
                "  Turning point observed: no within the swept range; "
                "consider missing costs such as capture/SINR, fragment-level fading, "
                "ACK traffic, retransmissions, receive-window energy, or stricter gateway capacity.")
        report_lines.append("")

    with open(os.path.join(base_dir, "fragmentation_turning_point.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))


def _plot_target_results(base_dir: str) -> None:
    target_rows: List[Dict[str, str]] = []
    for root, _, files in os.walk(base_dir):
        if "target_metrics.csv" not in files:
            continue
        with open(os.path.join(root, "target_metrics.csv"), newline="") as fh:
            target_rows.extend(csv.DictReader(fh))
    if not target_rows:
        return

    grouped: Dict[Tuple[str, int, float], List[Dict[str, str]]] = {}
    for row in target_rows:
        if row.get("reached") != "1":
            continue
        try:
            key = (
                row["policy"],
                int(float(row["fragmentation_level"])),
                float(row["target_accuracy"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(row)
    if not grouped:
        return

    aggregate_rows = []
    for (policy, frag, target), rows in sorted(grouped.items()):
        def mean(key: str) -> float:
            values = [_safe_float(row, key, np.nan) for row in rows]
            values = [value for value in values if np.isfinite(value)]
            return float(np.mean(values)) if values else np.nan

        aggregate_rows.append({
            "policy": policy,
            "fragmentation_level": frag,
            "target_accuracy": target,
            "num_reached_seeds": len(rows),
            "mean_wall_clock_s_to_target": mean("wall_clock_s_to_target"),
            "mean_packet_attempts_to_target": mean("packet_attempts_to_target"),
            "mean_total_airtime_s_to_target": mean("total_airtime_s_to_target"),
            "mean_tx_energy_j_to_target": mean("tx_energy_j_to_target"),
        })

    summary_path = os.path.join(base_dir, "target_tradeoff_summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not generate target tradeoff figure: {exc}")
        return

    targets = sorted({row["target_accuracy"] for row in aggregate_rows})
    fig, axes = plt.subplots(len(targets), 2, figsize=(13, 4.8 * len(targets)), squeeze=False)
    for row_idx, target in enumerate(targets):
        target_subset = [row for row in aggregate_rows if row["target_accuracy"] == target]
        by_policy: Dict[str, List[Dict[str, float]]] = {}
        for row in target_subset:
            by_policy.setdefault(row["policy"], []).append(row)
        for policy, rows in sorted(by_policy.items()):
            rows = sorted(rows, key=lambda row: row["fragmentation_level"])
            frags = [row["fragmentation_level"] for row in rows]
            wall = [row["mean_wall_clock_s_to_target"] for row in rows]
            energy = [row["mean_tx_energy_j_to_target"] for row in rows]
            axes[row_idx, 0].plot(frags, wall, marker="o", linewidth=2.0, label=policy)
            axes[row_idx, 1].plot(frags, energy, marker="o", linewidth=2.0, label=policy)
        axes[row_idx, 0].set_title(f"Wall-clock to target accuracy {target:.2f}")
        axes[row_idx, 1].set_title(f"Energy to target accuracy {target:.2f}")
        axes[row_idx, 0].set_ylabel("Seconds")
        axes[row_idx, 1].set_ylabel("TX energy proxy (J)")
        for ax in axes[row_idx]:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Fragmentation level")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "target_tradeoffs.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
