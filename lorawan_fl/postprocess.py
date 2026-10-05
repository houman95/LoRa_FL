"""Postprocessing helpers for saved LoRaWAN FL experiment results."""

from __future__ import annotations

import csv
import json
import os
from glob import glob
from typing import Dict, Iterable, List, Optional


def _float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: Dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _cumulative_successful_updates(rows: List[Dict[str, str]], episode_idx: int) -> int:
    return sum(_int(row, "num_successes") for row in rows
               if _int(row, "episode_idx") <= episode_idx)


def write_accuracy_target_tradeoffs(
    base_dir: str,
    targets: Iterable[float],
    output_name: str = "target_accuracy_tradeoff_summary.csv",
) -> str:
    """Write first-hit time and energy metrics for saved episode logs.

    The observed columns use the first logged episode at or above the target.
    The interpolated columns linearly interpolate between the previous logged
    episode and the first hit, which is useful because accuracy is evaluated
    once per completed communication round.
    """
    target_values = list(targets)
    fields = [
        "policy",
        "fragmentation_level",
        "seed",
        "target_accuracy",
        "reached",
        "first_episode_reached",
        "observed_wall_clock_s",
        "observed_tx_energy_j",
        "observed_total_airtime_s",
        "observed_packet_successes",
        "observed_successful_updates",
        "observed_test_accuracy",
        "interpolated_wall_clock_s",
        "interpolated_tx_energy_j",
        "interpolated_total_airtime_s",
        "interpolated_packet_successes",
        "interpolated_successful_updates",
    ]
    out_rows: List[Dict[str, object]] = []

    pattern = os.path.join(base_dir, "frag_*", "*", "seed_*", "episode_metrics.csv")
    for ep_path in sorted(glob(pattern)):
        with open(ep_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue

        first_row = rows[0]
        policy = first_row["policy"]
        fragmentation_level = int(first_row["fragmentation_level"])
        seed = first_row["seed"]

        for target in target_values:
            hit: Optional[Dict[str, str]] = None
            previous: Optional[Dict[str, str]] = None
            for row in rows:
                if _float(row, "test_accuracy") >= target:
                    hit = row
                    break
                previous = row

            out: Dict[str, object] = {
                "policy": policy,
                "fragmentation_level": fragmentation_level,
                "seed": seed,
                "target_accuracy": target,
                "reached": int(hit is not None),
                "first_episode_reached": "",
                "observed_wall_clock_s": "",
                "observed_tx_energy_j": "",
                "observed_total_airtime_s": "",
                "observed_packet_successes": "",
                "observed_successful_updates": "",
                "observed_test_accuracy": "",
                "interpolated_wall_clock_s": "",
                "interpolated_tx_energy_j": "",
                "interpolated_total_airtime_s": "",
                "interpolated_packet_successes": "",
                "interpolated_successful_updates": "",
            }
            if hit is None:
                out_rows.append(out)
                continue

            hit_episode = _int(hit, "episode_idx")
            observed_updates = _cumulative_successful_updates(rows, hit_episode)
            out.update({
                "first_episode_reached": hit_episode,
                "observed_wall_clock_s": hit["sim_time_s"],
                "observed_tx_energy_j": hit["cumulative_tx_energy_j"],
                "observed_total_airtime_s": hit["cumulative_total_airtime_s"],
                "observed_packet_successes": hit["cumulative_packet_successes"],
                "observed_successful_updates": observed_updates,
                "observed_test_accuracy": hit["test_accuracy"],
            })

            if previous is None:
                interpolated = {
                    "sim_time_s": _float(hit, "sim_time_s"),
                    "cumulative_tx_energy_j": _float(hit, "cumulative_tx_energy_j"),
                    "cumulative_total_airtime_s": _float(hit, "cumulative_total_airtime_s"),
                    "cumulative_packet_successes": _float(hit, "cumulative_packet_successes"),
                    "successful_updates": float(observed_updates),
                }
            else:
                prev_acc = _float(previous, "test_accuracy")
                hit_acc = _float(hit, "test_accuracy")
                fraction = 0.0 if hit_acc == prev_acc else (target - prev_acc) / (hit_acc - prev_acc)
                fraction = max(0.0, min(1.0, fraction))

                def lerp(key: str) -> float:
                    return _float(previous, key) + fraction * (_float(hit, key) - _float(previous, key))

                prev_updates = _cumulative_successful_updates(rows, _int(previous, "episode_idx"))
                interpolated = {
                    "sim_time_s": lerp("sim_time_s"),
                    "cumulative_tx_energy_j": lerp("cumulative_tx_energy_j"),
                    "cumulative_total_airtime_s": lerp("cumulative_total_airtime_s"),
                    "cumulative_packet_successes": lerp("cumulative_packet_successes"),
                    "successful_updates": prev_updates + fraction * _int(hit, "num_successes"),
                }

            out.update({
                "interpolated_wall_clock_s": f"{interpolated['sim_time_s']:.3f}",
                "interpolated_tx_energy_j": f"{interpolated['cumulative_tx_energy_j']:.6f}",
                "interpolated_total_airtime_s": f"{interpolated['cumulative_total_airtime_s']:.6f}",
                "interpolated_packet_successes": f"{interpolated['cumulative_packet_successes']:.3f}",
                "interpolated_successful_updates": f"{interpolated['successful_updates']:.3f}",
            })
            out_rows.append(out)

    out_rows.sort(key=lambda row: (
        float(row["target_accuracy"]),
        int(row["fragmentation_level"]),
        str(row["policy"]),
        str(row["seed"]),
    ))
    out_path = os.path.join(base_dir, output_name)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_path


def _trapezoid_area(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for idx in range(1, len(xs)):
        width = xs[idx] - xs[idx - 1]
        if width > 0:
            area += width * (ys[idx] + ys[idx - 1]) / 2.0
    return area


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def write_fundamental_tradeoffs(
    base_dir: str,
    output_name: str = "fundamental_tradeoff_summary.csv",
) -> str:
    """Write communication/learning efficiency metrics for saved runs.

    These metrics avoid selecting target accuracies. They summarize the core
    trade-off between update fidelity and update throughput under a fixed
    communication budget.
    """
    fields = [
        "policy",
        "fragmentation_level",
        "seed",
        "baseline_accuracy",
        "first_logged_accuracy",
        "final_accuracy",
        "accuracy_gain",
        "normalized_time_auc",
        "normalized_energy_auc",
        "normalized_airtime_auc",
        "gain_time_auc",
        "gain_energy_auc",
        "gain_airtime_auc",
        "total_wall_clock_s",
        "total_tx_energy_j",
        "total_airtime_s",
        "total_packet_successes",
        "total_packet_attempts",
        "total_attempted_updates",
        "total_successful_updates",
        "successful_update_ratio",
        "successful_updates_per_packet",
        "packets_per_attempted_update",
        "packets_per_successful_update",
        "successful_packet_overhead_per_delivered_update",
        "tx_energy_j_per_successful_update",
        "airtime_s_per_successful_update",
        "accuracy_gain_per_hour",
        "accuracy_gain_per_joule",
        "accuracy_gain_per_airtime_s",
        "accuracy_gain_per_1000_packets",
        "accuracy_gain_per_successful_update",
    ]
    out_rows: List[Dict[str, object]] = []
    baseline_accuracy: Optional[float] = None
    warm_summary_path = os.path.join(base_dir, "warm_start_summary.json")
    if os.path.exists(warm_summary_path):
        with open(warm_summary_path) as fh:
            baseline_accuracy = float(json.load(fh)["achieved_accuracy"])

    pattern = os.path.join(base_dir, "frag_*", "*", "seed_*", "episode_metrics.csv")
    for ep_path in sorted(glob(pattern)):
        with open(ep_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue

        first = rows[0]
        last = rows[-1]
        policy = first["policy"]
        fragmentation_level = int(first["fragmentation_level"])
        seed = first["seed"]

        run_baseline_accuracy = (
            baseline_accuracy
            if baseline_accuracy is not None
            else _float(first, "test_accuracy")
        )
        first_logged_accuracy = _float(first, "test_accuracy")
        times = [0.0] + [_float(row, "sim_time_s") for row in rows]
        energies = [0.0] + [_float(row, "cumulative_tx_energy_j") for row in rows]
        airtimes = [0.0] + [_float(row, "cumulative_total_airtime_s") for row in rows]
        accuracies = [run_baseline_accuracy] + [_float(row, "test_accuracy") for row in rows]

        final_accuracy = accuracies[-1]
        accuracy_gain = final_accuracy - run_baseline_accuracy
        total_time = times[-1]
        total_energy = energies[-1]
        total_airtime = airtimes[-1]
        total_packet_successes = _int(last, "cumulative_packet_successes")
        total_packet_attempts = _int(last, "cumulative_packet_attempts")
        total_attempted_updates = sum(_int(row, "num_attempts") for row in rows)
        total_successful_updates = sum(_int(row, "num_successes") for row in rows)

        gains = [acc - run_baseline_accuracy for acc in accuracies]
        time_auc = _trapezoid_area(times, accuracies)
        energy_auc = _trapezoid_area(energies, accuracies)
        airtime_auc = _trapezoid_area(airtimes, accuracies)
        gain_time_auc = _trapezoid_area(times, gains)
        gain_energy_auc = _trapezoid_area(energies, gains)
        gain_airtime_auc = _trapezoid_area(airtimes, gains)

        def safe_div(num: float, denom: float) -> float:
            return num / denom if denom else 0.0

        out_rows.append({
            "policy": policy,
            "fragmentation_level": fragmentation_level,
            "seed": seed,
            "baseline_accuracy": f"{run_baseline_accuracy:.6f}",
            "first_logged_accuracy": f"{first_logged_accuracy:.6f}",
            "final_accuracy": f"{final_accuracy:.6f}",
            "accuracy_gain": f"{accuracy_gain:.6f}",
            "normalized_time_auc": f"{safe_div(time_auc, total_time):.6f}",
            "normalized_energy_auc": f"{safe_div(energy_auc, total_energy):.6f}",
            "normalized_airtime_auc": f"{safe_div(airtime_auc, total_airtime):.6f}",
            "gain_time_auc": f"{safe_div(gain_time_auc, total_time):.6f}",
            "gain_energy_auc": f"{safe_div(gain_energy_auc, total_energy):.6f}",
            "gain_airtime_auc": f"{safe_div(gain_airtime_auc, total_airtime):.6f}",
            "total_wall_clock_s": f"{total_time:.6f}",
            "total_tx_energy_j": f"{total_energy:.6f}",
            "total_airtime_s": f"{total_airtime:.6f}",
            "total_packet_successes": total_packet_successes,
            "total_packet_attempts": total_packet_attempts,
            "total_attempted_updates": total_attempted_updates,
            "total_successful_updates": total_successful_updates,
            "successful_update_ratio": f"{safe_div(total_successful_updates, total_attempted_updates):.6f}",
            "successful_updates_per_packet": f"{safe_div(total_successful_updates, total_packet_successes):.6f}",
            "packets_per_attempted_update": f"{safe_div(total_packet_attempts, total_attempted_updates):.6f}",
            "packets_per_successful_update": f"{safe_div(total_packet_successes, total_successful_updates):.6f}",
            "successful_packet_overhead_per_delivered_update": f"{safe_div(total_packet_successes, total_successful_updates) - fragmentation_level:.6f}",
            "tx_energy_j_per_successful_update": f"{safe_div(total_energy, total_successful_updates):.6f}",
            "airtime_s_per_successful_update": f"{safe_div(total_airtime, total_successful_updates):.6f}",
            "accuracy_gain_per_hour": f"{safe_div(accuracy_gain, total_time / 3600.0):.6f}",
            "accuracy_gain_per_joule": f"{safe_div(accuracy_gain, total_energy):.6f}",
            "accuracy_gain_per_airtime_s": f"{safe_div(accuracy_gain, total_airtime):.9f}",
            "accuracy_gain_per_1000_packets": f"{1000.0 * safe_div(accuracy_gain, total_packet_successes):.6f}",
            "accuracy_gain_per_successful_update": f"{safe_div(accuracy_gain, total_successful_updates):.9f}",
        })

    out_rows.sort(key=lambda row: (
        str(row["policy"]),
        int(row["fragmentation_level"]),
        str(row["seed"]),
    ))
    out_path = os.path.join(base_dir, output_name)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    return out_path


def write_fundamental_tradeoff_aggregate(
    base_dir: str,
    per_seed_name: str = "fundamental_tradeoff_summary.csv",
    output_name: str = "fundamental_tradeoff_aggregate.csv",
) -> str:
    """Average fundamental trade-off metrics over seeds."""
    per_seed_path = os.path.join(base_dir, per_seed_name)
    if not os.path.exists(per_seed_path):
        write_fundamental_tradeoffs(base_dir, per_seed_name)

    with open(per_seed_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    numeric_metrics = [
        "final_accuracy",
        "accuracy_gain",
        "normalized_time_auc",
        "normalized_energy_auc",
        "normalized_airtime_auc",
        "gain_time_auc",
        "gain_energy_auc",
        "gain_airtime_auc",
        "total_wall_clock_s",
        "total_tx_energy_j",
        "total_airtime_s",
        "total_packet_successes",
        "total_packet_attempts",
        "total_attempted_updates",
        "total_successful_updates",
        "successful_update_ratio",
        "successful_updates_per_packet",
        "packets_per_attempted_update",
        "packets_per_successful_update",
        "successful_packet_overhead_per_delivered_update",
        "tx_energy_j_per_successful_update",
        "airtime_s_per_successful_update",
        "accuracy_gain_per_hour",
        "accuracy_gain_per_joule",
        "accuracy_gain_per_airtime_s",
        "accuracy_gain_per_1000_packets",
        "accuracy_gain_per_successful_update",
    ]

    grouped: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["policy"], int(row["fragmentation_level"]))
        grouped.setdefault(key, []).append(row)

    fields = ["policy", "fragmentation_level", "num_seeds"]
    for metric in numeric_metrics:
        fields.extend([f"mean_{metric}", f"std_{metric}"])

    aggregate_rows: List[Dict[str, object]] = []
    for (policy, fragmentation_level), group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        out: Dict[str, object] = {
            "policy": policy,
            "fragmentation_level": fragmentation_level,
            "num_seeds": len(group_rows),
        }
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in group_rows]
            out[f"mean_{metric}"] = f"{_mean(values):.6f}"
            out[f"std_{metric}"] = f"{_sample_std(values):.6f}"
        aggregate_rows.append(out)

    out_path = os.path.join(base_dir, output_name)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return out_path


def plot_seed_averaged_learning_curves(
    base_dir: str,
    output_name: str = "seed_averaged_learning_curves_no_extrapolation.png",
) -> str:
    """Plot mean accuracy over seeds versus wall-clock time and TX energy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"Could not import plotting dependencies: {exc}") from exc

    baseline_accuracy: Optional[float] = None
    warm_summary_path = os.path.join(base_dir, "warm_start_summary.json")
    if os.path.exists(warm_summary_path):
        with open(warm_summary_path) as fh:
            baseline_accuracy = float(json.load(fh)["achieved_accuracy"])

    t_end_s = None
    config_path = os.path.join(base_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as fh:
            t_end_s = float(json.load(fh).get("T_end_s", 0.0)) or None

    grouped: Dict[int, List[Dict[str, List[float]]]] = {}
    pattern = os.path.join(base_dir, "frag_*", "*", "seed_*", "episode_metrics.csv")
    for ep_path in sorted(glob(pattern)):
        with open(ep_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        frag = int(rows[0]["fragmentation_level"])
        base_acc = baseline_accuracy if baseline_accuracy is not None else _float(rows[0], "test_accuracy")
        grouped.setdefault(frag, []).append({
            "time_s": [0.0] + [_float(row, "sim_time_s") for row in rows],
            "energy_j": [0.0] + [_float(row, "cumulative_tx_energy_j") for row in rows],
            "accuracy": [base_acc] + [_float(row, "test_accuracy") for row in rows],
        })

    if not grouped:
        raise FileNotFoundError(f"No episode_metrics.csv files found under {base_dir}")

    max_time = t_end_s if t_end_s is not None else max(
        max(run["time_s"]) for runs in grouped.values() for run in runs)
    time_grid = np.linspace(0.0, max_time, 300)
    max_energy = max(max(run["energy_j"]) for runs in grouped.values() for run in runs)
    energy_grid = np.linspace(0.0, max_energy, 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(grouped)))

    def interp_without_extrapolation(grid, xs, ys):
        return np.interp(grid, xs, ys, left=np.nan, right=np.nan)

    def nan_mean_std(series):
        arr = np.vstack(series)
        valid = np.sum(np.isfinite(arr), axis=0)
        mean = np.full(arr.shape[1], np.nan)
        std = np.zeros(arr.shape[1])
        for idx in range(arr.shape[1]):
            vals = arr[:, idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            mean[idx] = vals.mean()
            if len(vals) > 1:
                std[idx] = vals.std(ddof=1)
        return mean, std, valid

    for color, frag in zip(colors, sorted(grouped)):
        time_series = []
        energy_series = []
        for run in grouped[frag]:
            time_series.append(interp_without_extrapolation(
                time_grid, run["time_s"], run["accuracy"]))
            energy_series.append(interp_without_extrapolation(
                energy_grid, run["energy_j"], run["accuracy"]))

        time_mean, time_std, _ = nan_mean_std(time_series)
        energy_mean, energy_std, _ = nan_mean_std(energy_series)
        time_mean *= 100.0
        time_std *= 100.0
        energy_mean *= 100.0
        energy_std *= 100.0

        label = f"{frag} fragments"
        axes[0].plot(time_grid / 3600.0, time_mean, color=color, linewidth=2, label=label)
        axes[0].fill_between(
            time_grid / 3600.0,
            time_mean - time_std,
            time_mean + time_std,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axes[1].plot(energy_grid, energy_mean, color=color, linewidth=2, label=label)
        axes[1].fill_between(
            energy_grid,
            energy_mean - energy_std,
            energy_mean + energy_std,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    axes[0].set_xlabel("Simulated wall-clock time (hours)")
    axes[1].set_xlabel("Cumulative uplink TX energy proxy (J)")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Accuracy vs. time")
    axes[1].set_title("Accuracy vs. uplink TX energy")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[1].legend(title="Fragmentation", fontsize=8, title_fontsize=9)
    fig.suptitle("Seed-averaged mild-heterogeneity learning curves", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(base_dir, output_name)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fundamental_tradeoff_aggregate(
    base_dir: str,
    aggregate_name: str = "fundamental_tradeoff_aggregate.csv",
    output_name: str = "seed_averaged_fundamental_tradeoffs.png",
) -> str:
    """Plot aggregate trade-off metrics with seed standard-deviation bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"Could not import plotting dependencies: {exc}") from exc

    aggregate_path = os.path.join(base_dir, aggregate_name)
    if not os.path.exists(aggregate_path):
        write_fundamental_tradeoff_aggregate(base_dir, output_name=aggregate_name)

    with open(aggregate_path, newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda row: int(row["fragmentation_level"]))
    if not rows:
        raise FileNotFoundError(f"No aggregate rows found in {aggregate_path}")

    frags = [int(row["fragmentation_level"]) for row in rows]
    x = np.arange(len(frags))

    panels = [
        ("mean_final_accuracy", "std_final_accuracy", 100.0, "Final accuracy (%)"),
        ("mean_accuracy_gain_per_joule", "std_accuracy_gain_per_joule", 100.0, "Accuracy gain (pp/J)"),
        ("mean_successful_update_ratio", "std_successful_update_ratio", 100.0, "Delivered updates (%)"),
        ("mean_packets_per_successful_update", "std_packets_per_successful_update", 1.0, "Packets per delivered update"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    bar_color = "#4c78a8"
    err_color = "#222222"
    for ax, (mean_key, std_key, scale, title) in zip(axes, panels):
        means = np.array([float(row[mean_key]) * scale for row in rows])
        stds = np.array([float(row[std_key]) * scale for row in rows])
        ax.bar(x, means, color=bar_color, alpha=0.86)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=err_color, capsize=4, linewidth=1.3)
        ax.set_title(title)
        ax.set_xticks(x, [str(frag) for frag in frags])
        ax.set_xlabel("Fragments")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Seed-averaged mild-heterogeneity trade-offs", y=1.01)
    fig.tight_layout()

    out_path = os.path.join(base_dir, output_name)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
