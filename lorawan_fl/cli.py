"""Command-line interface for the LoRaWAN FL policy comparison."""

from __future__ import annotations

import argparse

from .experiment import run_policy_comparison

def parse_args():
    p = argparse.ArgumentParser(
        description="Federated Learning over LoRaWAN Class B — policy comparison")
    p.add_argument("--T_end_s",          type=float, default=3600.0,
                   help="Simulation time horizon (s)")
    p.add_argument("--window_s",         type=float, default=120.0,
                   help="Max uplink collection window W (s); shared by both policies")
    p.add_argument("--target_k",         type=int,   default=5,
                   help="K for fixed-K-with-timeout policy")
    p.add_argument("--num_clients",      type=int,   default=20)
    p.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"],
                   default="mnist",
                   help="Learning dataset; fashion_mnist is the recommended harder baseline")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="Optional original labels to retain and remap to 0..C-1")
    p.add_argument("--model", choices=["tinycnn", "smallcnn", "parametriccnn"],
                   default="tinycnn", help="Model architecture")
    p.add_argument("--base_filters", type=int, default=None,
                   help=("Width knob for parametriccnn (filter depths are "
                         "base, 2x base, 4x base). Required for parametriccnn; "
                         "rejected by the fixed-width models."))
    p.add_argument("--model_norm", choices=["batch", "group"], default=None,
                   help=("Normalisation for parametriccnn. 'batch' keeps running "
                         "statistics in buffers, which are never transmitted or "
                         "aggregated; 'group' has no buffers and federates cleanly."))
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid",
                   help="Client data partitioning")
    p.add_argument("--dirichlet_alpha", type=float, default=0.3,
                   help="Lower values make non-IID label skew stronger")
    p.add_argument("--partition_seed", type=int, default=42)
    p.add_argument("--local_epochs", type=int, default=1,
                   help="Local training epochs per successful client update")
    p.add_argument("--local_lr", type=float, default=0.01,
                   help="Local client optimizer learning rate")
    p.add_argument("--server_lr", type=float, default=1.0,
                   help="Scale applied to the averaged server update")
    p.add_argument("--warm_start_target_accuracy", type=float, default=0.0,
                   help="Centrally pretrain the initial global model to this accuracy before FL")
    p.add_argument("--warm_start_max_epochs", type=int, default=5,
                   help="Maximum central warm-start epochs")
    p.add_argument("--warm_start_max_batches_per_epoch", type=int, default=None,
                   help="Optional cap on warm-start batches per epoch")
    p.add_argument("--warm_start_eval_interval_batches", type=int, default=None,
                   help="Evaluate warm-start every N batches and stop near target")
    p.add_argument("--warm_start_lr", type=float, default=0.001,
                   help="Warm-start optimizer learning rate")
    p.add_argument("--target_accuracies", type=float, nargs="+",
                   default=[0.70, 0.75, 0.80],
                   help="Accuracy targets used for time/energy-to-target summaries")
    p.add_argument("--stop_at_target_accuracy", type=float, default=None,
                   help="Optional early stop when test accuracy reaches this value")
    p.add_argument("--sf",               type=int,   default=7,
                   help="LoRa spreading factor (7-12)")
    p.add_argument("--FEC_rate",         type=float, default=0.5)
    p.add_argument("--uplink_policy", choices=["adaptive_complete", "fixed_window", "fixed_k_timeout", "all"],
                   default="adaptive_complete",
                   help="Uplink aggregation policy; adaptive_complete avoids a fixed collection timeout")
    p.add_argument("--initial_jitter_window_s", type=float, default=None,
                   help="Initial per-client jitter window W_init (default: 34 s)")
    p.add_argument("--contention_jitter_s", type=float, default=None,
                   help="Deprecated alias for --initial_jitter_window_s")
    p.add_argument(
        "--fragment_jitter_mode",
        choices=[
            "initial_only",
            "per_fragment_duty_interval",
            "per_fragment_fixed_window",
            "independent_per_fragment",
        ],
        default="independent_per_fragment",
        help=("Fragment timing: retain initial jitter only, add independent "
              "U(0,D) later-fragment jitter, or use the configured fixed "
              "later-fragment jitter window, or draw an independent jitter "
              "for every fragment"),
    )
    p.add_argument(
        "--fragment_jitter_window_s",
        type=float,
        default=2.0,
        help=("Fixed later-fragment jitter width in seconds when "
              "fragment jitter is enabled"),
    )
    p.add_argument("--enable_fragment_jitter", type=int, choices=[0, 1], default=1,
                   help="1 to add an independent jitter draw before every fragment")
    p.add_argument("--num_uplink_channels", type=int, default=3,
                   help="Number of orthogonal modeled uplink channels (use 1 for one-subchannel experiments)")
    p.add_argument("--duty_cycle", type=float, default=0.01,
                   help="Device/gateway duty-cycle fraction (0.01 means 1%%)")
    p.add_argument("--aggregation_time_s", type=float, default=0.0,
                   help="Server-side aggregation time added to each completed round")
    p.add_argument("--participation_probability", type=float, default=None,
                   help="Independent Bernoulli UL participation probability per eligible client")
    p.add_argument("--sampling_ratio", type=float, default=0.4,
                   help="Deprecated alias for --participation_probability")
    p.add_argument("--ping_periodicity", type=int,   default=0,
                   help="Class B ping periodicity (0=128 ping slots / beacon period)")
    p.add_argument("--quantization_bits",type=int,   default=4)
    p.add_argument("--fragmentation_levels", type=int, nargs="+", default=[1],
                   help="Legacy shared UL/DL payload-budget levels, in LoRa MTUs")
    p.add_argument("--fragment_count", type=int, default=None,
                   help="Single-run alias for one uplink/downlink fragmentation level")
    p.add_argument("--link_fragmentation_pairs", type=str, nargs="+", default=None,
                   help=("Independent payload-budget settings as UL:DL or UL:DL:ideal, "
                         "for example 8:32 32:8 32:32:ideal. Levels are LoRa MTUs."))
    p.add_argument("--output_label", type=str, default="lorawan_fl_policy_comparison",
                   help="Suffix used for the timestamped results directory")
    p.add_argument("--apply_zlib",       type=int,   default=0,
                   help="1 to enable zlib, 0 to disable")
    p.add_argument("--tx_power_w", type=float, default=0.1,
                   help="Transmit power used for airtime-to-energy proxy")
    p.add_argument("--max_client_packet_attempts", type=int, default=None,
                   help="Optional per-client packet-attempt budget")
    p.add_argument("--max_client_ul_airtime_s", type=float, default=None,
                   help="Optional per-client uplink airtime budget")
    p.add_argument("--max_client_tx_energy_j", type=float, default=None,
                   help="Optional per-client transmit-energy budget")
    p.add_argument("--max_round_packet_attempts", type=int, default=None,
                   help="Optional per-round packet-attempt budget")
    p.add_argument("--max_round_ul_airtime_s", type=float, default=None,
                   help="Optional per-round uplink airtime budget")
    p.add_argument("--seeds",            type=int,   nargs="+", default=None,
                   help="Random seeds; both policies are run for each seed")
    p.add_argument("--num_monte_carlo_runs", type=int, default=None,
                   help="Number of realizations; derives seeds 0..N-1 when --seeds is omitted")
    p.add_argument("--collision_free_ablation", type=int, default=0,
                   help=("1 to run the collision-free, timing-preserving ablation: "
                         "scheduling, airtime, duty-cycle waiting, and round timing "
                         "are unchanged, but overlapping fragments are still declared "
                         "successfully received instead of being discarded. Only "
                         "supported for adaptive_complete and fixed_window."))
    p.add_argument("--partial_fragment_aggregation", type=int, default=0,
                   help=("1 to aggregate every attempted local update after "
                         "zeroing coordinates assigned to collided fragments. "
                         "Delivered fragments remain usable and the client-average "
                         "denominator is not renormalized."))
    p.add_argument("--coordinate_wise_fragment_normalization", type=int, default=0,
                   help=("1 to divide each aggregated coordinate by the number "
                         "of clients for which that compressed coordinate was not "
                         "erased by collision. Requires partial fragment aggregation."))
    p.add_argument(
        "--aggregation_policy",
        choices=[
            "discard_all", "legacy_zero_fill", "masked_partial",
            "masked_fixed_denominator",
        ],
        default="masked_partial",
        help=(
            "Reception/aggregation baseline; masked_partial is observed-normalized "
            "and masked_fixed_denominator is coverage-scaled"
        ),
    )
    return p.parse_args()

def main():
    args = parse_args()
    if args.classes is not None:
        if len(args.classes) < 2:
            raise ValueError("--classes requires at least two labels")
        if len(set(args.classes)) != len(args.classes):
            raise ValueError("--classes labels must be unique")
        if any(label < 0 or label > 9 for label in args.classes):
            raise ValueError("--classes labels must be in [0, 9]")
    args.apply_zlib = bool(args.apply_zlib)
    args.collision_free_ablation = bool(args.collision_free_ablation)
    args.partial_fragment_aggregation = bool(args.partial_fragment_aggregation)
    args.coordinate_wise_fragment_normalization = bool(
        args.coordinate_wise_fragment_normalization
    )
    args.enable_fragment_jitter = bool(args.enable_fragment_jitter)
    if args.initial_jitter_window_s is None:
        args.initial_jitter_window_s = (
            args.contention_jitter_s
            if args.contention_jitter_s is not None
            else 34.0
        )
    if args.contention_jitter_s is None:
        args.contention_jitter_s = args.initial_jitter_window_s
    if args.fragment_count is not None:
        if args.fragment_count <= 0:
            raise ValueError("fragment_count must be positive")
        args.fragmentation_levels = [args.fragment_count]
    if args.seeds is None:
        num_runs = args.num_monte_carlo_runs or 3
        if num_runs <= 0:
            raise ValueError("num_monte_carlo_runs must be positive")
        args.seeds = list(range(num_runs))
    elif (
        args.num_monte_carlo_runs is not None
        and args.num_monte_carlo_runs != len(args.seeds)
    ):
        raise ValueError("num_monte_carlo_runs must equal the number of seeds")
    args.num_monte_carlo_runs = len(args.seeds)
    if args.participation_probability is None:
        args.participation_probability = args.sampling_ratio
    args.sampling_ratio = args.participation_probability
    run_policy_comparison(args)
