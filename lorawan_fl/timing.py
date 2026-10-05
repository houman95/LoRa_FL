"""Uplink timing configuration, reproducible RNG streams, and scheduling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Sequence


INITIAL_ONLY = "initial_only"
LEGACY_DUTY_INTERVAL = "per_fragment_duty_interval"
LEGACY_FIXED_WINDOW = "per_fragment_fixed_window"
INDEPENDENT_PER_FRAGMENT = "independent_per_fragment"
INDEPENDENT_LATER_FRAGMENTS = "independent_later_fragments"
TIMING_MODES = {
    INITIAL_ONLY,
    LEGACY_DUTY_INTERVAL,
    LEGACY_FIXED_WINDOW,
    INDEPENDENT_PER_FRAGMENT,
    INDEPENDENT_LATER_FRAGMENTS,
}


def derive_seed(base_seed: int, stream_name: str) -> int:
    """Derive a stable, process-independent 64-bit stream seed."""

    digest = hashlib.sha256(f"{int(base_seed)}:{stream_name}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


@dataclass
class UplinkRandomStreams:
    participation: random.Random
    initial_jitter: random.Random
    fragment_jitter: random.Random
    channel: random.Random
    seeds: dict[str, int]

    @classmethod
    def from_seed(cls, seed: int) -> "UplinkRandomStreams":
        names = (
            "participation",
            "initial_jitter",
            "fragment_jitter",
            "channel",
        )
        seeds = {name: derive_seed(seed, name) for name in names}
        return cls(
            participation=random.Random(seeds["participation"]),
            initial_jitter=random.Random(seeds["initial_jitter"]),
            fragment_jitter=random.Random(seeds["fragment_jitter"]),
            channel=random.Random(seeds["channel"]),
            seeds=seeds,
        )


@dataclass(frozen=True)
class UplinkTimingConfig:
    initial_jitter_window_s: float = 34.0
    fragment_jitter_window_s: float = 2.0
    enable_fragment_jitter: bool = True
    mode: str = INDEPENDENT_PER_FRAGMENT

    def __post_init__(self) -> None:
        if self.initial_jitter_window_s < 0:
            raise ValueError("initial_jitter_window_s must be nonnegative")
        if self.fragment_jitter_window_s < 0:
            raise ValueError("fragment_jitter_window_s must be nonnegative")
        if self.mode not in TIMING_MODES:
            raise ValueError(f"unknown fragment timing mode: {self.mode}")


@dataclass(frozen=True)
class FragmentTiming:
    fragment_index: int
    earliest_allowed_s: float
    start_s: float
    end_s: float
    initial_jitter_s: float
    fragment_jitter_s: float


def timing_config_from_legacy(
    contention_jitter_s: float,
    fragment_jitter_mode: str,
    fragment_jitter_window_s: float,
) -> UplinkTimingConfig:
    """Translate validated historical timing modes without changing them."""

    return UplinkTimingConfig(
        initial_jitter_window_s=float(contention_jitter_s),
        fragment_jitter_window_s=float(fragment_jitter_window_s),
        enable_fragment_jitter=fragment_jitter_mode != INITIAL_ONLY,
        mode=fragment_jitter_mode,
    )


def _fragment_jitters(
    airtimes_s: Sequence[float],
    duty_cycle_percent: float,
    config: UplinkTimingConfig,
    rng: random.Random,
) -> list[float]:
    count = len(airtimes_s)
    if not config.enable_fragment_jitter or config.mode == INITIAL_ONLY:
        return [0.0] * count
    if config.mode == INDEPENDENT_PER_FRAGMENT:
        # A fresh draw is made for every fragment, including fragment zero.
        return [
            rng.uniform(0.0, config.fragment_jitter_window_s)
            for _ in range(count)
        ]
    if config.mode == INDEPENDENT_LATER_FRAGMENTS:
        # The initial contention offset alone schedules fragment zero. Each
        # later fragment receives a fresh delay after its duty-cycle wait.
        return [0.0] + [
            rng.uniform(0.0, config.fragment_jitter_window_s)
            for _ in range(max(0, count - 1))
        ]
    if config.mode == LEGACY_FIXED_WINDOW:
        return [0.0] + [
            rng.uniform(0.0, config.fragment_jitter_window_s)
            for _ in range(max(0, count - 1))
        ]
    if config.mode == LEGACY_DUTY_INTERVAL:
        return [0.0] + [
            rng.uniform(0.0, airtime * (100.0 / duty_cycle_percent))
            for airtime in airtimes_s[:-1]
        ]
    raise ValueError(f"unknown fragment timing mode: {config.mode}")


def schedule_fragment_times(
    earliest_start_s: float,
    airtimes_s: Sequence[float],
    duty_cycle_percent: float,
    config: UplinkTimingConfig,
    initial_jitter_rng: random.Random,
    fragment_jitter_rng: random.Random,
    deadline_s: float | None = None,
) -> list[FragmentTiming] | None:
    """Schedule fragments after duty-cycle constraints plus nonnegative jitter.

    ``independent_per_fragment`` draws an initial client offset and an
    additional delay for every fragment. ``independent_later_fragments`` uses
    only the initial client offset for fragment zero and draws an additional
    delay for each later fragment. Every fragment starts no earlier than its
    mandatory duty-cycle time. Historical deadline-based modes retain their old
    uniform feasible-start behavior.
    """

    if duty_cycle_percent <= 0:
        raise ValueError("duty_cycle_percent must be positive")
    if not airtimes_s:
        raise ValueError("at least one fragment airtime is required")
    if any(airtime < 0 for airtime in airtimes_s):
        raise ValueError("fragment airtimes must be nonnegative")

    jitters = _fragment_jitters(
        airtimes_s, duty_cycle_percent, config, fragment_jitter_rng
    )
    legacy_deadline_mode = deadline_s is not None and config.mode not in {
        INDEPENDENT_PER_FRAGMENT,
        INDEPENDENT_LATER_FRAGMENTS,
    }
    if legacy_deadline_mode:
        duration_s = jitters[0] + airtimes_s[-1]
        for index, airtime in enumerate(airtimes_s[:-1]):
            duration_s += airtime * (100.0 / duty_cycle_percent)
            duration_s += jitters[index + 1]
        latest_start_s = deadline_s - duration_s
        if latest_start_s < earliest_start_s:
            return None
        base_start_s = initial_jitter_rng.uniform(
            earliest_start_s, latest_start_s
        )
        initial_jitter_s = base_start_s - earliest_start_s
    else:
        initial_jitter_s = initial_jitter_rng.uniform(
            0.0, config.initial_jitter_window_s
        )
        base_start_s = earliest_start_s + initial_jitter_s

    timings: list[FragmentTiming] = []
    previous_start_s: float | None = None
    previous_airtime_s: float | None = None
    for fragment_index, (airtime_s, fragment_jitter_s) in enumerate(
        zip(airtimes_s, jitters)
    ):
        if previous_start_s is None:
            earliest_allowed_s = base_start_s
        else:
            earliest_allowed_s = (
                previous_start_s
                + previous_airtime_s * (100.0 / duty_cycle_percent)
            )
        start_s = earliest_allowed_s + fragment_jitter_s
        end_s = start_s + airtime_s
        if start_s + 1e-12 < earliest_allowed_s:
            raise AssertionError("fragment jitter scheduled before earliest time")
        timings.append(FragmentTiming(
            fragment_index=fragment_index,
            earliest_allowed_s=earliest_allowed_s,
            start_s=start_s,
            end_s=end_s,
            initial_jitter_s=initial_jitter_s if fragment_index == 0 else 0.0,
            fragment_jitter_s=fragment_jitter_s,
        ))
        previous_start_s = start_s
        previous_airtime_s = airtime_s
    if deadline_s is not None and timings[-1].end_s > deadline_s + 1e-9:
        return None
    return timings
