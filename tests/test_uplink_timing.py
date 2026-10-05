"""Independent jitter, duty-cycle, and collision-decorrelation checks."""

import random
import unittest

from lorawan_fl.access_policy import DutyCycleTracker, collect_adaptive_complete
from lorawan_fl.channel import AlohaChannel
from lorawan_fl.config import ClientMacState, RoundPolicyConfig, RoundPolicyType
from lorawan_fl.fragmentation import (
    fragment_reception_diagnostics,
    receptions_from_transmission_log,
)
from lorawan_fl.lora_phy import LoRa_MTU
from lorawan_fl.timing import (
    INDEPENDENT_LATER_FRAGMENTS,
    INDEPENDENT_PER_FRAGMENT,
    INITIAL_ONLY,
    UplinkRandomStreams,
    UplinkTimingConfig,
    schedule_fragment_times,
)


class UplinkTimingTests(unittest.TestCase):
    def test_later_fragment_mode_does_not_double_jitter_first_fragment(self):
        streams = UplinkRandomStreams.from_seed(19)
        config = UplinkTimingConfig(
            initial_jitter_window_s=34.0,
            fragment_jitter_window_s=2.0,
            enable_fragment_jitter=True,
            mode=INDEPENDENT_LATER_FRAGMENTS,
        )
        timings = schedule_fragment_times(
            0.0, [0.05] * 5, 1.0, config,
            streams.initial_jitter, streams.fragment_jitter,
        )
        jitters = [timing.fragment_jitter_s for timing in timings]
        self.assertEqual(jitters[0], 0.0)
        self.assertEqual(len(set(jitters[1:])), 4)
        self.assertTrue(all(0.0 <= value <= 2.0 for value in jitters[1:]))
        self.assertAlmostEqual(
            timings[0].start_s,
            timings[0].earliest_allowed_s,
        )

    def test_fresh_jitter_for_every_fragment_and_client(self):
        streams = UplinkRandomStreams.from_seed(17)
        config = UplinkTimingConfig(
            initial_jitter_window_s=34.0,
            fragment_jitter_window_s=2.0,
            enable_fragment_jitter=True,
            mode=INDEPENDENT_PER_FRAGMENT,
        )
        first = schedule_fragment_times(
            0.0, [0.05] * 5, 1.0, config,
            streams.initial_jitter, streams.fragment_jitter,
        )
        second = schedule_fragment_times(
            0.0, [0.05] * 5, 1.0, config,
            streams.initial_jitter, streams.fragment_jitter,
        )
        first_jitters = [timing.fragment_jitter_s for timing in first]
        second_jitters = [timing.fragment_jitter_s for timing in second]
        self.assertEqual(len(set(first_jitters)), 5)
        self.assertEqual(len(set(second_jitters)), 5)
        self.assertNotEqual(first_jitters, second_jitters)
        self.assertTrue(all(0.0 <= value <= 2.0 for value in first_jitters))
        self.assertGreater(first_jitters[0], 0.0)

    def test_jitter_is_additional_to_mandatory_duty_cycle_wait(self):
        config = UplinkTimingConfig(
            initial_jitter_window_s=34.0,
            fragment_jitter_window_s=2.0,
            enable_fragment_jitter=True,
            mode=INDEPENDENT_PER_FRAGMENT,
        )
        timings = schedule_fragment_times(
            earliest_start_s=7.0,
            airtimes_s=[0.1, 0.2, 0.1],
            duty_cycle_percent=1.0,
            config=config,
            initial_jitter_rng=random.Random(1),
            fragment_jitter_rng=random.Random(2),
        )
        for timing in timings:
            self.assertGreaterEqual(timing.start_s, timing.earliest_allowed_s)
        for previous, current in zip(timings, timings[1:]):
            mandatory_next = previous.start_s + previous.end_s - previous.start_s
            mandatory_next = previous.start_s + (
                (previous.end_s - previous.start_s) * 100.0
            )
            self.assertGreaterEqual(current.start_s, mandatory_next)

    def test_named_streams_are_reproducible(self):
        left = UplinkRandomStreams.from_seed(3)
        right = UplinkRandomStreams.from_seed(3)
        self.assertEqual(left.seeds, right.seeds)
        self.assertEqual(
            [left.fragment_jitter.random() for _ in range(5)],
            [right.fragment_jitter.random() for _ in range(5)],
        )

    def test_controlled_collision_decorrelation_diagnostic(self):
        def run(enable_fragment_jitter):
            num_clients = 100
            fragmentation = 32
            streams = UplinkRandomStreams.from_seed(11)
            result = collect_adaptive_complete(
                collection_start_s=0.0,
                T_end_s=1e9,
                policy=RoundPolicyConfig(
                    policy_type=RoundPolicyType.ADAPTIVE_COMPLETE,
                    window_s=120.0,
                ),
                prepared_updates={
                    client_id: bytes(fragmentation * LoRa_MTU(7))
                    for client_id in range(num_clients)
                },
                dev_dc=[DutyCycleTracker(1.0) for _ in range(num_clients)],
                client_mac=[ClientMacState() for _ in range(num_clients)],
                spreading_factor=7,
                participation_probability=0.55,
                aloha=AlohaChannel(1),
                contention_jitter_s=34.0,
                timing_config=UplinkTimingConfig(
                    initial_jitter_window_s=34.0,
                    fragment_jitter_window_s=2.0,
                    enable_fragment_jitter=enable_fragment_jitter,
                    mode=(
                        INDEPENDENT_PER_FRAGMENT
                        if enable_fragment_jitter
                        else INITIAL_ONLY
                    ),
                ),
                rng_streams=streams,
                round_index=2,
            )
            receptions = receptions_from_transmission_log(result._txs_log)
            diagnostic, _ = fragment_reception_diagnostics(
                result.selected_clients, receptions
            )
            return result, diagnostic

        old_result, old = run(False)
        new_result, new = run(True)
        self.assertEqual(old_result.attempted_clients, new_result.attempted_clients)
        for diagnostic in (old, new):
            self.assertEqual(
                diagnostic["num_clients_full_success"]
                + diagnostic["num_clients_partial_success"]
                + diagnostic["num_clients_zero_success"],
                diagnostic["selected_clients"],
            )
        self.assertNotEqual(
            (
                old["num_clients_full_success"],
                old["num_clients_partial_success"],
                old["num_clients_zero_success"],
            ),
            (
                new["num_clients_full_success"],
                new["num_clients_partial_success"],
                new["num_clients_zero_success"],
            ),
        )
        print("controlled_collision_diagnostic", {"old": old, "new": new})


if __name__ == "__main__":
    unittest.main()
