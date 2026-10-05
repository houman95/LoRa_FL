import random
import unittest

from lorawan_fl.access_policy import DutyCycleTracker, _schedule_uplink_update
from lorawan_fl.channel import AlohaChannel


class UplinkChannelCountTests(unittest.TestCase):
    def test_one_channel_assigns_every_fragment_to_channel_zero(self):
        random.seed(7)
        txs, reason = _schedule_uplink_update(
            cid=0,
            blob=b"x" * 700,
            collection_start=0.0,
            deadline_s=None,
            dev_dc=[DutyCycleTracker()],
            spreading_factor=7,
            contention_jitter_s=0.0,
            num_uplink_channels=1,
        )
        self.assertIsNone(reason)
        self.assertGreater(len(txs), 1)
        self.assertEqual({tx.channel_idx for tx in txs}, {0})

    def test_channel_rejects_nonpositive_count(self):
        with self.assertRaises(ValueError):
            AlohaChannel(0)

    def test_initial_only_preserves_deterministic_fragment_spacing(self):
        txs, reason = _schedule_uplink_update(
            cid=0,
            blob=b"x" * 700,
            collection_start=0.0,
            deadline_s=None,
            dev_dc=[DutyCycleTracker(dc_percent=1.0)],
            spreading_factor=7,
            contention_jitter_s=0.0,
            num_uplink_channels=1,
            fragment_jitter_mode="initial_only",
        )
        self.assertIsNone(reason)
        for previous, current in zip(txs, txs[1:]):
            expected_d = previous.airtime_s * 100.0
            self.assertAlmostEqual(current.start_s - previous.start_s, expected_d)
            self.assertEqual(current.preceding_fragment_jitter_s, 0.0)

    def test_per_fragment_mode_draws_independent_jitter_in_duty_window(self):
        txs, reason = _schedule_uplink_update(
            cid=0,
            blob=b"x" * 1000,
            collection_start=0.0,
            deadline_s=None,
            dev_dc=[DutyCycleTracker(dc_percent=1.0)],
            spreading_factor=7,
            contention_jitter_s=0.0,
            num_uplink_channels=1,
            fragment_jitter_mode="per_fragment_duty_interval",
            fragment_jitter_rng=random.Random(1234),
        )
        self.assertIsNone(reason)
        self.assertEqual(txs[0].preceding_fragment_jitter_s, 0.0)
        sampled = []
        for previous, current in zip(txs, txs[1:]):
            duty_interval = previous.airtime_s * 100.0
            jitter = current.preceding_fragment_jitter_s
            self.assertGreaterEqual(jitter, 0.0)
            self.assertLessEqual(jitter, duty_interval)
            self.assertAlmostEqual(
                current.start_s - previous.start_s,
                duty_interval + jitter,
            )
            sampled.append(round(jitter, 12))
        self.assertGreater(len(set(sampled)), 1)

    def test_invalid_fragment_jitter_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _schedule_uplink_update(
                cid=0,
                blob=b"x" * 700,
                collection_start=0.0,
                deadline_s=None,
                dev_dc=[DutyCycleTracker()],
                spreading_factor=7,
                contention_jitter_s=0.0,
                num_uplink_channels=1,
                fragment_jitter_mode="unknown",
            )

    def test_fixed_per_fragment_window_uses_configured_two_seconds(self):
        txs, reason = _schedule_uplink_update(
            cid=0,
            blob=b"x" * 1000,
            collection_start=0.0,
            deadline_s=None,
            dev_dc=[DutyCycleTracker(dc_percent=1.0)],
            spreading_factor=7,
            contention_jitter_s=0.0,
            num_uplink_channels=1,
            fragment_jitter_mode="per_fragment_fixed_window",
            fragment_jitter_rng=random.Random(4321),
            fragment_jitter_window_s=2.0,
        )
        self.assertIsNone(reason)
        self.assertEqual(txs[0].preceding_fragment_jitter_s, 0.0)
        jitters = []
        for previous, current in zip(txs, txs[1:]):
            jitter = current.preceding_fragment_jitter_s
            self.assertGreaterEqual(jitter, 0.0)
            self.assertLessEqual(jitter, 2.0)
            self.assertAlmostEqual(
                current.start_s - previous.start_s,
                previous.airtime_s * 100.0 + jitter,
            )
            jitters.append(round(jitter, 12))
        self.assertGreater(len(set(jitters)), 1)

    def test_negative_fixed_jitter_window_is_rejected(self):
        with self.assertRaises(ValueError):
            _schedule_uplink_update(
                cid=0,
                blob=b"x" * 700,
                collection_start=0.0,
                deadline_s=None,
                dev_dc=[DutyCycleTracker()],
                spreading_factor=7,
                contention_jitter_s=0.0,
                num_uplink_channels=1,
                fragment_jitter_mode="per_fragment_fixed_window",
                fragment_jitter_window_s=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
