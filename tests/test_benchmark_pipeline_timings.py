import sys
import types
import unittest


sys.modules.setdefault(
    "minescript",
    types.SimpleNamespace(
        await_loaded_region=lambda *_args: None,
        player_position=lambda: (0.0, 0.0, 0.0),
        player_orientation=lambda: (0.0, 0.0),
        echo=lambda *_args: None,
    ),
)

import benchmark_pipeline_timings as benchmark
from minescript_miner.minescript.scanner import ScanTimings


class BenchmarkPipelineTimingsTest(unittest.TestCase):
    def test_collect_samples_warms_up_and_measures_without_awaiting_region(self):
        calls = []

        def fake_scan(*, await_region):
            calls.append(await_region)
            timing = ScanTimings()
            timing.area.total_ms = float(len(calls))
            timing.total_ms = float(len(calls) * 2)
            return timing

        samples = benchmark.collect_samples(
            fake_scan,
            warmup_runs=3,
            measured_runs=5,
        )

        self.assertEqual([False] * 8, calls)
        self.assertEqual(5, len(samples["total"]))
        self.assertEqual([8.0, 10.0, 12.0, 14.0, 16.0], samples["total"])
        self.assertEqual([4.0, 5.0, 6.0, 7.0, 8.0], samples["area_total"])

    def test_summary_uses_median_and_nearest_rank_p95(self):
        values = [float(value) for value in range(1, 101)]

        summary = benchmark.summarize_samples({"stage": values})

        self.assertEqual((50.5, 95.0), summary["stage"])

    def test_percentile_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            benchmark.percentile_nearest_rank([], 0.95)


if __name__ == "__main__":
    unittest.main()
