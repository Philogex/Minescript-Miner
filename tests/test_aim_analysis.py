import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

loaded_package = sys.modules.get("minescript_miner")
if loaded_package is not None:
    package_path = getattr(loaded_package, "__file__", "")
    if package_path and not package_path.startswith(src_path):
        for module_name in list(sys.modules):
            if module_name == "minescript_miner" or module_name.startswith(
                "minescript_miner."
            ):
                del sys.modules[module_name]

from minescript_miner.adapter.native_bridge import AimPoint, TargetMetrics
from minescript_miner.aim_analysis import compute_aim_path_features, unwrap_yaws


class AimAnalysisTest(unittest.TestCase):
    def test_compute_aim_path_features_reports_fitts_and_geometry(self):
        path = (
            AimPoint(0.0, 0.0, 0.0),
            AimPoint(5.0, 1.0, 50.0),
            AimPoint(10.0, 0.0, 100.0),
        )
        target = TargetMetrics(
            yaw=10.0,
            pitch=0.0,
            width_yaw=2.0,
            width_pitch=2.0,
            distance=4.0,
        )

        features = compute_aim_path_features(
            path,
            target,
            fitts_a_ms=50.0,
            fitts_b_ms=100.0,
            fallback_width_deg=0.15,
        )

        self.assertEqual(100.0, features.fitts_mt)
        self.assertAlmostEqual(math.log2(10.0 / 2.0 + 1.0), features.fitts_id)
        self.assertLess(features.fitts_residual, 0.0)
        self.assertEqual(1, features.sub_peak_count)
        self.assertLess(features.geo_path_efficiency, 1.0)
        self.assertAlmostEqual(1.0, features.geo_max_deviation)
        self.assertGreaterEqual(features.geo_curvature_integral, 0.0)

    def test_compute_aim_path_features_handles_empty_path(self):
        features = compute_aim_path_features(
            (),
            TargetMetrics(
                yaw=0.0,
                pitch=0.0,
                width_yaw=1.0,
                width_pitch=1.0,
                distance=1.0,
            ),
            fitts_a_ms=50.0,
            fitts_b_ms=100.0,
            fallback_width_deg=0.15,
        )

        self.assertEqual(0.0, features.fitts_mt)
        self.assertEqual(0, features.sub_peak_count)
        self.assertTrue(math.isnan(features.geo_path_efficiency))

    def test_screen_coordinate_mode_does_not_wrap_x_axis(self):
        path = (
            AimPoint(350.0, 0.0, 0.0),
            AimPoint(10.0, 0.0, 10.0),
        )

        self.assertEqual([350.0, 370.0], unwrap_yaws(path))
        self.assertEqual([350.0, 10.0], unwrap_yaws(path, wrap_yaw=False))


if __name__ == "__main__":
    unittest.main()
