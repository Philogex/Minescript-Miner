import sys
import tempfile
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

from minescript_miner import aim
from minescript_miner.adapter.native_bridge import AimPoint, TargetMetrics


class AimConfigTest(unittest.TestCase):
    def test_load_aim_config_reads_name_value_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aim_config.txt"
            config_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "aim_model: minimum_jerk",
                        "fallback_angular_step_deg: 0.2",
                        "fitts_a_ms: 10",
                        "fitts_b_ms: 20",
                        "min_duration_ms: 30",
                        "max_duration_ms: 300",
                        "sample_hz: 60",
                        "correction_probability: 0.25",
                        "max_corrections: 2",
                    ]
                ),
                encoding="utf-8",
            )

            config = aim.load_aim_config(config_path)

        self.assertEqual("minimum_jerk", config.aim_model)
        self.assertEqual(0.2, config.fallback_angular_step_deg)
        self.assertEqual(10.0, config.fitts_a_ms)
        self.assertEqual(20.0, config.fitts_b_ms)
        self.assertEqual(30.0, config.min_duration_ms)
        self.assertEqual(300.0, config.max_duration_ms)
        self.assertEqual(60, config.sample_hz)
        self.assertEqual(0.25, config.correction_probability)
        self.assertEqual(2, config.max_corrections)

    def test_load_aim_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aim_config.txt"
            config_path.write_text("unknown: value\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown aim config key"):
                aim.load_aim_config(config_path)

    def test_sensitivity_to_angular_step_matches_minecraft_100_percent(self):
        self.assertAlmostEqual(0.15, aim.sensitivity_to_angular_step_deg(0.5))

    def test_generate_aim_path_dispatches_to_minimum_jerk_native_stub(self):
        path = aim.generate_aim_path(
            (0.0, 0.0),
            TargetMetrics(
                yaw=10.0,
                pitch=-5.0,
                width_yaw=2.0,
                width_pitch=1.0,
                distance=4.0,
            ),
            aim.AimConfig(
                fallback_angular_step_deg=0.15,
                fitts_a_ms=50.0,
                fitts_b_ms=100.0,
                min_duration_ms=25.0,
                max_duration_ms=500.0,
                sample_hz=120,
            ),
            angular_step_deg=0.15,
        )

        self.assertEqual(2, len(path))
        self.assertIsInstance(path[0], AimPoint)
        self.assertEqual((0.0, 0.0, 0.0), (path[0].yaw, path[0].pitch, path[0].t_ms))
        self.assertEqual(10.0, path[-1].yaw)
        self.assertEqual(-5.0, path[-1].pitch)
        self.assertGreater(path[-1].t_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
