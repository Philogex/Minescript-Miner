import csv
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
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

minescript = sys.modules.setdefault(
    "minescript",
    types.SimpleNamespace(
        script_loop=nullcontext(),
        player_set_orientation=lambda _yaw, _pitch: None,
    ),
)
minescript.script_loop = getattr(minescript, "script_loop", nullcontext())
minescript.player_set_orientation = getattr(
    minescript,
    "player_set_orientation",
    lambda _yaw, _pitch: None,
)

from minescript_miner import aim
from minescript_miner.adapter.native_bridge import AimPoint, TargetMetrics
from minescript_miner.minescript import io


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
                        "minimum_jerk[",
                        "fitts_a_ms: 10",
                        "fitts_b_ms: 20",
                        "min_duration_ms: 30",
                        "max_duration_ms: 300",
                        "sample_hz: 60",
                        "correction_probability: 0.25",
                        "max_corrections: 2",
                        "]",
                        "sigmadrift[",
                        "target_width: 12",
                        "overshoot_prob: 0.2",
                        "sample_dt_mean: 6.5",
                        "]",
                        "geometry_feedback_sigmadrift[",
                        "feedback_latency_mean_ms: 75",
                        "feedback_latency_stddev_ms: 10",
                        "feedback_latency_min_ms: 40",
                        "feedback_latency_max_ms: 120",
                        "undershoot_width_min: 0.1",
                        "undershoot_width_max: 0.3",
                        "overshoot_width_min: 0.2",
                        "overshoot_width_max: 0.4",
                        "feedback_position_uncertainty_steps: 0.4",
                        "safe_margin_steps: 1.5",
                        "max_corrections: 4",
                        "]",
                    ]
                ),
                encoding="utf-8",
            )

            config = aim.load_aim_config(config_path)

        self.assertEqual("minimum_jerk", config.aim_model)
        self.assertEqual(0.2, config.fallback_angular_step_deg)
        self.assertEqual(10.0, config.minimum_jerk.fitts_a_ms)
        self.assertEqual(20.0, config.minimum_jerk.fitts_b_ms)
        self.assertEqual(30.0, config.minimum_jerk.min_duration_ms)
        self.assertEqual(300.0, config.minimum_jerk.max_duration_ms)
        self.assertEqual(60, config.minimum_jerk.sample_hz)
        self.assertEqual(0.25, config.minimum_jerk.correction_probability)
        self.assertEqual(2, config.minimum_jerk.max_corrections)
        self.assertEqual(12.0, config.sigmadrift.target_width)
        self.assertEqual(0.2, config.sigmadrift.overshoot_prob)
        self.assertEqual(6.5, config.sigmadrift.sample_dt_mean)
        self.assertEqual(
            75.0,
            config.geometry_feedback_sigmadrift.feedback_latency_mean_ms,
        )
        self.assertEqual(
            10.0,
            config.geometry_feedback_sigmadrift.feedback_latency_stddev_ms,
        )
        self.assertEqual(
            (40.0, 120.0),
            (
                config.geometry_feedback_sigmadrift.feedback_latency_min_ms,
                config.geometry_feedback_sigmadrift.feedback_latency_max_ms,
            ),
        )
        self.assertEqual(
            (0.1, 0.3, 0.2, 0.4),
            (
                config.geometry_feedback_sigmadrift.undershoot_width_min,
                config.geometry_feedback_sigmadrift.undershoot_width_max,
                config.geometry_feedback_sigmadrift.overshoot_width_min,
                config.geometry_feedback_sigmadrift.overshoot_width_max,
            ),
        )
        self.assertEqual(
            1.5,
            config.geometry_feedback_sigmadrift.safe_margin_steps,
        )
        self.assertEqual(
            0.4,
            config.geometry_feedback_sigmadrift.
            feedback_position_uncertainty_steps,
        )
        self.assertEqual(
            4,
            config.geometry_feedback_sigmadrift.max_corrections,
        )

    def test_load_aim_config_accepts_legacy_top_level_minimum_jerk_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aim_config.txt"
            config_path.write_text(
                "\n".join(
                    [
                        "fitts_a_ms: 10",
                        "fitts_b_ms: 20",
                        "sample_hz: 60",
                    ]
                ),
                encoding="utf-8",
            )

            config = aim.load_aim_config(config_path)

        self.assertEqual(10.0, config.minimum_jerk.fitts_a_ms)
        self.assertEqual(20.0, config.minimum_jerk.fitts_b_ms)
        self.assertEqual(60, config.minimum_jerk.sample_hz)

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
                minimum_jerk=aim.MinimumJerkConfig(
                    fitts_a_ms=50.0,
                    fitts_b_ms=100.0,
                    min_duration_ms=25.0,
                    max_duration_ms=500.0,
                    sample_hz=120,
                ),
            ),
            angular_step_deg=0.15,
        )

        self.assertEqual(2, len(path))
        self.assertIsInstance(path[0], AimPoint)
        self.assertEqual((0.0, 0.0, 0.0), (path[0].yaw, path[0].pitch, path[0].t_ms))
        self.assertEqual(10.0, path[-1].yaw)
        self.assertEqual(-5.0, path[-1].pitch)
        self.assertGreater(path[-1].t_ms, 0.0)

    def test_minimum_jerk_prefers_effective_target_width(self):
        config = aim.AimConfig(
            fallback_angular_step_deg=0.15,
            minimum_jerk=aim.MinimumJerkConfig(
                fitts_a_ms=0.0,
                fitts_b_ms=100.0,
                min_duration_ms=0.0,
                max_duration_ms=1000.0,
                sample_hz=120,
            ),
        )
        local_only = TargetMetrics(
            yaw=10.0,
            pitch=0.0,
            width_yaw=1.0,
            width_pitch=1.0,
            distance=4.0,
        )
        full_region = TargetMetrics(
            yaw=10.0,
            pitch=0.0,
            width_yaw=1.0,
            width_pitch=1.0,
            distance=4.0,
            effective_width=4.0,
        )

        local_path = aim.generate_aim_path(
            (0.0, 0.0),
            local_only,
            config,
            angular_step_deg=0.15,
        )
        full_region_path = aim.generate_aim_path(
            (0.0, 0.0),
            full_region,
            config,
            angular_step_deg=0.15,
        )

        self.assertLess(full_region_path[-1].t_ms, local_path[-1].t_ms)

    def test_generate_aim_path_dispatches_to_sigmadrift_native_generator(self):
        request = (
            (0.0, 0.0),
            TargetMetrics(
                yaw=12.0,
                pitch=-4.0,
                width_yaw=2.0,
                width_pitch=1.0,
                distance=4.0,
            ),
            aim.AimConfig(aim_model="sigmadrift"),
        )
        path = aim.generate_aim_path(
            *request,
            angular_step_deg=0.15,
            seed=12345,
        )
        repeated = aim.generate_aim_path(
            *request,
            angular_step_deg=0.15,
            seed=12345,
        )

        self.assertGreater(len(path), 2)
        self.assertEqual(path, repeated)
        self.assertIsInstance(path[0], AimPoint)
        self.assertEqual((0.0, 0.0, 0.0), (path[0].yaw, path[0].pitch, path[0].t_ms))
        self.assertEqual(12.0, path[-1].yaw)
        self.assertEqual(-4.0, path[-1].pitch)
        self.assertGreater(path[-1].t_ms, 0.0)

    def test_generate_aim_path_dispatches_to_geometry_feedback_sigmadrift(self):
        target = TargetMetrics(
            yaw=0.0,
            pitch=0.0,
            width_yaw=2.0,
            width_pitch=2.0,
            distance=4.0,
            effective_width=2.0,
            visible_components=((
                (-0.25, -0.25, 1.0),
                (0.25, -0.25, 1.0),
                (0.25, 0.25, 1.0),
                (-0.25, 0.25, 1.0),
            ),),
        )

        path = aim.generate_aim_path(
            (10.0, -2.0),
            target,
            aim.AimConfig(aim_model="geometry_feedback_sigmadrift"),
            angular_step_deg=0.15,
            seed=12345,
        )

        repeated = aim.generate_aim_path(
            (10.0, -2.0),
            target,
            aim.AimConfig(aim_model="geometry_feedback_sigmadrift"),
            angular_step_deg=0.15,
            seed=12345,
        )

        self.assertGreater(len(path), 2)
        self.assertEqual(path, repeated)
        self.assertEqual((10.0, -2.0, 0.0), (
            path[0].yaw,
            path[0].pitch,
            path[0].t_ms,
        ))
        self.assertGreater(path[-1].t_ms, 0.0)
        self.assertTrue(all(
            current.t_ms > previous.t_ms
            for previous, current in zip(path, path[1:])
        ))

        with self.assertRaisesRegex(ValueError, "visible_components"):
            aim.generate_aim_path(
                (10.0, -2.0),
                TargetMetrics(
                    yaw=0.0,
                    pitch=0.0,
                    width_yaw=2.0,
                    width_pitch=2.0,
                    distance=4.0,
                ),
                aim.AimConfig(aim_model="geometry_feedback_sigmadrift"),
                angular_step_deg=0.15,
                seed=12345,
            )

    def test_geometry_feedback_diagnostics_preserve_generated_path(self):
        target = TargetMetrics(
            yaw=0.0,
            pitch=0.0,
            width_yaw=2.0,
            width_pitch=2.0,
            distance=4.0,
            effective_width=2.0,
            visible_components=((
                (-0.25, -0.25, 1.0),
                (0.25, -0.25, 1.0),
                (0.25, 0.25, 1.0),
                (-0.25, 0.25, 1.0),
            ),),
        )
        config = aim.AimConfig(aim_model="geometry_feedback_sigmadrift")

        plain_path = aim.generate_aim_path(
            (10.0, -2.0),
            target,
            config,
            angular_step_deg=0.15,
            seed=12345,
        )
        generated = aim.generate_aim_path_with_diagnostics(
            (10.0, -2.0),
            target,
            config,
            angular_step_deg=0.15,
            seed=12345,
        )

        self.assertEqual(plain_path, generated.points)
        self.assertIsNotNone(generated.diagnostics)
        diagnostics = generated.diagnostics
        assert diagnostics is not None
        self.assertGreaterEqual(diagnostics.feedback_check_count, 1)
        self.assertGreaterEqual(diagnostics.correction_count, 0)
        self.assertLessEqual(
            diagnostics.correction_count,
            diagnostics.feedback_check_count,
        )
        self.assertGreaterEqual(diagnostics.applied_margin_steps, 0.0)
        self.assertGreater(diagnostics.directional_width_steps, 0.0)
        self.assertLess(diagnostics.s_enter_steps, diagnostics.s_anchor_steps)
        self.assertLess(diagnostics.s_anchor_steps, diagnostics.s_exit_steps)
        self.assertGreaterEqual(diagnostics.first_feedback_observation_ms, 0.0)
        self.assertGreaterEqual(diagnostics.first_feedback_latency_ms, 0.0)
        self.assertAlmostEqual(
            diagnostics.first_feedback_observation_ms
            + diagnostics.first_feedback_latency_ms,
            diagnostics.first_feedback_application_ms,
        )
        self.assertGreaterEqual(
            diagnostics.first_prediction_sigma_major_steps,
            diagnostics.first_prediction_sigma_minor_steps,
        )
        self.assertGreaterEqual(diagnostics.unsafe_prediction_count, 0)
        self.assertGreaterEqual(diagnostics.visible_entry_count, 1)
        self.assertTrue(diagnostics.final_visible)

    def test_geometry_feedback_scales_endpoint_error_by_directional_width(self):
        target = TargetMetrics(
            yaw=0.0,
            pitch=0.0,
            width_yaw=2.0,
            width_pitch=2.0,
            distance=4.0,
            effective_width=2.0,
            visible_components=((
                (-0.25, -0.25, 1.0),
                (0.25, -0.25, 1.0),
                (0.25, 0.25, 1.0),
                (-0.25, 0.25, 1.0),
            ),),
        )
        config = aim.AimConfig(
            aim_model="geometry_feedback_sigmadrift",
            sigmadrift=aim.SigmaDriftConfig(overshoot_prob=0.0),
            geometry_feedback_sigmadrift=aim.GeometryFeedbackSigmaDriftConfig(
                feedback_latency_mean_ms=75.0,
                feedback_latency_stddev_ms=0.0,
                feedback_latency_min_ms=75.0,
                feedback_latency_max_ms=75.0,
                undershoot_width_min=0.2,
                undershoot_width_max=0.2,
            ),
        )

        generated = aim.generate_aim_path_with_diagnostics(
            (10.0, -2.0),
            target,
            config,
            angular_step_deg=0.15,
            seed=12345,
        )
        diagnostics = generated.diagnostics
        assert diagnostics is not None

        normalized_error = (
            diagnostics.s_anchor_steps - diagnostics.primary_endpoint_steps
        ) / diagnostics.directional_width_steps
        self.assertAlmostEqual(0.2, normalized_error, places=12)
        self.assertEqual(75.0, diagnostics.first_feedback_latency_ms)
        self.assertGreater(diagnostics.first_feedback_observation_ms, 0.0)

    def test_generate_aim_path_exports_synthetic_daq_session_on_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = aim.generate_aim_path(
                (0.0, 0.0),
                TargetMetrics(
                    yaw=10.0,
                    pitch=-5.0,
                    width_yaw=2.0,
                    width_pitch=1.0,
                    distance=4.0,
                    target_block=(4, 70, -3),
                    face_id="north",
                    hit_point=(4.5, 70.2, -3.0),
                    block_state_before="minecraft:diamond_ore",
                    neighbors=((0, 1, 0, "minecraft:air"),),
                ),
                aim.AimConfig(),
                angular_step_deg=0.15,
                synthetic_export_root=Path(temp_dir),
            )
            sessions = list(Path(temp_dir).glob("synthetic-*"))

            self.assertEqual(1, len(sessions))
            session = sessions[0]
            metadata = json.loads((session / "metadata.json").read_text())
            self.assertEqual("minescript-miner-synthetic", metadata["source"])
            self.assertEqual("minimum_jerk", metadata["generator"])
            self.assertTrue((session / "events.csv").is_file())
            self.assertTrue((session / "state_samples.csv").is_file())
            self.assertTrue((session / "mouse_trajectory.csv").is_file())
            with (session / "events.csv").open(newline="") as file:
                event = next(csv.DictReader(file))
            self.assertEqual("4", event["target_x"])
            self.assertEqual("70", event["target_y"])
            self.assertEqual("-3", event["target_z"])
            self.assertEqual("north", event["face_id"])
            self.assertEqual("minecraft:diamond_ore", event["block_state_before"])
            self.assertEqual(
                [{"dx": 0, "dy": 1, "dz": 0, "state": "minecraft:air"}],
                json.loads(event["neighbors_json"]),
            )
            self.assertEqual(2, len(path))

    def test_execute_aim_path_applies_samples_with_relative_delays(self):
        applied = []
        delays = []

        original_set_orientation = io.set_orientation
        try:
            io.set_orientation = lambda yaw, pitch: applied.append((yaw, pitch))
            completed = aim.execute_aim_path(
                (
                    AimPoint(1.0, 2.0, 0.0),
                    AimPoint(3.0, 4.0, 25.0),
                    AimPoint(5.0, 6.0, 40.0),
                ),
                sleep=delays.append,
            )
        finally:
            io.set_orientation = original_set_orientation

        self.assertTrue(completed)
        self.assertEqual([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], applied)
        self.assertEqual([0.025, 0.015], delays)

    def test_execute_aim_path_repeats_final_sample_after_settle_delay(self):
        applied = []
        delays = []

        original_set_orientation = io.set_orientation
        try:
            io.set_orientation = lambda yaw, pitch: applied.append((yaw, pitch))
            completed = aim.execute_aim_path(
                (
                    AimPoint(1.0, 2.0, 0.0),
                    AimPoint(3.0, 4.0, 10.0),
                ),
                sleep=delays.append,
                settle_delay_s=0.05,
            )
        finally:
            io.set_orientation = original_set_orientation

        self.assertTrue(completed)
        self.assertEqual([(1.0, 2.0), (3.0, 4.0), (3.0, 4.0)], applied)
        self.assertEqual([0.01, 0.05], delays)


if __name__ == "__main__":
    unittest.main()
