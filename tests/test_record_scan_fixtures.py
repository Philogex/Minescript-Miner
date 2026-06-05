import importlib.util
import sys
import types
import unittest
from pathlib import Path


class RecordScanFixturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project_root / "src"))
        sys.modules.setdefault(
            "minescript",
            types.SimpleNamespace(
                player_position=lambda: (0.0, 0.0, 0.0),
                player_orientation=lambda: (0.0, 0.0),
                echo=lambda _message: None,
            ),
        )
        spec = importlib.util.spec_from_file_location(
            "record_scan_fixtures",
            project_root / "record_scan_fixtures.py",
        )
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_fixture_lines_preserve_sparse_blocks_and_targets(self):
        shape_ids = [0] * 125
        shape_ids[1] = 1
        shape_ids[7] = 44

        lines = self.module.fixture_lines(
            side=5,
            position=(10.25, 65.62, -3.75),
            orientation=(-98.1, -13.2),
            shape_ids=shape_ids,
            target_indices=[7],
        )

        self.assertIn("side 5", lines)
        self.assertIn("position 10.25 65.62 -3.75", lines)
        self.assertIn("orientation_yaw_pitch -98.1 -13.2", lines)
        self.assertIn("block 1 1", lines)
        self.assertIn("target 7 44", lines)
        self.assertNotIn("block 7 44", lines)
        self.assertFalse(any(line.startswith("block 0 ") for line in lines))

    def test_reach_matches_requested_odd_side(self):
        self.assertEqual(2, self.module.reach_for_side(5))
        self.assertEqual(19, self.module.reach_for_side(39))
        with self.assertRaises(ValueError):
            self.module.reach_for_side(4)


if __name__ == "__main__":
    unittest.main()
