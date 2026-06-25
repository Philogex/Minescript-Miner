import math
import sys
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
        player_position=lambda: (0.0, 0.0, 0.0),
    ),
)
minescript.script_loop = getattr(minescript, "script_loop", nullcontext())
minescript.player_position = getattr(
    minescript,
    "player_position",
    lambda: (0.0, 0.0, 0.0),
)

from minescript_miner import queries
from minescript_miner.adapter.catalog_contract import SHAPE_CATALOG_VERSION
from minescript_miner.adapter.shape_catalog import SHAPE_ID_BY_NAME


class QueryApiTest(unittest.TestCase):
    def setUp(self):
        self.original_read_block_region = queries.io.read_block_region
        self.original_acquire_target = queries.acquire_target
        self.original_player_position = queries.io.player_position

    def tearDown(self):
        queries.io.read_block_region = self.original_read_block_region
        queries.acquire_target = self.original_acquire_target
        queries.io.player_position = self.original_player_position

    def test_can_see_block_targets_single_block_index(self):
        captured = {}

        class Region:
            blocks = ["minecraft:air"] * 27

        def fake_read_block_region(pos1, pos2):
            captured["bounds"] = (pos1, pos2)
            return Region()

        def fake_acquire_target(
            position,
            orientation,
            shape_catalog_version,
            side,
            reach,
            shape_ids,
            target_indices,
        ):
            captured["position"] = position
            captured["orientation"] = orientation
            captured["shape_catalog_version"] = shape_catalog_version
            captured["side"] = side
            captured["reach"] = reach
            captured["shape_ids"] = list(shape_ids)
            captured["target_indices"] = list(target_indices)
            return 12.0, -3.0

        queries.io.read_block_region = fake_read_block_region
        queries.acquire_target = fake_acquire_target

        self.assertTrue(queries.can_see_block((0.5, 64.5, 0.5), (1, 64, 0)))
        self.assertEqual(((-1, 63, -1), (1, 65, 1)), captured["bounds"])
        self.assertEqual((0.5, 64.5, 0.5), captured["position"])
        self.assertEqual((-90.0, -0.0), captured["orientation"])
        self.assertEqual(SHAPE_CATALOG_VERSION, captured["shape_catalog_version"])
        self.assertEqual(3, captured["side"])
        self.assertEqual([14], captured["target_indices"])
        self.assertEqual(27, len(captured["shape_ids"]))
        self.assertEqual(SHAPE_ID_BY_NAME["empty"], captured["shape_ids"][0])
        self.assertAlmostEqual(math.sqrt(2.75), captured["reach"])

    def test_get_angle_to_block_uses_player_eye_position(self):
        captured = {}

        class Region:
            blocks = ["minecraft:air"]

        queries.io.player_position = lambda: (3.25, 10.0, -2.25)

        def fake_read_block_region(pos1, pos2):
            captured["bounds"] = (pos1, pos2)
            return Region()

        queries.io.read_block_region = fake_read_block_region

        def fake_acquire_target(
            position,
            _orientation,
            _shape_catalog_version,
            _side,
            _reach,
            _shape_ids,
            _target_indices,
        ):
            captured["position"] = position
            return 1.0, 2.0

        queries.acquire_target = fake_acquire_target

        self.assertEqual((1.0, 2.0), queries.get_angle_to_block((3, 11, -3)))
        self.assertEqual(3.25, captured["position"][0])
        self.assertAlmostEqual(11.62, captured["position"][1])
        self.assertEqual(-2.25, captured["position"][2])

    def test_can_see_block_rejects_blocks_outside_scan_cube_limit(self):
        with self.assertRaisesRegex(ValueError, "side=.*side must be <= 39"):
            queries.can_see_block((0.5, 0.5, 0.5), (20, 0, 0))


if __name__ == "__main__":
    unittest.main()
