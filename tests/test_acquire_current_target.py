import sys
import types
import unittest


sys.modules.setdefault(
    "minescript",
    types.SimpleNamespace(await_loaded_region=lambda *_args: None),
)

from minescript_miner.adapter.shape_catalog import SHAPE_CATALOG_VERSION, SHAPE_ID_BY_NAME
from minescript_miner.minescript import io


class AcquireCurrentTargetTest(unittest.TestCase):
    def test_acquire_current_target_encodes_blocks_before_native_bridge(self):
        captured = {}
        original_get_area = io.get_area
        original_acquire_target = io.acquire_target

        def fake_get_area(position, reach):
            block_strings = [
                "minecraft:air",
                "minecraft:oak_slab[type=top,waterlogged=false]",
                "minecraft:unknown_block",
                None,
            ]
            block_strings.extend(["minecraft:air"] * 23)
            return [
                ((index, 0, 0), block_string)
                for index, block_string in enumerate(block_strings)
            ]

        def fake_acquire_target(position, orientation, shape_catalog_version, side, shape_ids):
            captured["position"] = position
            captured["orientation"] = orientation
            captured["shape_catalog_version"] = shape_catalog_version
            captured["side"] = side
            captured["shape_ids"] = list(shape_ids)
            return 0.25, -0.5

        try:
            io.get_area = fake_get_area
            io.acquire_target = fake_acquire_target

            result = io.acquire_current_target((0.5, 0.5, 0.5), (90.0, 10.0), reach=0.5)
        finally:
            io.get_area = original_get_area
            io.acquire_target = original_acquire_target

        self.assertEqual((0.25, -0.5), result)
        self.assertEqual((0.5, 0.5, 0.5), captured["position"])
        self.assertEqual((90.0, 10.0), captured["orientation"])
        self.assertEqual(SHAPE_CATALOG_VERSION, captured["shape_catalog_version"])
        self.assertEqual(3, captured["side"])
        self.assertEqual(27, len(captured["shape_ids"]))
        self.assertEqual(SHAPE_ID_BY_NAME["empty"], captured["shape_ids"][0])
        self.assertEqual(SHAPE_ID_BY_NAME["slab_top"], captured["shape_ids"][1])
        self.assertEqual(SHAPE_ID_BY_NAME["full_cube"], captured["shape_ids"][2])
        self.assertEqual(SHAPE_ID_BY_NAME["empty"], captured["shape_ids"][3])


if __name__ == "__main__":
    unittest.main()
