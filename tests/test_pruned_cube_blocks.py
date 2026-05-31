import sys
import types
import unittest


sys.modules.setdefault(
    "minescript",
    types.SimpleNamespace(await_loaded_region=lambda *_args: None),
)

from minescript_miner.minescript import world


class PrunedCubeBlocksTest(unittest.TestCase):
    def test_get_area_preserves_cube_shape_and_fills_air(self):
        calls = []
        original_positions_within_reach = world.positions_within_reach
        original_read_blocks_region_prune = world.read_blocks_region_prune
        original_await_loaded_region = world.m.await_loaded_region

        def fake_await_loaded_region(*args):
            calls.append(args)

        def fake_positions_within_reach(position, reach, pitch_range):
            return [[9, 63, -5], [10, 64, -4]]

        def fake_read_blocks_region_prune(positions):
            return ["minecraft:stone", None]

        try:
            world.m.await_loaded_region = fake_await_loaded_region
            world.positions_within_reach = fake_positions_within_reach
            world.read_blocks_region_prune = fake_read_blocks_region_prune

            area = world.get_area(
                (10.1, 64.2, -3.7),
                reach=1.0,
            )
        finally:
            world.positions_within_reach = original_positions_within_reach
            world.read_blocks_region_prune = original_read_blocks_region_prune
            world.m.await_loaded_region = original_await_loaded_region

        min_pos, max_pos = world.fixed_cube_bounds((10.1, 64.2, -3.7), 1.0)
        block_strings = [block_string for _pos, block_string in area]

        self.assertEqual((9, 63, -5), min_pos)
        self.assertEqual((11, 65, -3), max_pos)
        self.assertEqual(27, len(area))
        self.assertEqual([(9, -5, 11, -3)], calls)
        self.assertEqual((9, 63, -5), area[0][0])
        self.assertEqual("minecraft:stone", block_strings[world.cube_block_index((9, 63, -5), min_pos, 3)])
        self.assertEqual("minecraft:air", block_strings[world.cube_block_index((10, 64, -4), min_pos, 3)])
        self.assertEqual(26, block_strings.count("minecraft:air"))


if __name__ == "__main__":
    unittest.main()
