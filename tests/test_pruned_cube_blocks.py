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

    def test_positions_within_reach_use_block_aabb_distance(self):
        positions = {
            tuple(position)
            for position in world.positions_within_reach(
                (0.9, 0.5, 0.5),
                reach=1.1,
            )
        }

        self.assertIn((2, 0, 0), positions)
        self.assertNotIn((-2, 0, 0), positions)

    def test_get_area_records_pipeline_timings(self):
        original_get_block_region = getattr(world.m, "get_block_region", None)
        original_await_loaded_region = world.m.await_loaded_region

        class Region:
            min_pos = (0, 0, 0)
            max_pos = (0, 0, 0)
            x_length = 1
            z_length = 1
            blocks = ["minecraft:stone"]

        try:
            world.m.await_loaded_region = lambda *_args: None
            world.m.get_block_region = lambda *_args: Region()
            timings = world.AreaTimings()
            area = world.get_area(
                (0.5, 0.5, 0.5),
                reach=0.1,
                timings=timings,
            )
        finally:
            world.m.await_loaded_region = original_await_loaded_region
            if original_get_block_region is None:
                del world.m.get_block_region
            else:
                world.m.get_block_region = original_get_block_region

        self.assertEqual(27, len(area))
        self.assertGreaterEqual(timings.await_region_ms, 0.0)
        self.assertGreaterEqual(timings.prune_positions_ms, 0.0)
        self.assertGreaterEqual(timings.region_read_ms, 0.0)
        self.assertGreaterEqual(timings.region_extract_ms, 0.0)
        self.assertGreaterEqual(timings.cube_rebuild_ms, 0.0)
        self.assertGreaterEqual(timings.total_ms, timings.region_read_ms)


if __name__ == "__main__":
    unittest.main()
