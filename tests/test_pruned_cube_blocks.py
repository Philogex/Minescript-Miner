import sys
import types
import unittest
from contextlib import nullcontext


minescript = sys.modules.setdefault(
    "minescript",
    types.SimpleNamespace(
        await_loaded_region=lambda *_args: None,
        script_loop=nullcontext(),
    ),
)
minescript.await_loaded_region = getattr(
    minescript,
    "await_loaded_region",
    lambda *_args: None,
)
minescript.script_loop = getattr(minescript, "script_loop", nullcontext())

from minescript_miner.minescript import io, world


class PrunedCubeBlocksTest(unittest.TestCase):
    def test_read_block_region_uses_script_loop(self):
        calls = []
        original_get_block_region = getattr(io.m, "get_block_region", None)
        original_script_loop = getattr(io.m, "script_loop", None)

        class ScriptLoop:
            def __enter__(self):
                calls.append("enter")

            def __exit__(self, _exc_type, _exc_value, _traceback):
                calls.append("exit")

        try:
            io.m.script_loop = ScriptLoop()
            io.m.get_block_region = lambda pos1, pos2: calls.append(
                (pos1, pos2)
            ) or "region"

            region = world.read_block_region((1, 2, 3), (4, 5, 6))
        finally:
            if original_get_block_region is None:
                del io.m.get_block_region
            else:
                io.m.get_block_region = original_get_block_region
            if original_script_loop is None:
                del io.m.script_loop
            else:
                io.m.script_loop = original_script_loop

        self.assertEqual("region", region)
        self.assertEqual(
            ["enter", ((1, 2, 3), (4, 5, 6)), "exit"],
            calls,
        )

    def test_get_area_preserves_cube_shape_and_fills_air(self):
        calls = []
        original_positions_within_reach = world.positions_within_reach
        original_read_blocks_region_prune = world.read_blocks_region_prune
        original_await_loaded_region = io.await_loaded_region

        def fake_await_loaded_region(*args):
            calls.append(args)

        def fake_positions_within_reach(position, reach, pitch_range):
            return [[9, 63, -5], [10, 64, -4]]

        def fake_read_blocks_region_prune(positions):
            return ["minecraft:stone", None]

        try:
            io.await_loaded_region = fake_await_loaded_region
            world.positions_within_reach = fake_positions_within_reach
            world.read_blocks_region_prune = fake_read_blocks_region_prune

            area = world.get_area(
                (10.1, 64.2, -3.7),
                reach=1.0,
            )
        finally:
            world.positions_within_reach = original_positions_within_reach
            world.read_blocks_region_prune = original_read_blocks_region_prune
            io.await_loaded_region = original_await_loaded_region

        min_pos, max_pos = world.fixed_cube_bounds((10.1, 64.2, -3.7), 1.0)
        block_strings = [block_string for _pos, block_string in area]

        self.assertEqual((9, 63, -5), min_pos)
        self.assertEqual((11, 65, -3), max_pos)
        self.assertEqual(27, len(area))
        self.assertEqual([(9, -5, 11, -3)], calls)
        self.assertEqual((9, 63, -5), area[0][0])
        self.assertEqual(
            "minecraft:stone",
            block_strings[world.cube_block_index((9, 63, -5), min_pos, 3)],
        )
        self.assertEqual(
            "minecraft:air",
            block_strings[world.cube_block_index((10, 64, -4), min_pos, 3)],
        )
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


if __name__ == "__main__":
    unittest.main()
