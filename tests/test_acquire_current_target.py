import sys
import tempfile
import types
import unittest
from pathlib import Path


sys.modules.setdefault(
    "minescript",
    types.SimpleNamespace(await_loaded_region=lambda *_args: None),
)

from minescript_miner.adapter.shape_catalog import SHAPE_CATALOG_VERSION, SHAPE_ID_BY_NAME
from minescript_miner.minescript import io


class AcquireCurrentTargetTest(unittest.TestCase):
    def test_acquire_current_target_uses_preloaded_target_blocks(self):
        original_get_area = io.get_area
        original_acquire_target = io.acquire_target
        original_load_target_blocks = io.load_target_blocks

        def fake_get_area(position, reach, *, await_region=True):
            return [
                ((index, 0, 0), "minecraft:stone")
                for index in range(27)
            ]

        try:
            io.get_area = fake_get_area
            io.acquire_target = lambda *_args: (1.0, 2.0)
            io.load_target_blocks = lambda *_args: self.fail(
                "target config was read despite preloaded targets"
            )

            result = io.acquire_current_target(
                (0.5, 0.5, 0.5),
                (90.0, 10.0),
                reach=0.5,
                target_blocks=frozenset({"minecraft:stone"}),
            )
        finally:
            io.get_area = original_get_area
            io.acquire_target = original_acquire_target
            io.load_target_blocks = original_load_target_blocks

        self.assertEqual((1.0, 2.0), result)

    def test_acquire_current_target_encodes_blocks_before_native_bridge(self):
        captured = {}
        original_get_area = io.get_area
        original_acquire_target = io.acquire_target

        def fake_get_area(position, reach, *, await_region=True):
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
            return 0.25, -0.5

        with tempfile.TemporaryDirectory() as temp_dir:
            target_config = Path(temp_dir) / "targets.txt"
            target_config.write_text(
                "minecraft:oak_slab\nminecraft:unknown_block\n",
                encoding="utf-8",
            )
            try:
                io.get_area = fake_get_area
                io.acquire_target = fake_acquire_target

                result = io.acquire_current_target(
                    (0.5, 0.5, 0.5),
                    (90.0, 10.0),
                    reach=0.5,
                    target_config=target_config,
                )
            finally:
                io.get_area = original_get_area
                io.acquire_target = original_acquire_target

        self.assertEqual((0.25, -0.5), result)
        self.assertEqual((0.5, 0.5, 0.5), captured["position"])
        self.assertEqual((90.0, 10.0), captured["orientation"])
        self.assertEqual(SHAPE_CATALOG_VERSION, captured["shape_catalog_version"])
        self.assertEqual(3, captured["side"])
        self.assertEqual(0.5, captured["reach"])
        self.assertEqual(27, len(captured["shape_ids"]))
        self.assertEqual(SHAPE_ID_BY_NAME["empty"], captured["shape_ids"][0])
        self.assertEqual(SHAPE_ID_BY_NAME["slab_top"], captured["shape_ids"][1])
        self.assertEqual(SHAPE_ID_BY_NAME["full_cube"], captured["shape_ids"][2])
        self.assertEqual(SHAPE_ID_BY_NAME["empty"], captured["shape_ids"][3])
        self.assertEqual([1, 2], captured["target_indices"])

    def test_acquire_current_target_rejects_large_cube_before_uint16_targets(self):
        original_get_area = io.get_area
        block_count = 41 * 41 * 41

        def fake_get_area(position, reach, *, await_region=True):
            return [
                (
                    (index, 0, 0),
                    "minecraft:stone" if index == block_count - 1 else "minecraft:air",
                )
                for index in range(block_count)
            ]

        try:
            io.get_area = fake_get_area
            with self.assertRaisesRegex(ValueError, "side must be <= 39"):
                io.acquire_current_target(
                    (0.5, 0.5, 0.5),
                    (90.0, 10.0),
                    reach=20.0,
                    target_blocks=frozenset({"minecraft:stone"}),
                )
        finally:
            io.get_area = original_get_area

    def test_load_target_blocks_uses_literal_line_matching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target_config = Path(temp_dir) / "targets.txt"
            target_config.write_text(
                "\n"
                "# comment\n"
                " minecraft:stone \n"
                "minecraft:cobblestone # inline comment\n",
                encoding="utf-8",
            )

            self.assertEqual(
                frozenset({"minecraft:stone", "minecraft:cobblestone"}),
                io.load_target_blocks(target_config),
            )

    def test_acquire_current_target_skips_native_call_without_targets(self):
        original_get_area = io.get_area
        original_acquire_target = io.acquire_target
        native_called = False

        def fake_get_area(position, reach, *, await_region=True):
            return [
                ((index, 0, 0), "minecraft:air")
                for index in range(27)
            ]

        def fake_acquire_target(*_args):
            nonlocal native_called
            native_called = True
            return 0.0, 0.0

        with tempfile.TemporaryDirectory() as temp_dir:
            target_config = Path(temp_dir) / "targets.txt"
            target_config.write_text("minecraft:stone\n", encoding="utf-8")
            try:
                io.get_area = fake_get_area
                io.acquire_target = fake_acquire_target
                result = io.acquire_current_target(
                    (0.5, 0.5, 0.5),
                    (90.0, 10.0),
                    reach=0.5,
                    target_config=target_config,
                )
            finally:
                io.get_area = original_get_area
                io.acquire_target = original_acquire_target

        self.assertIsNone(result)
        self.assertFalse(native_called)

    def test_acquire_current_target_records_python_and_native_timings(self):
        original_get_area = io.get_area
        original_acquire_target = io.acquire_target

        def fake_get_area(
            position,
            reach,
            *,
            await_region=True,
            timings=None,
        ):
            if timings is not None:
                timings.total_ms = 1.0
            return [
                ((index, 0, 0), "minecraft:stone")
                for index in range(27)
            ]

        def fake_acquire_target(*_args):
            return 0.0, 0.0

        with tempfile.TemporaryDirectory() as temp_dir:
            target_config = Path(temp_dir) / "targets.txt"
            target_config.write_text("minecraft:stone\n", encoding="utf-8")
            timings = io.ScanTimings()
            try:
                io.get_area = fake_get_area
                io.acquire_target = fake_acquire_target
                result = io.acquire_current_target(
                    (0.5, 0.5, 0.5),
                    (90.0, 10.0),
                    reach=0.5,
                    target_config=target_config,
                    timings=timings,
                )
            finally:
                io.get_area = original_get_area
                io.acquire_target = original_acquire_target

        self.assertEqual((0.0, 0.0), result)
        self.assertEqual(1.0, timings.area.total_ms)
        self.assertGreaterEqual(timings.target_config_ms, 0.0)
        self.assertGreaterEqual(timings.target_match_ms, 0.0)
        self.assertGreaterEqual(timings.shape_encode_ms, 0.0)
        self.assertGreaterEqual(timings.native_call_ms, 0.0)
        self.assertGreaterEqual(timings.total_ms, timings.native_call_ms)


if __name__ == "__main__":
    unittest.main()
