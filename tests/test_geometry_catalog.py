import os
import tempfile
import unittest
from pathlib import Path

from minescript_miner.adapter.shape_catalog import (
    DEFAULT_CATALOG,
    MAX_CUBE_SIDE,
    SHAPE_CATALOG_VERSION,
    SHAPE_ID_BY_NAME,
    SHAPE_NAMES,
)
from minescript_miner.api import acquire_target, geometry_catalog_debug


class GeometryCatalogTest(unittest.TestCase):
    def test_geometry_catalog_matches_python_block_mapping(self):
        debug = geometry_catalog_debug()

        self.assertEqual(SHAPE_CATALOG_VERSION, debug["shape_catalog_version"])
        self.assertEqual(SHAPE_NAMES, debug["shape_names"])
        self.assertEqual(len(SHAPE_NAMES), debug["shape_count"])
        self.assertEqual(len(SHAPE_NAMES), len(debug["box_counts"]))
        self.assertEqual(len(SHAPE_NAMES), len(debug["face_counts"]))

    def test_geometry_catalog_has_full_cube_geometry(self):
        debug = geometry_catalog_debug()

        self.assertEqual(0, debug["box_counts"][0])
        self.assertEqual(0, debug["face_counts"][0])
        self.assertEqual(1, debug["box_counts"][1])
        self.assertEqual(6, debug["face_counts"][1])

    def test_geometry_catalog_has_boundary_faces_for_non_empty_shapes(self):
        debug = geometry_catalog_debug()
        box_counts = debug["box_counts"]
        face_counts = debug["face_counts"]

        for shape_id, shape_name in enumerate(SHAPE_NAMES[1:], start=1):
            self.assertGreater(box_counts[shape_id], 0, shape_name)
            self.assertGreater(face_counts[shape_id], 0, shape_name)

    def test_oak_blocks_map_to_generic_shapes(self):
        cases = {
            "minecraft:oak_slab[type=bottom,waterlogged=false]": "slab_bottom",
            "minecraft:oak_slab[type=top,waterlogged=false]": "slab_top",
            (
                "minecraft:oak_stairs["
                "facing=north,half=bottom,shape=straight,waterlogged=false"
                "]"
            ): "stairs_north_bottom_straight",
            "minecraft:oak_fence[north=true,east=false,south=false,west=false,waterlogged=false]": "fence_north",
        }

        for block_state, shape_name in cases.items():
            with self.subTest(block_state=block_state):
                self.assertEqual(
                    SHAPE_ID_BY_NAME[shape_name],
                    DEFAULT_CATALOG.shape_id(block_state),
                )

    def test_python_encoder_rejects_cube_sides_above_uint16_index_limit(self):
        with self.assertRaisesRegex(ValueError, "side must be <= 39"):
            DEFAULT_CATALOG.encode_region(MAX_CUBE_SIDE + 1, [])

    def test_native_acquire_target_rejects_cube_sides_above_uint16_index_limit(self):
        with self.assertRaisesRegex(ValueError, "side must be <= 39"):
            acquire_target(
                (0.5, 64.5, 0.5),
                (90.0, 10.0),
                SHAPE_CATALOG_VERSION,
                MAX_CUBE_SIDE + 1,
                [],
                [],
            )

    def test_native_acquire_target_rejects_target_indices_outside_cube(self):
        with self.assertRaisesRegex(ValueError, "target_indices values must be valid"):
            acquire_target(
                (0.5, 64.5, 0.5),
                (90.0, 10.0),
                SHAPE_CATALOG_VERSION,
                3,
                [0] * 27,
                [27],
            )

    def test_native_acquire_target_rejects_shape_ids_outside_catalog(self):
        with self.assertRaisesRegex(ValueError, "shape_ids values must be valid shape ids"):
            acquire_target(
                (0.5, 64.5, 0.5),
                (90.0, 10.0),
                SHAPE_CATALOG_VERSION,
                3,
                [len(SHAPE_NAMES)] * 27,
                [],
            )

    def test_native_acquire_target_preserves_orientation_without_target(self):
        self.assertEqual(
            (90.0, 10.0),
            acquire_target(
                (0.5, 64.5, 0.5),
                (90.0, 10.0),
                SHAPE_CATALOG_VERSION,
                3,
                [SHAPE_ID_BY_NAME["empty"]] * 27,
                [],
            ),
        )

    def test_native_acquire_target_builds_sorted_target_face_candidates(self):
        original_log_path = os.environ.get("MINESCRIPT_MINER_NATIVE_LOG")

        with tempfile.TemporaryDirectory(prefix="minescript-miner-target-order-") as temp_dir:
            log_path = Path(temp_dir) / "native.log"
            os.environ["MINESCRIPT_MINER_NATIVE_LOG"] = str(log_path)
            try:
                shape_ids = [SHAPE_ID_BY_NAME["empty"]] * 27
                for target_index in (10, 14, 16):
                    shape_ids[target_index] = SHAPE_ID_BY_NAME["full_cube"]

                result = acquire_target(
                    (0.5, 0.5, 0.5),
                    (0.0, 0.0),
                    SHAPE_CATALOG_VERSION,
                    3,
                    shape_ids,
                    [10, 14, 16],
                )
            finally:
                if original_log_path is None:
                    os.environ.pop("MINESCRIPT_MINER_NATIVE_LOG", None)
                else:
                    os.environ["MINESCRIPT_MINER_NATIVE_LOG"] = original_log_path

            log_text = log_path.read_text(encoding="utf-8")

        self.assertIn("world_face_count: 18", log_text)
        self.assertIn("target_face_count: 3", log_text)
        self.assertIn("first_target_block_indices: 10 14 16", log_text)
        self.assertIn("first_target_face_indices: 16 6 5", log_text)
        self.assertIn(
            "first_target_face_center_angles_rad: 0.000000 1.570796 3.141593",
            log_text,
        )
        self.assertAlmostEqual(0.0, result[0], places=12)
        self.assertAlmostEqual(0.0, result[1], places=12)
        self.assertIn("solver_found: 1", log_text)
        self.assertIn("returned_orientation_yaw_pitch: 0.000000, 0.000000", log_text)


if __name__ == "__main__":
    unittest.main()
