import unittest

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
            )


if __name__ == "__main__":
    unittest.main()
