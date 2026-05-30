import unittest

from minescript_miner.adapter.block_ids import CATALOG_VERSION, SHAPE_NAMES
from minescript_miner.api import geometry_catalog_debug


class GeometryCatalogTest(unittest.TestCase):
    def test_geometry_catalog_matches_python_block_mapping(self):
        debug = geometry_catalog_debug()

        self.assertEqual(CATALOG_VERSION, debug["version"])
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


if __name__ == "__main__":
    unittest.main()
