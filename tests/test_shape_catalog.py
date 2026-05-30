import unittest

from minescript_miner.adapter.block_ids import CATALOG_VERSION, SHAPE_NAMES
from minescript_miner.api import shape_catalog_debug


class ShapeCatalogTest(unittest.TestCase):
    def test_native_catalog_matches_python_catalog(self):
        debug = shape_catalog_debug()

        self.assertEqual(CATALOG_VERSION, debug["version"])
        self.assertEqual(SHAPE_NAMES, debug["shape_names"])


if __name__ == "__main__":
    unittest.main()
