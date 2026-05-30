import sys
import types
import unittest


sys.modules.setdefault("minescript", types.SimpleNamespace())

import shape_fixture_check
import shape_fixture_overlay
from minescript_miner.adapter.block_ids import SHAPE_NAMES


class ShapeFixtureCheckTest(unittest.TestCase):
    def test_fixture_generator_covers_every_shape_id(self):
        fixtures = shape_fixture_check.build_shape_fixtures()
        shape_fixture_check.assert_fixture_coverage(fixtures)

        covered = {fixture.expected_shape_id for fixture in fixtures}
        self.assertEqual(set(range(len(SHAPE_NAMES))), covered)

    def test_fixture_generator_includes_aliases_and_fallback(self):
        fixtures = shape_fixture_check.build_shape_fixtures()

        self.assertTrue(any(fixture.mode == "fallback" for fixture in fixtures))
        self.assertTrue(any("iron_bars" in fixture.label for fixture in fixtures))
        self.assertTrue(any("glass_pane" in fixture.block_state for fixture in fixtures))

    def test_overlay_defines_non_empty_corners_for_non_empty_shapes(self):
        for shape_id, shape_name in enumerate(SHAPE_NAMES):
            corners = shape_fixture_overlay.corner_points(shape_id, (0, 0, 0))
            if shape_name == "empty":
                self.assertEqual(set(), corners)
            else:
                self.assertTrue(corners, shape_name)


if __name__ == "__main__":
    unittest.main()
