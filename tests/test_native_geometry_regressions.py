import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


def boost_include_dir():
    configured = os.environ.get("BOOST_INCLUDEDIR")
    candidates = [configured, "/usr/local/include", "/usr/include"]
    for candidate in candidates:
        if candidate and (
            Path(candidate) / "boost/multiprecision/cpp_int.hpp"
        ).is_file():
            return Path(candidate)
    return None


class NativeGeometryRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            raise unittest.SkipTest("No C++ compiler available")
        boost_include = boost_include_dir()
        if boost_include is None:
            raise unittest.SkipTest(
                "Boost headers unavailable; install boost-devel or set BOOST_INCLUDEDIR"
            )

        cls.project_root = Path(__file__).resolve().parents[1]
        cls.temp_dir = tempfile.TemporaryDirectory(
            prefix="minescript-geometry-regression-test-"
        )
        cls.executable = Path(cls.temp_dir.name) / "geometry_regression_test"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-I",
                str(cls.project_root / "native/include"),
                "-I",
                str(boost_include),
                str(
                    cls.project_root
                    / "native/tests/geometry_regression_test.cpp"
                ),
                str(cls.project_root / "native/src/branch_bound.cpp"),
                str(cls.project_root / "native/src/clipping.cpp"),
                str(cls.project_root / "native/src/constraint_region.cpp"),
                str(cls.project_root / "native/src/exact_branch_bound.cpp"),
                str(cls.project_root / "native/src/exact_geometry.cpp"),
                str(cls.project_root / "native/src/exact_geometry_store.cpp"),
                str(cls.project_root / "native/src/exact_projection.cpp"),
                str(cls.project_root / "native/src/visibility.cpp"),
                "-o",
                str(cls.executable),
            ],
            check=True,
            cwd=cls.project_root,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def run_regression(self, case_name):
        subprocess.run(
            [str(self.executable), case_name],
            check=True,
            cwd=self.project_root,
        )

    def test_nearly_collinear_orientation_keeps_exact_sign(self):
        self.run_regression("robust_orientation")

    def test_face_crossing_near_plane_is_clipped_instead_of_discarded(self):
        self.run_regression("near_plane")

    def test_selected_target_point_respects_reach(self):
        self.run_regression("reach")

    def test_thin_visible_sliver_uses_point_with_stable_clearance(self):
        self.run_regression("thin_sliver")

    def test_zero_angle_target_corner_moves_to_face_interior(self):
        self.run_regression("target_corner")

    def test_selected_point_survives_float_camera_orientation(self):
        self.run_regression("float_camera_edge_clearance")

    def test_adjacent_full_cube_hides_shared_target_face(self):
        self.run_regression("adjacent_full_cube_occlusion")


if __name__ == "__main__":
    unittest.main()
