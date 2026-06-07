import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class NativeGeometryRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            raise unittest.SkipTest("No C++ compiler available")

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
                str(
                    cls.project_root
                    / "native/tests/geometry_regression_test.cpp"
                ),
                str(cls.project_root / "native/src/branch_bound.cpp"),
                str(cls.project_root / "native/src/clipping.cpp"),
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

    @unittest.expectedFailure
    def test_face_crossing_near_plane_is_clipped_instead_of_discarded(self):
        self.run_regression("near_plane")

    @unittest.expectedFailure
    def test_selected_target_point_respects_reach(self):
        self.run_regression("reach")

    @unittest.expectedFailure
    def test_thin_visible_sliver_uses_point_with_stable_clearance(self):
        self.run_regression("thin_sliver")


if __name__ == "__main__":
    unittest.main()
