import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.native_build_support import BOOST_INCLUDE


class NativeBranchBoundTest(unittest.TestCase):
    def test_single_target_solver(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="minescript-branch-bound-test-"
        ) as temp_dir:
            executable = Path(temp_dir) / "branch_bound_test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-I",
                    str(project_root / "native/include"),
                    "-I",
                    str(BOOST_INCLUDE),
                    str(
                        project_root
                        / "native/tests/branch_bound_test.cpp"
                    ),
                    str(project_root / "native/src/angle.cpp"),
                    str(project_root / "native/src/clipping.cpp"),
                    str(project_root / "native/src/constraint_region.cpp"),
                    str(project_root / "native/src/branch_bound.cpp"),
                    str(project_root / "native/src/geometry.cpp"),
                    str(project_root / "native/src/geometry_catalog.cpp"),
                    str(
                        project_root
                        / "native/src/geometry_store.cpp"
                    ),
                    str(project_root / "native/src/projection.cpp"),
                    str(project_root / "native/src/scan_region.cpp"),
                    str(project_root / "native/src/target_solver.cpp"),
                    str(project_root / "native/src/visibility.cpp"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=project_root,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
