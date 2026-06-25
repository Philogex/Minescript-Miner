import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.native_build_support import BOOST_INCLUDE


class NativeConstraintSubtractionTest(unittest.TestCase):
    def test_convex_region_subtraction(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="minescript-constraint-subtraction-test-"
        ) as temp_dir:
            executable = Path(temp_dir) / "constraint_subtraction_test"
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
                        / "native/tests/constraint_subtraction_test.cpp"
                    ),
                    str(project_root / "native/src/geometry/geometry.cpp"),
                    str(
                        project_root
                        / "native/src/geometry/geometry_store.cpp"
                    ),
                    str(project_root / "native/src/geometry/constraint_region.cpp"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=project_root,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
