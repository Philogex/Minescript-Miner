import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.native_build_support import BOOST_INCLUDE


class NativeProjectionTest(unittest.TestCase):
    def test_projection_and_depth_half_planes(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="minescript-projection-test-"
        ) as temp_dir:
            executable = Path(temp_dir) / "projection_test"
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
                        / "native/tests/projection_test.cpp"
                    ),
                    str(project_root / "native/src/geometry.cpp"),
                    str(
                        project_root
                        / "native/src/geometry_store.cpp"
                    ),
                    str(project_root / "native/src/constraint_region.cpp"),
                    str(project_root / "native/src/projection.cpp"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=project_root,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
