import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.native_build_support import BOOST_INCLUDE


class NativeGeometryTest(unittest.TestCase):
    def test_projective_kernel(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="minescript-geometry-test-"
        ) as temp_dir:
            executable = Path(temp_dir) / "geometry_test"
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
                    str(project_root / "native/tests/geometry_test.cpp"),
                    str(project_root / "native/src/geometry/geometry.cpp"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=project_root,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
