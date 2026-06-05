import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class NativeClippingTest(unittest.TestCase):
    def test_native_clipping_geometry(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="minescript-clipping-test-") as temp_dir:
            executable = Path(temp_dir) / "clipping_test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-I",
                    str(project_root / "native/include"),
                    str(project_root / "native/tests/clipping_test.cpp"),
                    str(project_root / "native/src/clipping.cpp"),
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
