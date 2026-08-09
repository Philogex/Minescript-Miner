import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class NativeAimTargetRegionTest(unittest.TestCase):
    def test_projection_and_visible_region_membership(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="minecraft-miner-aim-target-region-test-"
        ) as temp_dir:
            executable = Path(temp_dir) / "aim_target_region_test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-I",
                    str(project_root / "native/include"),
                    str(project_root / "native/tests/aim_target_region_test.cpp"),
                    str(project_root / "native/src/aim/angle.cpp"),
                    str(
                        project_root
                        / "native/src/aim/geometry_feedback_sigmadrift.cpp"
                    ),
                    str(project_root / "native/src/aim/minimum_jerk.cpp"),
                    str(project_root / "native/src/aim/sigmadrift.cpp"),
                    str(project_root / "native/src/aim/target_region.cpp"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=project_root,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
