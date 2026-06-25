import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.native_build_support import BOOST_INCLUDE


class NativeRandomScanPipelineTest(unittest.TestCase):
    def test_uniform_side_39_random_scan_pipeline(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="minescript-random-scan-pipeline-test-"
        ) as temp_dir:
            executable = Path(temp_dir) / "random_scan_pipeline_test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-I",
                    str(project_root / "native/include"),
                    "-I",
                    str(BOOST_INCLUDE),
                    str(project_root / "native/tests/random_scan_pipeline_test.cpp"),
                    str(project_root / "native/src/aim/angle.cpp"),
                    str(project_root / "native/src/geometry/clipping.cpp"),
                    str(project_root / "native/src/geometry/constraint_region.cpp"),
                    str(project_root / "native/src/scanner/branch_bound.cpp"),
                    str(project_root / "native/src/geometry/geometry.cpp"),
                    str(project_root / "native/src/geometry/geometry_store.cpp"),
                    str(project_root / "native/src/scanner/projection.cpp"),
                    str(project_root / "native/src/catalog/geometry_catalog.cpp"),
                    str(project_root / "native/src/scanner/scan_region.cpp"),
                    str(project_root / "native/src/scanner/target_solver.cpp"),
                    str(project_root / "native/src/scanner/view_projection.cpp"),
                    str(project_root / "native/src/scanner/reach_projection.cpp"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=project_root,
            )
            result = subprocess.run(
                [
                    str(executable),
                    os.environ.get("MINESCRIPT_MINER_RANDOM_SCAN_SEED", "0x1234"),
                    os.environ.get("MINESCRIPT_MINER_RANDOM_SCAN_CASES", "3"),
                    os.environ.get("MINESCRIPT_MINER_RANDOM_SCAN_DENSITY", "0.25"),
                    os.environ.get("MINESCRIPT_MINER_RANDOM_SCAN_TARGETS", "5"),
                ],
                check=True,
                cwd=project_root,
                text=True,
                stdout=subprocess.PIPE,
            )
            self.assertIn("random_scan", result.stdout)
            self.assertIn("side=39", result.stdout)


if __name__ == "__main__":
    unittest.main()
