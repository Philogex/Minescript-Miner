import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.native_build_support import BOOST_INCLUDE


class NativeScanPipelineTest(unittest.TestCase):
    def test_native_scan_fixtures(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")
        project_root = Path(__file__).resolve().parents[1]
        fixtures = sorted((project_root / "native/tests/fixtures").glob("*.scan"))
        self.assertTrue(fixtures)
        with tempfile.TemporaryDirectory(prefix="minescript-scan-pipeline-test-") as temp_dir:
            executable = Path(temp_dir) / "scan_pipeline_test"
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
                    str(project_root / "native/tests"),
                    "-I",
                    str(BOOST_INCLUDE),
                    str(project_root / "native/tests/scan_pipeline_test.cpp"),
                    str(project_root / "native/tests/scan_fixture.cpp"),
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
            for fixture in fixtures:
                with self.subTest(fixture=fixture.name):
                    subprocess.run([str(executable), str(fixture)], check=True)


if __name__ == "__main__":
    unittest.main()
