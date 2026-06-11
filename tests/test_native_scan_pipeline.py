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


class NativeScanPipelineTest(unittest.TestCase):
    def test_native_scan_fixtures(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")
        boost_include = boost_include_dir()
        if boost_include is None:
            self.skipTest(
                "Boost headers unavailable; install boost-devel or set BOOST_INCLUDEDIR"
            )

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
                    str(boost_include),
                    str(project_root / "native/tests/scan_pipeline_test.cpp"),
                    str(project_root / "native/tests/scan_fixture.cpp"),
                    str(project_root / "native/src/angle.cpp"),
                    str(project_root / "native/src/clipping.cpp"),
                    str(project_root / "native/src/constraint_region.cpp"),
                    str(project_root / "native/src/branch_bound.cpp"),
                    str(project_root / "native/src/geometry.cpp"),
                    str(project_root / "native/src/geometry_store.cpp"),
                    str(project_root / "native/src/projection.cpp"),
                    str(project_root / "native/src/geometry_catalog.cpp"),
                    str(project_root / "native/src/scan_region.cpp"),
                    str(project_root / "native/src/target_solver.cpp"),
                    str(project_root / "native/src/visibility.cpp"),
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
