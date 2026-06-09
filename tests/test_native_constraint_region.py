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


class NativeConstraintRegionTest(unittest.TestCase):
    def test_geometry_interning_and_constraint_regions(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")

        boost_include = boost_include_dir()
        if boost_include is None:
            self.skipTest(
                "Boost headers unavailable; install boost-devel or set BOOST_INCLUDEDIR"
            )

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
            prefix="minescript-constraint-region-test-"
        ) as temp_dir:
            executable = Path(temp_dir) / "constraint_region_test"
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
                    str(boost_include),
                    str(
                        project_root
                        / "native/tests/constraint_region_test.cpp"
                    ),
                    str(project_root / "native/src/geometry.cpp"),
                    str(
                        project_root
                        / "native/src/geometry_store.cpp"
                    ),
                    str(project_root / "native/src/constraint_region.cpp"),
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=project_root,
            )
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
