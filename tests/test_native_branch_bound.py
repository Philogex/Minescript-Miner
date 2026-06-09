import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class NativeBranchBoundTest(unittest.TestCase):
    def test_native_branch_bound_solver(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("No C++ compiler available")

        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="minescript-branch-bound-test-") as temp_dir:
            executable = Path(temp_dir) / "branch_bound_test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-I",
                    str(project_root / "native/include"),
                    str(project_root / "native/tests/branch_bound_test.cpp"),
                    str(project_root / "native/src/branch_bound.cpp"),
                    str(project_root / "native/src/clipping.cpp"),
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
