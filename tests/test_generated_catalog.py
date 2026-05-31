import subprocess
import sys
import unittest
from pathlib import Path


class GeneratedCatalogTest(unittest.TestCase):
    def test_generated_catalog_files_are_current(self):
        project_root = Path(__file__).resolve().parents[1]
        subprocess.run(
            [sys.executable, "tools/generate_shape_catalog.py", "--check"],
            cwd=project_root,
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
