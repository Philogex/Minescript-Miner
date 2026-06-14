import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.build_minescript_bundle import build_bundle


class BuildMinescriptBundleTest(unittest.TestCase):
    def test_bundles_windows_native_runtime_dependencies(self):
        with tempfile.TemporaryDirectory(
            prefix="minescript-miner-bundle-test-"
        ) as temp_dir:
            temp_path = Path(temp_dir)
            wheel = temp_path / "test-cp39-abi3-win_amd64.whl"
            output = temp_path / "bundle.zip"

            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "_minescript_miner_native.pyd",
                    b"native extension",
                )
                archive.writestr(
                    "test.data/platlib/msvcp140-test.dll",
                    b"native runtime",
                )

            build_bundle(wheel, output)

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    (
                        Path(__file__).resolve().parents[1]
                        / "third_party/boost/LICENSE_1_0.txt"
                    ).read_bytes(),
                    archive.read("Minescript-Miner/BOOST_LICENSE_1_0.txt"),
                )
                self.assertEqual(
                    b"native extension",
                    archive.read(
                        "Minescript-Miner/_minescript_miner_native.pyd"
                    ),
                )
                self.assertEqual(
                    b"native runtime",
                    archive.read("Minescript-Miner/msvcp140-test.dll"),
                )


if __name__ == "__main__":
    unittest.main()
