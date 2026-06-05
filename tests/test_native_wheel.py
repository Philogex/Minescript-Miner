import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path


class NativeWheelTest(unittest.TestCase):
    def test_built_wheel_exports_hello(self):
        project_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory(prefix="minescript-miner-wheel-") as temp_dir:
            temp_path = Path(temp_dir)
            venv_dir = temp_path / "venv"
            wheel = os.environ.get("MINESCRIPT_MINER_WHEEL")

            if wheel is None:
                if importlib.util.find_spec("build") is None:
                    self.skipTest(
                        "Set MINESCRIPT_MINER_WHEEL or install 'build' to run this test."
                    )

                wheelhouse = temp_path / "wheelhouse"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "build",
                        "--wheel",
                        "--outdir",
                        str(wheelhouse),
                        str(project_root),
                    ],
                    check=True,
                    cwd=project_root,
                )

                wheels = sorted(wheelhouse.glob("*.whl"))
                self.assertEqual(1, len(wheels), wheels)
                wheel = str(wheels[0])

            venv.EnvBuilder(with_pip=True).create(venv_dir)
            python = (
                venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
            )

            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    wheel,
                ],
                check=True,
            )

            result = subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "import _minescript_miner_native as native; "
                        "import minescript_miner; "
                        "print(native.hello()); "
                        "print(minescript_miner.hello()); "
                        "print(minescript_miner.geometry_catalog_debug()['version']); "
                        "print(native.acquire_target((0.5, 64.5, 0.5), (90.0, 10.0), 1, 3, [0] * 27, [])); "
                        "print(minescript_miner.acquire_target((0.5, 64.5, 0.5), (90.0, 10.0), 1, 3, [0] * 27, []))"
                    ),
                ],
                check=True,
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "MINESCRIPT_MINER_NATIVE_LOG": str(temp_path / "native.log"),
                },
            )

            self.assertEqual(
                (
                    "hello from native extension\n"
                    "hello from native extension\n"
                    "1\n"
                    "(90.0, 10.0)\n"
                    "(90.0, 10.0)"
                ),
                result.stdout.strip(),
            )


if __name__ == "__main__":
    unittest.main()
