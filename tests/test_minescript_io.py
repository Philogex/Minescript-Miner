import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

loaded_package = sys.modules.get("minescript_miner")
if loaded_package is not None:
    package_path = getattr(loaded_package, "__file__", "")
    if package_path and not package_path.startswith(src_path):
        for module_name in list(sys.modules):
            if module_name == "minescript_miner" or module_name.startswith(
                "minescript_miner."
            ):
                del sys.modules[module_name]

minescript = sys.modules.setdefault(
    "minescript",
    types.SimpleNamespace(
        script_loop=nullcontext(),
        player_set_orientation=lambda _yaw, _pitch: None,
    ),
)
minescript.script_loop = getattr(minescript, "script_loop", nullcontext())
minescript.player_set_orientation = getattr(
    minescript,
    "player_set_orientation",
    lambda _yaw, _pitch: None,
)

from minescript_miner.minescript import io


class MinescriptIoTest(unittest.TestCase):
    def test_minecraft_angular_step_falls_back_when_java_is_unavailable(self):
        original_query = io.query
        try:
            io.query = lambda _function: (_ for _ in ()).throw(RuntimeError("no java"))
            self.assertEqual(0.17, io.minecraft_angular_step_deg(0.17))
        finally:
            io.query = original_query

    def test_set_orientation_forwards_to_minescript_runtime(self):
        calls = []
        original_query = io.query

        def fake_query(function, *args):
            calls.append((function, args))
            return function(*args)

        def fake_set_orientation(yaw, pitch):
            calls.append(("set", (yaw, pitch)))

        original_set_orientation = io.m.player_set_orientation
        try:
            io.query = fake_query
            io.m.player_set_orientation = fake_set_orientation
            io.set_orientation(12.0, -3.0)
        finally:
            io.query = original_query
            io.m.player_set_orientation = original_set_orientation

        self.assertEqual((fake_set_orientation, (12.0, -3.0)), calls[0])
        self.assertEqual(("set", (12.0, -3.0)), calls[1])


if __name__ == "__main__":
    unittest.main()
