import sys
import types
import unittest
from contextlib import nullcontext


sys.modules.setdefault(
    "minescript",
    types.SimpleNamespace(
        await_loaded_region=lambda *_args: None,
        script_loop=nullcontext(),
    ),
)

from minescript_miner.minescript import runtime


class MinescriptRuntimeTest(unittest.TestCase):
    def test_query_executes_function_inside_script_loop(self):
        calls = []
        original_script_loop = getattr(runtime.m, "script_loop", None)

        class ScriptLoop:
            def __enter__(self):
                calls.append("enter")

            def __exit__(self, _exc_type, _exc_value, _traceback):
                calls.append("exit")

        try:
            runtime.m.script_loop = ScriptLoop()
            result = runtime.query(
                lambda first, *, second: calls.append((first, second)) or 7,
                3,
                second=4,
            )
        finally:
            if original_script_loop is None:
                del runtime.m.script_loop
            else:
                runtime.m.script_loop = original_script_loop

        self.assertEqual(7, result)
        self.assertEqual(["enter", (3, 4), "exit"], calls)


if __name__ == "__main__":
    unittest.main()
