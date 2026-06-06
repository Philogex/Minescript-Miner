import importlib.util
import sys
import types
import unittest
from pathlib import Path


class MinerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project_root / "src"))
        minescript = sys.modules.setdefault(
            "minescript",
            types.SimpleNamespace(
                script_loop=object(),
                EventType=types.SimpleNamespace(KEY="key"),
                player_press_attack=lambda _pressed: None,
            ),
        )
        minescript.script_loop = getattr(minescript, "script_loop", object())
        minescript.EventType = getattr(
            minescript,
            "EventType",
            types.SimpleNamespace(KEY="key"),
        )
        minescript.player_press_attack = getattr(
            minescript,
            "player_press_attack",
            lambda _pressed: None,
        )
        minescript.player_orientation = getattr(
            minescript,
            "player_orientation",
            lambda: (0.0, 0.0),
        )
        minescript.player_set_orientation = getattr(
            minescript,
            "player_set_orientation",
            lambda _yaw, _pitch: True,
        )
        spec = importlib.util.spec_from_file_location("miner_entrypoint", project_root / "miner.py")
        assert spec is not None and spec.loader is not None
        cls.miner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.miner)

    def setUp(self):
        self.miner.active.clear()

    def test_targeted_block_must_match_configured_literal(self):
        targeted = types.SimpleNamespace(
            position=(1, 2, 3),
            type="minecraft:stone[waterlogged=false]",
        )
        self.assertTrue(
            self.miner.targeted_block_is_configured(
                targeted,
                frozenset({"minecraft:stone"}),
            )
        )
        self.assertFalse(
            self.miner.targeted_block_is_configured(
                targeted,
                frozenset({"minecraft:dirt"}),
            )
        )

    def test_mining_releases_attack_after_block_changes(self):
        calls = []
        target = types.SimpleNamespace(position=(1, 2, 3), type="minecraft:stone")
        changed = types.SimpleNamespace(position=(1, 2, 4), type="minecraft:stone")
        targeted_blocks = iter((target, changed))

        original_get_targeted_block = getattr(
            self.miner.m,
            "player_get_targeted_block",
            None,
        )
        original_press_attack = self.miner.m.player_press_attack
        try:
            self.miner.m.player_get_targeted_block = (
                lambda _reach: next(targeted_blocks)
            )
            self.miner.m.player_press_attack = calls.append
            self.miner.active.set()

            self.miner.mine_targeted_block(frozenset({"minecraft:stone"}))
        finally:
            if original_get_targeted_block is None:
                del self.miner.m.player_get_targeted_block
            else:
                self.miner.m.player_get_targeted_block = original_get_targeted_block
            self.miner.m.player_press_attack = original_press_attack

        self.assertEqual([True, False], calls)


if __name__ == "__main__":
    unittest.main()
