"""Thin wrappers around direct Minescript runtime access."""

from __future__ import annotations

from typing import Tuple

import minescript as m

from minescript_miner.aim import (
    DEFAULT_FALLBACK_ANGULAR_STEP_DEG,
    sensitivity_to_angular_step_deg,
)
from minescript_miner.minescript.runtime import query
from minescript_miner.minescript.world import (
    BlockPos,
    read_block_region as world_read_block_region,
)


def player_position() -> Tuple[float, float, float]:
    return query(m.player_position)


def player_orientation() -> Tuple[float, float]:
    return query(m.player_orientation)


def set_orientation(yaw: float, pitch: float) -> None:
    query(m.player_set_orientation, yaw, pitch)


def read_block_region(pos1: BlockPos, pos2: BlockPos):
    return world_read_block_region(pos1, pos2)


def _java_mouse_sensitivity() -> float:
    from java import JavaClass

    Minecraft = JavaClass("net.minecraft.client.Minecraft")
    minecraft = Minecraft.getInstance()
    options = minecraft.options

    for name in ("sensitivity", "mouseSensitivity"):
        try:
            member = getattr(options, name)
        except Exception:
            continue
        value = member() if callable(member) else member
        getter = getattr(value, "get", None)
        if callable(getter):
            value = getter()
        return float(value)

    raise AttributeError("Minecraft options expose no known mouse sensitivity member")


def minecraft_angular_step_deg(
    fallback: float = DEFAULT_FALLBACK_ANGULAR_STEP_DEG,
) -> float:
    try:
        return sensitivity_to_angular_step_deg(query(_java_mouse_sensitivity))
    except Exception:
        return fallback
