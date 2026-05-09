"""Minescript integration layer for world reads and player actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import _minescript_miner_native as native


ScanPosition = Tuple[float, float, float]
Orientation = Tuple[float, float]
BlockPos = Tuple[int, int, int]


# Prototype IDs for the native visibility rewrite. The exact catalog can become
# richer later; for the transfer test we only need deterministic small integers.
BLOCK_ID_TRANSPARENT = 0
BLOCK_ID_FULL = 1
BLOCK_ID_STAIRS = 2
BLOCK_ID_SLAB = 3
BLOCK_ID_PANE = 4
BLOCK_ID_TARGET = 5

DEFAULT_TARGET_BLOCKS = frozenset(
    {
        "minecraft:diamond_ore",
        "minecraft:deepslate_diamond_ore",
    }
)


@dataclass(frozen=True)
class NativeRegionInput:
    position: ScanPosition
    orientation: Orientation
    min_pos: BlockPos
    max_pos: BlockPos
    side: int
    block_ids: List[int]


def fixed_cube_bounds(position: ScanPosition, reach: float = 4.8) -> Tuple[BlockPos, BlockPos]:
    """Return the fixed cube queried for the native prototype."""

    half = int(math.ceil(reach))
    px, py, pz = position
    center_x = int(math.floor(px))
    center_y = int(math.floor(py))
    center_z = int(math.floor(pz))
    return (
        center_x - half,
        center_y - half,
        center_z - half,
    ), (
        center_x + half,
        center_y + half,
        center_z + half,
    )


def block_string_to_native_id(
    block_string: Optional[str],
    target_blocks: Sequence[str] = tuple(DEFAULT_TARGET_BLOCKS),
) -> int:
    """Map a Minescript block state string to the prototype native block id."""

    if block_string is None:
        return BLOCK_ID_TRANSPARENT

    base = block_string.split("[", 1)[0].strip().lower()
    if not base:
        return BLOCK_ID_TRANSPARENT

    if base in target_blocks:
        return BLOCK_ID_TARGET
    if base.endswith(":air") or base == "minecraft:air":
        return BLOCK_ID_TRANSPARENT
    if base.endswith(":water") or base == "minecraft:water":
        return BLOCK_ID_TRANSPARENT
    if base.endswith("_stairs") or "stairs" in base or ":stair" in base:
        return BLOCK_ID_STAIRS
    if base.endswith("_slab") or ":slab" in base:
        return BLOCK_ID_SLAB
    if base.endswith("_pane") or "pane" in base:
        return BLOCK_ID_PANE
    return BLOCK_ID_FULL


def read_native_region_input(
    position: ScanPosition,
    orientation: Optional[Orientation] = None,
    reach: float = 4.8,
    *,
    target_blocks: Sequence[str] = tuple(DEFAULT_TARGET_BLOCKS),
    await_region: bool = True,
) -> NativeRegionInput:
    """Read a fixed cube with Minescript and encode it for the native prototype."""

    import minescript as m

    if orientation is None:
        yaw, pitch = m.player_orientation()
        orientation = (float(yaw), float(pitch))

    min_pos, max_pos = fixed_cube_bounds(position, reach)
    if await_region:
        m.await_loaded_region(min_pos[0], min_pos[2], max_pos[0], max_pos[2])

    region = m.get_block_region(min_pos, max_pos)
    side = max_pos[0] - min_pos[0] + 1
    block_ids = [
        block_string_to_native_id(block_string, target_blocks)
        for block_string in region.blocks
    ]

    return NativeRegionInput(
        position=position,
        orientation=orientation,
        min_pos=min_pos,
        max_pos=max_pos,
        side=side,
        block_ids=block_ids,
    )


def scan_region_debug(
    position: ScanPosition,
    orientation: Optional[Orientation] = None,
    reach: float = 4.8,
    *,
    target_blocks: Sequence[str] = tuple(DEFAULT_TARGET_BLOCKS),
) -> Tuple[float, float]:
    """Run the native transfer/logging prototype and return its 2D direction."""

    region_input = read_native_region_input(
        position,
        orientation,
        reach,
        target_blocks=target_blocks,
    )
    x, z = native.scan_region_debug(
        region_input.position,
        region_input.orientation,
        region_input.block_ids,
    )
    return float(x), float(z)
