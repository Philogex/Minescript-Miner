"""User-facing visibility queries built on the native target solver."""

from __future__ import annotations

import math
from array import array
from typing import Optional, Tuple

from minescript_miner.adapter.catalog_contract import MAX_CUBE_SIDE
from minescript_miner.adapter.native_bridge import (
    Orientation,
    ScanPosition,
    acquire_target,
)
from minescript_miner.adapter.shape_catalog import DEFAULT_CATALOG
from minescript_miner.minescript import io
from minescript_miner.minescript.world import BlockPos, cube_block_index


EYE_HEIGHT = 1.62
BlockQueryPosition = Tuple[int, int, int]


def _block_pos(block_pos: BlockQueryPosition) -> BlockPos:
    if len(block_pos) != 3:
        raise ValueError(
            f"block_pos must contain exactly three values, got {block_pos!r}"
        )
    return int(block_pos[0]), int(block_pos[1]), int(block_pos[2])


def _scan_position(position: ScanPosition) -> ScanPosition:
    if len(position) != 3:
        raise ValueError(
            f"position must contain exactly three values, got {position!r}"
        )
    return float(position[0]), float(position[1]), float(position[2])


def _yaw_pitch_to_block_center(
    position: ScanPosition,
    block_pos: BlockPos,
) -> Orientation:
    dx = (block_pos[0] + 0.5) - position[0]
    dy = (block_pos[1] + 0.5) - position[1]
    dz = (block_pos[2] + 0.5) - position[2]
    horizontal = math.hypot(dx, dz)
    if horizontal == 0.0 and dy == 0.0:
        return 0.0, 0.0
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = math.degrees(math.atan2(-dy, horizontal))
    return yaw, pitch


def _cube_bounds_for_target(
    position: ScanPosition,
    block_pos: BlockPos,
) -> Tuple[BlockPos, BlockPos, int]:
    center = (
        math.floor(position[0]),
        math.floor(position[1]),
        math.floor(position[2]),
    )
    half = max(
        abs(block_pos[0] - center[0]),
        abs(block_pos[1] - center[1]),
        abs(block_pos[2] - center[2]),
    )
    side = half * 2 + 1
    if side > MAX_CUBE_SIDE:
        raise ValueError(
            f"target block requires side={side}, but side must be <= {MAX_CUBE_SIDE}"
        )
    return (
        center[0] - half,
        center[1] - half,
        center[2] - half,
    ), (
        center[0] + half,
        center[1] + half,
        center[2] + half,
    ), side


def _reach_to_block(position: ScanPosition, block_pos: BlockPos) -> float:
    farthest_squared = 0.0
    for x in (block_pos[0], block_pos[0] + 1):
        for y in (block_pos[1], block_pos[1] + 1):
            for z in (block_pos[2], block_pos[2] + 1):
                dx = x - position[0]
                dy = y - position[1]
                dz = z - position[2]
                farthest_squared = max(
                    farthest_squared,
                    dx * dx + dy * dy + dz * dz,
                )
    return math.sqrt(farthest_squared)


def _acquire_block_target_from(
    position: ScanPosition,
    block_pos: BlockPos,
) -> Optional[Orientation]:
    min_pos, _max_pos, side = _cube_bounds_for_target(position, block_pos)
    region = io.read_block_region(min_pos, _max_pos)
    block_strings = tuple(region.blocks)
    encoded = DEFAULT_CATALOG.encode_region(side, block_strings)
    target_index = cube_block_index(block_pos, min_pos, side)
    return acquire_target(
        position,
        _yaw_pitch_to_block_center(position, block_pos),
        encoded.shape_catalog_version,
        encoded.side,
        _reach_to_block(position, block_pos),
        encoded.shape_ids,
        array("H", [target_index]),
    )


def get_angle_to_block(block_pos: BlockQueryPosition) -> Optional[Orientation]:
    """Return a visible yaw/pitch from the player's eye position to a block."""

    px, py, pz = io.player_position()
    return _acquire_block_target_from(
        (float(px), float(py) + EYE_HEIGHT, float(pz)),
        _block_pos(block_pos),
    )


def can_see_block(
    source: ScanPosition,
    block_pos: BlockQueryPosition,
) -> bool:
    """Return whether any analytically visible target point exists on a block."""

    return _acquire_block_target_from(
        _scan_position(source),
        _block_pos(block_pos),
    ) is not None
