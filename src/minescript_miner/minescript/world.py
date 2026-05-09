import math
from typing import List, Optional, Tuple

import minescript as m


BlockPos = Tuple[int, int, int]
BlockSample = Tuple[BlockPos, Optional[str]]


def fixed_cube_bounds(
    position: Tuple[float, float, float],
    reach: float = 4.8,
) -> Tuple[BlockPos, BlockPos]:
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


def read_region_blocks(
    position: Tuple[float, float, float],
    reach: float = 4.8,
    *,
    await_region: bool = True,
) -> Tuple[BlockPos, BlockPos, Tuple[Optional[str], ...]]:
    min_pos, max_pos = fixed_cube_bounds(position, reach)
    if await_region:
        m.await_loaded_region(min_pos[0], min_pos[2], max_pos[0], max_pos[2])

    region = m.get_block_region(min_pos, max_pos)
    return region.min_pos, region.max_pos, tuple(region.blocks)


def positions_within_reach(
    position: Tuple[float, float, float],
    reach: float = 4.8,
    pitch_range: Tuple[float, float] = (-90.0, 90.0),
) -> List[List[int]]:
    px, py, pz = position
    pitch_min, pitch_max = pitch_range

    ir = int(math.ceil(reach))
    sqr = (reach - 0.5) * (reach - 0.5)
    fx, fy, fz = math.floor(px), math.floor(py), math.floor(pz)

    positions: List[List[int]] = []
    for dx in range(-ir, ir + 1):
        for dy in range(-ir, ir + 1):
            for dz in range(-ir, ir + 1):
                if dx * dx + dy * dy > sqr:
                    continue
                if dx * dx + dy * dy + dz * dz > sqr:
                    continue

                x = fx + dx
                y = fy + dy
                z = fz + dz

                vx = (x + 0.5) - px
                vy = (y + 0.5) - py
                vz = (z + 0.5) - pz
                h = math.hypot(vx, vz)
                pitch_deg = math.degrees(math.atan2(-vy, h))

                if pitch_min <= pitch_deg <= pitch_max:
                    positions.append([x, y, z])

    return positions


def bounds_for_positions(positions: List[List[int]]) -> Tuple[BlockPos, BlockPos]:
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    zs = [pos[2] for pos in positions]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def read_blocks_region_prune(positions: List[List[int]]) -> List[Optional[str]]:
    if not positions:
        return []

    pos1, pos2 = bounds_for_positions(positions)
    region = m.get_block_region(pos1, pos2)
    min_x, min_y, min_z = region.min_pos
    x_length = region.x_length
    z_length = region.z_length
    plane_length = x_length * z_length
    blocks = region.blocks

    return [
        blocks[(x - min_x) + (z - min_z) * x_length + (y - min_y) * plane_length]
        for x, y, z in positions
    ]


def get_area(
    position: Tuple[float, float, float],
    reach: float = 4.8,
    pitch_range: Tuple[float, float] = (-90.0, 90.0),
) -> List[BlockSample]:
    positions = positions_within_reach(position, reach, pitch_range)
    block_strings = read_blocks_region_prune(positions)
    return [
        ((pos[0], pos[1], pos[2]), block_string)
        for pos, block_string in zip(positions, block_strings)
    ]
