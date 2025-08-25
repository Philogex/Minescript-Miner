from visibility_scanner.scanner import BlockPos, _chunk_list, _parse_block_string, _dda_ray_voxels, _expand_neighbors

import math
from typing import List, Optional, Tuple, Any, Dict

import minescript as m

# ------------------------------
# user helper functions
# ------------------------------

def get_area(position: Tuple[float, float, float],
    reach: float = 4.8,
    pitch_range: Tuple[float, float] = (-90.0, 90.0),
) -> List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]]:
    px, py, pz = position
    r = float(reach)
    ir = int(math.ceil(r))
    pitch_min, pitch_max = pitch_range

    positions: List[List[int]] = []
    for dx in range(-ir, ir + 1):
        for dy in range(-ir, ir + 1):
            for dz in range(-ir, ir + 1):
                if (dx * dx + dy * dy + dz * dz) <= (r + 1e-9 - 0.5) ** 2:
                    x = int(math.floor(px)) + dx
                    y = int(math.floor(py)) + dy
                    z = int(math.floor(pz)) + dz
                    
                    vx = (x + 0.5) - px
                    vy = (y + 0.5) - py
                    vz = (z + 0.5) - pz
                    h = math.hypot(vx, vz)
                    pitch_deg = math.degrees(math.atan2(-vy, h))

                    if pitch_min <= pitch_deg <= pitch_max:
                        positions.append([x, y, z])

    block_strings: List[str] = []
    for chunk in _chunk_list(positions, 1000):
        res = m.getblocklist(list(chunk))
        block_strings.extend(res)

    out: List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]] = []
    for pos, bs in zip(positions, block_strings):
        base, simple, meta = _parse_block_string(bs)
        out.append(((pos[0], pos[1], pos[2]), base, simple, meta))
    return out

def get_line(
    position: Tuple[float, float, float],
    target: Tuple[float, float, float]
) -> List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]]:
    px, py, pz = position
    tx, ty, tz = target

    sbx, sby, sbz = int(math.floor(px)), int(math.floor(py)), int(math.floor(pz))
    tbx, tby, tbz = int(math.floor(tx)), int(math.floor(ty)), int(math.floor(tz))

    offsets = [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    visited_set = set()

    for ox, oy, oz in offsets:
        start_pt = (sbx + ox + 0.5, sby + oy + 0.5, sbz + oz + 0.5)
        end_pt = (tbx + ox + 0.5, tby + oy + 0.5, tbz + oz + 0.5)

        voxels = _dda_ray_voxels(*start_pt, *end_pt)
        voxels_exp = _expand_neighbors(voxels, radius=2)

        for v in voxels_exp:
            visited_set.add((v[0], v[1], v[2]))

    visited = list(visited_set)
    visited.sort(key=lambda v: ( (v[0]+0.5 - px)**2 + (v[1]+0.5 - py)**2 + (v[2]+0.5 - pz)**2 ))

    block_strings: List[str] = []
    for chunk in _chunk_list(visited, 1000):
        block_strings.extend(m.getblocklist(list(chunk)))

    out: List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]] = []
    for pos, bs in zip(visited, block_strings):
        base, simple, meta = _parse_block_string(bs)
        out.append((pos, base, simple, meta))

    return out
