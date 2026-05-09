import math
from typing import List, Optional, Tuple, Any, Dict

import minescript as m


BlockPos = Tuple[int, int, int]


def _chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _parse_block_string(bs: Optional[str]) -> Tuple[str, str, Dict[str, Any]]:
    if bs is None:
        bs = "minecraft:air"

    s = bs.strip()
    meta: Dict[str, Any] = {}
    if '[' in s and s.endswith(']'):
        head, rest = s.split('[', 1)
        rest = rest[:-1]
        kvs = rest.split(',')
        for kv in kvs:
            if '=' not in kv:
                continue
            k, v = kv.split('=', 1)
            k = k.strip()
            v = v.strip()
            if v in ('true', 'false'):
                meta[k] = (v == 'true')
            else:
                try:
                    meta[k] = int(v)
                except Exception:
                    meta[k] = v
        base = head
    else:
        base = s

    base = base.strip()
    base_lower = base.lower()

    if base_lower.endswith(':water') or base_lower == 'minecraft:water':
        simple = 'transparent'
    elif base_lower.endswith(':air') or base_lower == 'minecraft:air':
        simple = 'transparent'
    elif base_lower.endswith('_slab') or ':slab' in base_lower:
        simple = 'slab'
        if 'type' in meta:
            meta['half'] = meta.get('type')
            meta.pop('type', None)
    elif base_lower.endswith('_stairs') or 'stairs' in base_lower or ':stair' in base_lower:
        simple = 'stair'
    elif base_lower.endswith('_pane') or 'pane' in base_lower:
        simple = 'pane'
        conns = []
        for d in ('east', 'west', 'north', 'south'):
            if meta.get(d, False):
                conns.append(d)
        meta['connections'] = conns
    else:
        short = base_lower.split(':')[-1]
        simple = short
    return base, simple, meta


def _dda_ray_voxels(px, py, pz, ex, ey, ez) -> List[BlockPos]:
    x = int(math.floor(px))
    y = int(math.floor(py))
    z = int(math.floor(pz))

    tx = int(math.floor(ex))
    ty = int(math.floor(ey))
    tz = int(math.floor(ez))

    dx = ex - px
    dy = ey - py
    dz = ez - pz

    dist2 = dx * dx + dy * dy + dz * dz
    if dist2 <= 1e-18:
        return [(x, y, z)]

    inf = float("inf")

    def axis_step_tmax_tdelta(delta, start, block):
        if abs(delta) <= 1e-12:
            return 0, inf, inf
        if delta > 0:
            inv = 1.0 / delta
            return 1, ((block + 1.0) - start) * inv, abs(inv)
        inv = 1.0 / delta
        return -1, (start - block) * (-inv), abs(inv)

    step_x, tmax_x, tdelta_x = axis_step_tmax_tdelta(dx, px, x)
    step_y, tmax_y, tdelta_y = axis_step_tmax_tdelta(dy, py, y)
    step_z, tmax_z, tdelta_z = axis_step_tmax_tdelta(dz, pz, z)

    out = [(x, y, z)]
    max_steps = int(math.ceil(math.sqrt(dist2) * 3.0)) + 10000

    for _ in range(max_steps):
        if (x == tx) and (y == ty) and (z == tz):
            break

        if (tmax_x > 1.0) and (tmax_y > 1.0) and (tmax_z > 1.0):
            if not ((x == tx) and (y == ty) and (z == tz)):
                out.append((tx, ty, tz))
            break

        if (tmax_x <= tmax_y) and (tmax_x <= tmax_z):
            x += step_x
            tmax_x += tdelta_x
        elif (tmax_y <= tmax_x) and (tmax_y <= tmax_z):
            y += step_y
            tmax_y += tdelta_y
        else:
            z += step_z
            tmax_z += tdelta_z

        out.append((x, y, z))
    else:
        if not ((x == tx) and (y == ty) and (z == tz)):
            out.append((tx, ty, tz))

    return out


def _expand_neighbors(voxels: List[BlockPos], radius: int = 1) -> List[BlockPos]:
    visited = set()
    for x0, y0, z0 in voxels:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    visited.add((x0 + dx, y0 + dy, z0 + dz))
    return sorted(visited)

# ------------------------------
# user helper functions
# ------------------------------

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


def read_blocks_getblocklist(positions: List[List[int]], chunk_size: int = 1000) -> List[Optional[str]]:
    block_strings: List[Optional[str]] = []
    for chunk in _chunk_list(positions, chunk_size):
        block_strings.extend(m.getblocklist(list(chunk)))
    return block_strings


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


def parse_block_results(
    positions: List[List[int]],
    block_strings: List[Optional[str]],
) -> List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]]:
    out: List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]] = []
    for pos, bs in zip(positions, block_strings):
        base, simple, meta = _parse_block_string(bs)
        out.append(((pos[0], pos[1], pos[2]), base, simple, meta))
    return out


def count_missing_block_strings(block_strings: List[Optional[str]]) -> int:
    return sum(1 for block_string in block_strings if block_string is None)


def get_area(
    position: Tuple[float, float, float],
    reach: float = 4.8,
    pitch_range: Tuple[float, float] = (-90.0, 90.0),
    read_strategy: str = "getblocklist",
    chunk_size: int = 1000,
) -> List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]]:
    positions = positions_within_reach(position, reach, pitch_range)

    if read_strategy == "getblocklist":
        block_strings = read_blocks_getblocklist(positions, chunk_size)
    elif read_strategy in ("region", "region_prune", "get_block_region"):
        block_strings = read_blocks_region_prune(positions)
    else:
        raise ValueError(f"Unknown get_area read_strategy: {read_strategy}")

    return parse_block_results(positions, block_strings)


def get_area_getblocklist(
    position: Tuple[float, float, float],
    reach: float = 4.8,
    pitch_range: Tuple[float, float] = (-90.0, 90.0),
    chunk_size: int = 1000,
) -> List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]]:
    return get_area(position, reach, pitch_range, read_strategy="getblocklist", chunk_size=chunk_size)


def get_area_region_prune(
    position: Tuple[float, float, float],
    reach: float = 4.8,
    pitch_range: Tuple[float, float] = (-90.0, 90.0),
) -> List[Tuple[BlockPos, str, str, Optional[Dict[str, Any]]]]:
    return get_area(position, reach, pitch_range, read_strategy="region_prune")


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
        voxels_exp = _expand_neighbors(voxels, radius=1)

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
