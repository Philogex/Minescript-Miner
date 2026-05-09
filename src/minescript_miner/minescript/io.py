"""Minescript-facing IO helpers.

This module is the only layer in the package that should talk to the
`minescript` runtime directly.
"""

from __future__ import annotations

from typing import Tuple

from minescript_miner.adapter.native_bridge import Orientation, ScanPosition, scan_region_debug
from minescript_miner.minescript.world import read_region_blocks


def scan_current_region_debug(
    position: ScanPosition,
    orientation: Orientation,
    reach: float = 4.8,
) -> Tuple[float, float]:
    min_pos, max_pos, block_strings = read_region_blocks(position, reach)
    side = max_pos[0] - min_pos[0] + 1
    if (
        max_pos[1] - min_pos[1] + 1 != side
        or max_pos[2] - min_pos[2] + 1 != side
    ):
        raise ValueError(f"Expected a cube region, got min={min_pos} max={max_pos}")

    return scan_region_debug(
        position,
        orientation,
        side,
        block_strings,
    )
