"""Minescript-facing IO helpers.

This module is the only layer in the package that should talk to the
`minescript` runtime directly.
"""

from __future__ import annotations

from typing import Tuple

from minescript_miner.adapter.native_bridge import Orientation, ScanPosition, acquire_target
from minescript_miner.adapter.shape_catalog import BlockShapeCatalog, DEFAULT_CATALOG
from minescript_miner.minescript.world import fixed_cube_bounds, get_area


def acquire_current_target(
    position: ScanPosition,
    orientation: Orientation,
    reach: float = 4.8,
    *,
    catalog: BlockShapeCatalog = DEFAULT_CATALOG,
) -> Tuple[float, float]:
    min_pos, max_pos = fixed_cube_bounds(position, reach)
    block_strings = tuple(block_string for _pos, block_string in get_area(position, reach))
    side = max_pos[0] - min_pos[0] + 1
    if (
        max_pos[1] - min_pos[1] + 1 != side
        or max_pos[2] - min_pos[2] + 1 != side
    ):
        raise ValueError(f"Expected a cube region, got min={min_pos} max={max_pos}")

    encoded = catalog.encode_region(side, block_strings)
    return acquire_target(
        position,
        orientation,
        encoded.shape_catalog_version,
        encoded.side,
        encoded.shape_ids,
    )
