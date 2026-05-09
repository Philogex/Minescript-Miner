"""Minescript-facing IO helpers.

This module is the only layer in the package that should talk to the
`minescript` runtime directly.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from minescript_miner.adapter.block_ids import DEFAULT_TARGET_BLOCKS
from minescript_miner.adapter.native_bridge import Orientation, ScanPosition, scan_region_debug
from minescript_miner.minescript.world import read_region_blocks


def scan_current_region_debug(
    position: ScanPosition,
    orientation: Orientation,
    reach: float = 4.8,
    *,
    target_blocks: Sequence[str] = tuple(DEFAULT_TARGET_BLOCKS),
) -> Tuple[float, float]:
    _, _, block_strings = read_region_blocks(position, reach)
    return scan_region_debug(
        position,
        orientation,
        block_strings,
        target_blocks=target_blocks,
    )
