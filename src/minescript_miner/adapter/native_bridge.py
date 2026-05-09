"""Pure Python adapter between raw Minescript block data and the native module."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import _minescript_miner_native as native

from .block_ids import DEFAULT_TARGET_BLOCKS, block_strings_to_native_ids


ScanPosition = Tuple[float, float, float]
Orientation = Tuple[float, float]


def scan_region_debug(
    position: ScanPosition,
    orientation: Orientation,
    block_strings: Sequence[Optional[str]],
    *,
    target_blocks: Sequence[str] = tuple(DEFAULT_TARGET_BLOCKS),
) -> Tuple[float, float]:
    """Encode raw block strings and run the native transfer/logging prototype."""

    block_ids = block_strings_to_native_ids(block_strings, target_blocks)
    x, z = native.scan_region_debug(
        position,
        orientation,
        block_ids,
    )
    return float(x), float(z)
