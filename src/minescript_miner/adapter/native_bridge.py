"""Pure Python adapter between encoded Python data and the native module."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import _minescript_miner_native as native


ScanPosition = Tuple[float, float, float]
Orientation = Tuple[float, float]


def acquire_target(
    position: ScanPosition,
    orientation: Orientation,
    shape_catalog_version: int,
    side: int,
    reach: float,
    shape_ids: Sequence[int],
    target_indices: Sequence[int],
) -> Optional[Orientation]:
    """Return the nearest visible target as Minecraft yaw and pitch."""

    result = native.acquire_target(
        position,
        orientation,
        shape_catalog_version,
        side,
        reach,
        shape_ids,
        target_indices,
    )
    if result is None:
        return None

    yaw, pitch = result
    return float(yaw), float(pitch)
