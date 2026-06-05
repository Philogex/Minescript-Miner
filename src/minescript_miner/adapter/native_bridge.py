"""Pure Python adapter between encoded Python data and the native module."""

from __future__ import annotations

from typing import Sequence, Tuple

import _minescript_miner_native as native


ScanPosition = Tuple[float, float, float]
Orientation = Tuple[float, float]


def acquire_target(
    position: ScanPosition,
    orientation: Orientation,
    shape_catalog_version: int,
    side: int,
    shape_ids: Sequence[int],
    target_indices: Sequence[int],
) -> Tuple[float, float]:
    """Return the nearest visible target as Minecraft yaw and pitch."""

    yaw, pitch = native.acquire_target(
        position,
        orientation,
        shape_catalog_version,
        side,
        shape_ids,
        target_indices,
    )
    return float(yaw), float(pitch)
