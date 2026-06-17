"""Pure Python adapter between encoded Python data and the native module."""

from __future__ import annotations

from array import array
from typing import Optional, Sequence, Tuple

import _minescript_miner_native as native


ScanPosition = Tuple[float, float, float]
Orientation = Tuple[float, float]


def _uint16_payload(values: Sequence[int]):
    if isinstance(values, array):
        if values.typecode != "H":
            raise TypeError(
                f"Expected array('H') for compact uint16 payload, "
                f"got array({values.typecode!r})"
            )
        return values.tobytes()
    if isinstance(values, (bytes, bytearray)):
        return bytes(values)
    return values


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
        _uint16_payload(shape_ids),
        _uint16_payload(target_indices),
    )
    if result is None:
        return None

    yaw, pitch = result
    return float(yaw), float(pitch)
