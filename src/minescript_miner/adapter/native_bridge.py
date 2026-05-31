"""Pure Python adapter between raw Minescript block data and the native module."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import _minescript_miner_native as native

from .shape_catalog import BlockShapeCatalog, DEFAULT_CATALOG


ScanPosition = Tuple[float, float, float]
Orientation = Tuple[float, float]


def scan_region_debug(
    position: ScanPosition,
    orientation: Orientation,
    side: int,
    block_strings: Sequence[Optional[str]],
    *,
    catalog: BlockShapeCatalog = DEFAULT_CATALOG,
) -> Tuple[float, float]:
    """Encode raw block strings and run the native transfer/logging prototype."""

    encoded = catalog.encode_region(side, block_strings)
    x, z = native.scan_region_debug(
        position,
        orientation,
        encoded.catalog_version,
        encoded.side,
        encoded.shape_ids,
    )
    return float(x), float(z)
