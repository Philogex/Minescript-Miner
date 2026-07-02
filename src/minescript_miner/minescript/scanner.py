"""Minescript world acquisition wrapper for the native scanner pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Optional, Union

from minescript_miner.adapter.native_bridge import Orientation, ScanPosition
from minescript_miner.adapter.shape_catalog import BlockShapeCatalog, DEFAULT_CATALOG
from minescript_miner.adapter.target_pipeline import (
    DEFAULT_TARGET_CONFIG,
    acquire_target_from_area,
    elapsed_ms,
    load_target_blocks,
)
from minescript_miner.minescript.world import AreaTimings, fixed_cube_bounds, get_area


@dataclass
class ScanTimings:
    area: AreaTimings = field(default_factory=AreaTimings)
    target_config_ms: float = 0.0
    target_match_ms: float = 0.0
    shape_encode_ms: float = 0.0
    native_call_ms: float = 0.0
    total_ms: float = 0.0


def acquire_current_target(
    position: ScanPosition,
    orientation: Orientation,
    reach: float = 4.8,
    *,
    catalog: BlockShapeCatalog = DEFAULT_CATALOG,
    target_config: Union[str, Path] = DEFAULT_TARGET_CONFIG,
    target_blocks: Optional[AbstractSet[str]] = None,
    await_region: bool = True,
    timings: Optional[ScanTimings] = None,
) -> Optional[Orientation]:
    total_start = time.perf_counter_ns() if timings is not None else 0
    min_pos, max_pos = fixed_cube_bounds(position, reach)
    if target_blocks is None:
        config_start = time.perf_counter_ns() if timings is not None else 0
        target_blocks = load_target_blocks(target_config)
        if timings is not None:
            timings.target_config_ms = elapsed_ms(config_start)

    if timings is None:
        area = get_area(position, reach, await_region=await_region)
    else:
        area = get_area(
            position,
            reach,
            await_region=await_region,
            timings=timings.area,
        )

    result = acquire_target_from_area(
        position,
        orientation,
        reach,
        min_pos=min_pos,
        max_pos=max_pos,
        area=area,
        target_blocks=target_blocks,
        catalog=catalog,
        timings=timings,
    )
    if timings is not None:
        timings.total_ms = elapsed_ms(total_start)
    return result
