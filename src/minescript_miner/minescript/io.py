"""Minescript-facing IO helpers.

This module is the only layer in the package that should talk to the
`minescript` runtime directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, FrozenSet, List, Optional, Tuple, Union

from minescript_miner.adapter.native_bridge import Orientation, ScanPosition, acquire_target
from minescript_miner.adapter.shape_catalog import BlockShapeCatalog, DEFAULT_CATALOG
from minescript_miner.minescript.world import AreaTimings, fixed_cube_bounds, get_area


DEFAULT_TARGET_CONFIG = Path("targets.txt")
AIR_BLOCK = "minecraft:air"


@dataclass
class ScanTimings:
    area: AreaTimings = field(default_factory=AreaTimings)
    target_config_ms: float = 0.0
    target_match_ms: float = 0.0
    shape_encode_ms: float = 0.0
    native_call_ms: float = 0.0
    total_ms: float = 0.0


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def load_target_blocks(path: Union[str, Path] = DEFAULT_TARGET_CONFIG) -> FrozenSet[str]:
    target_path = Path(path)
    if not target_path.exists():
        return frozenset()

    blocks = []
    for line in target_path.read_text(encoding="utf-8").splitlines():
        block_id = line.split("#", 1)[0].strip().lower()
        if block_id:
            blocks.append(block_id)
    return frozenset(blocks)


def block_id_literal(block_string: Optional[str]) -> str:
    if block_string is None:
        return AIR_BLOCK

    raw = block_string.strip().lower()
    if not raw:
        return AIR_BLOCK
    return raw.split("[", 1)[0].strip()


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
    match_start = time.perf_counter_ns() if timings is not None else 0
    block_strings: List[Optional[str]] = []
    target_indices: List[int] = []
    for index, (_pos, block_string) in enumerate(area):
        block_strings.append(block_string)
        if block_id_literal(block_string) in target_blocks:
            target_indices.append(index)
    if timings is not None:
        timings.target_match_ms = elapsed_ms(match_start)

    side = max_pos[0] - min_pos[0] + 1
    if (
        max_pos[1] - min_pos[1] + 1 != side
        or max_pos[2] - min_pos[2] + 1 != side
    ):
        raise ValueError(f"Expected a cube region, got min={min_pos} max={max_pos}")

    encode_start = time.perf_counter_ns() if timings is not None else 0
    encoded = catalog.encode_region(side, block_strings)
    if timings is not None:
        timings.shape_encode_ms = elapsed_ms(encode_start)
    if not target_indices:
        if timings is not None:
            timings.total_ms = elapsed_ms(total_start)
        return None

    native_start = time.perf_counter_ns() if timings is not None else 0
    result = acquire_target(
        position,
        orientation,
        encoded.shape_catalog_version,
        encoded.side,
        reach,
        encoded.shape_ids,
        target_indices,
    )
    if timings is not None:
        timings.native_call_ms = elapsed_ms(native_start)
        timings.total_ms = elapsed_ms(total_start)
    return result
