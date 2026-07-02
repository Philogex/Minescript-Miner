"""Preparation helpers for native target acquisition."""

from __future__ import annotations

import time
from array import array
from pathlib import Path
from typing import AbstractSet, Optional, Sequence, Tuple, Union

from minescript_miner.adapter.native_bridge import Orientation, ScanPosition, acquire_target
from minescript_miner.adapter.shape_catalog import BlockShapeCatalog, DEFAULT_CATALOG


BlockPos = Tuple[int, int, int]
BlockSample = Tuple[BlockPos, Optional[str]]
DEFAULT_TARGET_CONFIG = Path("targets.txt")
AIR_BLOCK = "minecraft:air"


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def load_target_blocks(path: Union[str, Path] = DEFAULT_TARGET_CONFIG) -> frozenset[str]:
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


def acquire_target_from_area(
    position: ScanPosition,
    orientation: Orientation,
    reach: float,
    *,
    min_pos: BlockPos,
    max_pos: BlockPos,
    area: Sequence[BlockSample],
    target_blocks: AbstractSet[str],
    catalog: BlockShapeCatalog = DEFAULT_CATALOG,
    timings=None,
) -> Optional[Orientation]:
    match_start = time.perf_counter_ns() if timings is not None else 0
    block_strings: list[Optional[str]] = []
    target_index_values: list[int] = []
    for index, (_pos, block_string) in enumerate(area):
        block_strings.append(block_string)
        if block_id_literal(block_string) in target_blocks:
            target_index_values.append(index)
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
    if not target_index_values:
        return None

    target_indices = array("H", target_index_values)
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
    return result
