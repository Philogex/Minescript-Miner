"""Minescript-facing IO helpers.

This module is the only layer in the package that should talk to the
`minescript` runtime directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple, Union

from minescript_miner.adapter.native_bridge import Orientation, ScanPosition, acquire_target
from minescript_miner.adapter.shape_catalog import BlockShapeCatalog, DEFAULT_CATALOG
from minescript_miner.minescript.world import fixed_cube_bounds, get_area


DEFAULT_TARGET_CONFIG = Path("targets.txt")
AIR_BLOCK = "minecraft:air"


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
) -> Tuple[float, float]:
    min_pos, max_pos = fixed_cube_bounds(position, reach)
    target_blocks = load_target_blocks(target_config)
    block_strings: List[Optional[str]] = []
    target_indices: List[int] = []
    for index, (_pos, block_string) in enumerate(get_area(position, reach)):
        block_strings.append(block_string)
        if block_id_literal(block_string) in target_blocks:
            target_indices.append(index)

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
        target_indices,
    )
