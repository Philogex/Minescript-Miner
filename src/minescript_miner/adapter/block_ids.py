"""Block-state string to native prototype id mapping."""

from __future__ import annotations

from typing import List, Optional, Sequence


BLOCK_ID_TRANSPARENT = 0
BLOCK_ID_FULL = 1
BLOCK_ID_STAIRS = 2
BLOCK_ID_SLAB = 3
BLOCK_ID_PANE = 4
BLOCK_ID_TARGET = 5

DEFAULT_TARGET_BLOCKS = frozenset(
    {
        "minecraft:diamond_ore",
        "minecraft:deepslate_diamond_ore",
    }
)


def block_string_to_native_id(
    block_string: Optional[str],
    target_blocks: Sequence[str] = tuple(DEFAULT_TARGET_BLOCKS),
) -> int:
    if block_string is None:
        return BLOCK_ID_TRANSPARENT

    base = block_string.split("[", 1)[0].strip().lower()
    if not base:
        return BLOCK_ID_TRANSPARENT

    if base in target_blocks:
        return BLOCK_ID_TARGET
    if base.endswith(":air") or base == "minecraft:air":
        return BLOCK_ID_TRANSPARENT
    if base.endswith(":water") or base == "minecraft:water":
        return BLOCK_ID_TRANSPARENT
    if base.endswith("_stairs") or "stairs" in base or ":stair" in base:
        return BLOCK_ID_STAIRS
    if base.endswith("_slab") or ":slab" in base:
        return BLOCK_ID_SLAB
    if base.endswith("_pane") or "pane" in base:
        return BLOCK_ID_PANE
    return BLOCK_ID_FULL


def block_strings_to_native_ids(
    block_strings: Sequence[Optional[str]],
    target_blocks: Sequence[str] = tuple(DEFAULT_TARGET_BLOCKS),
) -> List[int]:
    return [
        block_string_to_native_id(block_string, target_blocks)
        for block_string in block_strings
    ]
