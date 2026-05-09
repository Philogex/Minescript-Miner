"""Block-state string to compact native ids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


BlockStateKey = Tuple[Tuple[str, str], ...]

TRANSPARENT_BLOCKS = frozenset(
    {
        "minecraft:air",
        "minecraft:cave_air",
        "minecraft:void_air",
        "minecraft:water",
    }
)

DEFAULT_TARGET_BLOCKS = frozenset(
    {
        "minecraft:diamond_ore",
        "minecraft:deepslate_diamond_ore",
    }
)


@dataclass(frozen=True)
class EncodedBlockRegion:
    side: int
    type_ids: List[int]
    state_ids: List[int]


def parse_block_state(block_string: Optional[str]) -> Tuple[str, BlockStateKey]:
    if block_string is None:
        return "minecraft:air", ()

    raw = block_string.strip().lower()
    if not raw:
        return "minecraft:air", ()

    if "[" not in raw or not raw.endswith("]"):
        return raw, ()

    block_type, properties_raw = raw.split("[", 1)
    properties_raw = properties_raw[:-1]
    properties = []
    for item in properties_raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            properties.append((key, value))

    return block_type.strip(), tuple(sorted(properties))


@dataclass
class BlockIdCatalog:
    type_to_id: Dict[str, int] = field(default_factory=lambda: {"minecraft:air": 0})
    id_to_type: Dict[int, str] = field(default_factory=lambda: {0: "minecraft:air"})
    state_to_id: Dict[BlockStateKey, int] = field(default_factory=lambda: {(): 0})
    id_to_state: Dict[int, BlockStateKey] = field(default_factory=lambda: {0: ()})
    transparent_blocks: frozenset[str] = TRANSPARENT_BLOCKS

    def type_id(self, block_type: str) -> int:
        if block_type in self.transparent_blocks:
            block_type = "minecraft:air"

        existing = self.type_to_id.get(block_type)
        if existing is not None:
            return existing

        new_id = len(self.type_to_id)
        self.type_to_id[block_type] = new_id
        self.id_to_type[new_id] = block_type
        return new_id

    def state_id(self, state: BlockStateKey) -> int:
        existing = self.state_to_id.get(state)
        if existing is not None:
            return existing

        new_id = len(self.state_to_id)
        self.state_to_id[state] = new_id
        self.id_to_state[new_id] = state
        return new_id

    def encode_block(self, block_string: Optional[str]) -> Tuple[int, int]:
        block_type, state = parse_block_state(block_string)
        type_id = self.type_id(block_type)
        if type_id == 0:
            return 0, 0
        return type_id, self.state_id(state)

    def encode_region(
        self,
        side: int,
        block_strings: Sequence[Optional[str]],
    ) -> EncodedBlockRegion:
        expected_count = side * side * side
        if expected_count != len(block_strings):
            raise ValueError(
                f"Expected {expected_count} block strings for side={side}, "
                f"got {len(block_strings)}"
            )

        type_ids: List[int] = []
        state_ids: List[int] = []
        for block_string in block_strings:
            type_id, state_id = self.encode_block(block_string)
            type_ids.append(type_id)
            state_ids.append(state_id)

        return EncodedBlockRegion(side=side, type_ids=type_ids, state_ids=state_ids)


DEFAULT_CATALOG = BlockIdCatalog()
