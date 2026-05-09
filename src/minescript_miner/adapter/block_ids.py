"""Conservative block-state string to stable native shape-id mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


CATALOG_VERSION = 1

DIRECTIONS = ("north", "east", "south", "west")
HALVES = ("bottom", "top")
STAIR_SHAPES = ("straight", "inner_left", "inner_right", "outer_left", "outer_right")

EMPTY_BLOCKS = frozenset(
    {
        "minecraft:air",
        "minecraft:cave_air",
        "minecraft:void_air",
        "minecraft:water",
        "minecraft:lava",
    }
)

BlockStateKey = Tuple[Tuple[str, str], ...]
BlockShapeKey = Tuple[str, BlockStateKey]


@dataclass(frozen=True)
class EncodedBlockRegion:
    catalog_version: int
    side: int
    shape_ids: List[int]


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


def _register_shape(name: str) -> int:
    shape_id = len(SHAPE_NAMES)
    SHAPE_NAMES.append(name)
    SHAPE_ID_BY_NAME[name] = shape_id
    return shape_id


def _state_key(*properties: Tuple[str, str]) -> BlockStateKey:
    return tuple(sorted(properties))


def _connection_state_key(mask: int) -> BlockStateKey:
    return _state_key(
        *(
            (direction, "true" if (mask & (1 << bit)) else "false")
            for bit, direction in enumerate(DIRECTIONS)
        )
    )


def _connection_name(mask: int) -> str:
    parts = [
        direction
        for bit, direction in enumerate(DIRECTIONS)
        if mask & (1 << bit)
    ]
    return "_".join(parts) if parts else "none"


SHAPE_NAMES: List[str] = []
SHAPE_ID_BY_NAME: Dict[str, int] = {}
SHAPE_ID_BY_BLOCK_STATE: Dict[BlockShapeKey, int] = {}

SHAPE_EMPTY = _register_shape("empty")
SHAPE_FULL_CUBE = _register_shape("full_cube")
SHAPE_OAK_SLAB_BOTTOM = _register_shape("oak_slab_bottom")
SHAPE_OAK_SLAB_TOP = _register_shape("oak_slab_top")

SHAPE_ID_BY_BLOCK_STATE[("minecraft:oak_slab", _state_key(("type", "bottom")))] = (
    SHAPE_OAK_SLAB_BOTTOM
)
SHAPE_ID_BY_BLOCK_STATE[("minecraft:oak_slab", _state_key(("type", "top")))] = (
    SHAPE_OAK_SLAB_TOP
)
SHAPE_ID_BY_BLOCK_STATE[("minecraft:oak_slab", _state_key(("type", "double")))] = (
    SHAPE_FULL_CUBE
)

for facing in DIRECTIONS:
    for half in HALVES:
        for stair_shape in STAIR_SHAPES:
            shape_id = _register_shape(f"oak_stairs_{facing}_{half}_{stair_shape}")
            SHAPE_ID_BY_BLOCK_STATE[
                (
                    "minecraft:oak_stairs",
                    _state_key(
                        ("facing", facing),
                        ("half", half),
                        ("shape", stair_shape),
                    ),
                )
            ] = shape_id

for mask in range(16):
    shape_id = _register_shape(f"pane_{_connection_name(mask)}")
    state_key = _connection_state_key(mask)
    SHAPE_ID_BY_BLOCK_STATE[("minecraft:iron_bars", state_key)] = shape_id
    SHAPE_ID_BY_BLOCK_STATE[("minecraft:glass_pane", state_key)] = shape_id

for mask in range(16):
    shape_id = _register_shape(f"oak_fence_{_connection_name(mask)}")
    SHAPE_ID_BY_BLOCK_STATE[("minecraft:oak_fence", _connection_state_key(mask))] = (
        shape_id
    )

SHAPE_COUNT = len(SHAPE_NAMES)

RELEVANT_PROPERTIES: Dict[str, Tuple[str, ...]] = {
    "minecraft:oak_slab": ("type",),
    "minecraft:oak_stairs": ("facing", "half", "shape"),
    "minecraft:iron_bars": DIRECTIONS,
    "minecraft:glass_pane": DIRECTIONS,
    "minecraft:oak_fence": DIRECTIONS,
}


def normalized_shape_key(block_type: str, state: BlockStateKey) -> Optional[BlockShapeKey]:
    relevant = RELEVANT_PROPERTIES.get(block_type)
    if relevant is None:
        return None

    properties = dict(state)
    if any(key not in properties for key in relevant):
        return None

    return block_type, _state_key(*((key, properties[key]) for key in relevant))


@dataclass
class BlockShapeCatalog:
    catalog_version: int = CATALOG_VERSION
    cache: Dict[Optional[str], int] = field(default_factory=dict)

    def shape_id(self, block_string: Optional[str]) -> int:
        if block_string in self.cache:
            return self.cache[block_string]

        shape_id = self._shape_id_uncached(block_string)
        self.cache[block_string] = shape_id
        return shape_id

    def _shape_id_uncached(self, block_string: Optional[str]) -> int:
        block_type, state = parse_block_state(block_string)
        if block_type in EMPTY_BLOCKS:
            return SHAPE_EMPTY

        key = normalized_shape_key(block_type, state)
        if key is None:
            return SHAPE_FULL_CUBE

        return SHAPE_ID_BY_BLOCK_STATE.get(key, SHAPE_FULL_CUBE)

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

        return EncodedBlockRegion(
            catalog_version=self.catalog_version,
            side=side,
            shape_ids=[self.shape_id(block_string) for block_string in block_strings],
        )


DEFAULT_CATALOG = BlockShapeCatalog()
