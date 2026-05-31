#!/usr/bin/env python3
"""Generate Python and C++ shape/geometry catalog files from JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "catalog" / "shape_catalog.json"
PYTHON_TARGET = ROOT / "src" / "minescript_miner" / "adapter" / "shape_catalog.py"
CPP_HEADER_TARGET = ROOT / "native" / "include" / "minescript_miner" / "geometry_catalog.hpp"
CPP_SOURCE_TARGET = ROOT / "native" / "src" / "geometry_catalog.cpp"

Box = Tuple[int, int, int, int, int, int]
Face = Tuple[str, int, int, int, int, int, int]


@dataclass(frozen=True)
class Shape:
    name: str
    boxes: Tuple[Box, ...]


def load_catalog() -> Dict[str, Any]:
    with SOURCE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def connection_name(mask: int, directions: Sequence[str]) -> str:
    parts = [direction for bit, direction in enumerate(directions) if mask & (1 << bit)]
    return "_".join(parts) if parts else "none"


def connection_state_key(mask: int, directions: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(
        sorted(
            (direction, "true" if mask & (1 << bit) else "false")
            for bit, direction in enumerate(directions)
        )
    )


def stair_quadrant(direction: str, side: str, front: bool, y_min: int, y_max: int) -> Box:
    front_x = half_bounds_1d(direction, front, True)
    front_z = half_bounds_1d(direction, front, False)
    lateral = lateral_direction(direction, side)
    lateral_x = half_bounds_1d(lateral, True, True)
    lateral_z = half_bounds_1d(lateral, True, False)
    return (
        max(front_x[0], lateral_x[0]),
        y_min,
        max(front_z[0], lateral_z[0]),
        min(front_x[1], lateral_x[1]),
        y_max,
        min(front_z[1], lateral_z[1]),
    )


def half_bounds_1d(direction: str, front: bool, x_axis: bool) -> Tuple[int, int]:
    if direction == "north":
        return (0, 16) if x_axis else ((0, 8) if front else (8, 16))
    if direction == "south":
        return (0, 16) if x_axis else ((8, 16) if front else (0, 8))
    if direction == "east":
        return ((8, 16) if front else (0, 8)) if x_axis else (0, 16)
    if direction == "west":
        return ((0, 8) if front else (8, 16)) if x_axis else (0, 16)
    raise ValueError(f"unknown direction: {direction}")


def lateral_direction(direction: str, side: str) -> str:
    left = {
        "north": "west",
        "east": "north",
        "south": "east",
        "west": "south",
    }
    right = {
        "north": "east",
        "east": "south",
        "south": "west",
        "west": "north",
    }
    return (left if side == "left" else right)[direction]


def stair_boxes(direction: str, half: str, stair_shape: str) -> Tuple[Box, ...]:
    boxes: List[Box] = []
    y_min, y_max = 8, 16
    if half == "bottom":
        boxes.append((0, 0, 0, 16, 8, 16))
    else:
        boxes.append((0, 8, 0, 16, 16, 16))
        y_min, y_max = 0, 8

    quadrants = {
        "straight": (("left", True), ("right", True)),
        "outer_left": (("left", True),),
        "outer_right": (("right", True),),
        "inner_left": (("left", True), ("right", True), ("left", False)),
        "inner_right": (("left", True), ("right", True), ("right", False)),
    }[stair_shape]
    for side, front in quadrants:
        boxes.append(stair_quadrant(direction, side, front, y_min, y_max))
    return tuple(boxes)


def connection_boxes(mask: int, center_min: int, center_max: int) -> Tuple[Box, ...]:
    boxes: List[Box] = [(center_min, 0, center_min, center_max, 16, center_max)]
    if mask & 1:
        boxes.append((center_min, 0, 0, center_max, 16, center_min))
    if mask & 2:
        boxes.append((center_max, 0, center_min, 16, 16, center_max))
    if mask & 4:
        boxes.append((center_min, 0, center_max, center_max, 16, 16))
    if mask & 8:
        boxes.append((0, 0, center_min, center_min, 16, center_max))
    return tuple(boxes)


def expand_shapes(catalog: Dict[str, Any]) -> List[Shape]:
    directions = catalog["directions"]
    halves = catalog["halves"]
    stair_shapes = catalog["stair_shapes"]
    shapes: List[Shape] = []

    for spec in catalog["shapes"]:
        if spec.get("kind") == "empty":
            shapes.append(Shape(spec["name"], ()))
        elif "boxes" in spec:
            shapes.append(Shape(spec["name"], tuple(tuple(box) for box in spec["boxes"])))
        elif spec.get("family") == "stairs":
            for direction in directions:
                for half in halves:
                    for stair_shape in stair_shapes:
                        shapes.append(
                            Shape(
                                spec["template"].format(
                                    facing=direction,
                                    half=half,
                                    shape=stair_shape,
                                ),
                                stair_boxes(direction, half, stair_shape),
                            )
                        )
        elif spec.get("family") == "connection":
            center_min, center_max = spec["center"]
            for mask in range(16):
                shapes.append(
                    Shape(
                        spec["template"].format(connection=connection_name(mask, directions)),
                        connection_boxes(mask, center_min, center_max),
                    )
                )
        else:
            raise ValueError(f"unsupported shape spec: {spec}")

    return shapes


def axis_min(box: Box, axis: str) -> int:
    return box[{"x": 0, "y": 1, "z": 2}[axis]]


def axis_max(box: Box, axis: str) -> int:
    return box[{"x": 3, "y": 4, "z": 5}[axis]]


def uv_axes(axis: str) -> Tuple[str, str]:
    return {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }[axis]


def overlaps_1d(a_min: int, a_max: int, b_min: int, b_max: int) -> bool:
    return a_min < b_max and b_min < a_max


def add_split(values: List[int], value: int, min_value: int, max_value: int) -> None:
    if min_value < value < max_value:
        values.append(value)


def midpoint_inside(min_value: int, max_value: int, midpoint_times_2: int) -> bool:
    return min_value * 2 < midpoint_times_2 < max_value * 2


def outside_occupied(
    face: Face,
    u_midpoint_times_2: int,
    v_midpoint_times_2: int,
    boxes: Sequence[Box],
) -> bool:
    axis, coord, *_unused, normal_sign = face
    u_axis, v_axis = uv_axes(axis)
    for box in boxes:
        if normal_sign > 0:
            crosses_face = axis_min(box, axis) <= coord and axis_max(box, axis) > coord
        else:
            crosses_face = axis_min(box, axis) < coord and axis_max(box, axis) >= coord
        if not crosses_face:
            continue
        if midpoint_inside(axis_min(box, u_axis), axis_max(box, u_axis), u_midpoint_times_2) and midpoint_inside(
            axis_min(box, v_axis),
            axis_max(box, v_axis),
            v_midpoint_times_2,
        ):
            return True
    return False


def face_cells(face: Face, boxes: Sequence[Box]) -> List[Face]:
    axis, coord, u_min, u_max, v_min, v_max, normal_sign = face
    u_axis, v_axis = uv_axes(axis)
    u_splits = [u_min, u_max]
    v_splits = [v_min, v_max]

    for box in boxes:
        if axis_min(box, axis) > coord or axis_max(box, axis) < coord:
            continue

        box_u_min = axis_min(box, u_axis)
        box_u_max = axis_max(box, u_axis)
        box_v_min = axis_min(box, v_axis)
        box_v_max = axis_max(box, v_axis)
        if not overlaps_1d(u_min, u_max, box_u_min, box_u_max) or not overlaps_1d(
            v_min,
            v_max,
            box_v_min,
            box_v_max,
        ):
            continue

        add_split(u_splits, box_u_min, u_min, u_max)
        add_split(u_splits, box_u_max, u_min, u_max)
        add_split(v_splits, box_v_min, v_min, v_max)
        add_split(v_splits, box_v_max, v_min, v_max)

    u_splits = sorted(set(u_splits))
    v_splits = sorted(set(v_splits))
    cells: List[Face] = []
    for u0, u1 in zip(u_splits, u_splits[1:]):
        for v0, v1 in zip(v_splits, v_splits[1:]):
            if u1 <= u0 or v1 <= v0:
                continue
            if outside_occupied(face, u0 + u1, v0 + v1, boxes):
                continue
            cells.append((axis, coord, u0, u1, v0, v1, normal_sign))
    return cells


def faces_for_box(box: Box, boxes: Sequence[Box]) -> List[Face]:
    min_x, min_y, min_z, max_x, max_y, max_z = box
    candidates: List[Face] = [
        ("x", min_x, min_y, max_y, min_z, max_z, -1),
        ("x", max_x, min_y, max_y, min_z, max_z, 1),
        ("y", min_y, min_x, max_x, min_z, max_z, -1),
        ("y", max_y, min_x, max_x, min_z, max_z, 1),
        ("z", min_z, min_x, max_x, min_y, max_y, -1),
        ("z", max_z, min_x, max_x, min_y, max_y, 1),
    ]
    faces: List[Face] = []
    for face in candidates:
        faces.extend(face_cells(face, boxes))
    return faces


def faces_for_shape(shape: Shape) -> List[Face]:
    faces: List[Face] = []
    for box in shape.boxes:
        faces.extend(faces_for_box(box, shape.boxes))
    return faces


def state_key(properties: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(properties.items()))


def expand_block_mappings(catalog: Dict[str, Any], shape_ids: Dict[str, int]) -> Tuple[Dict[Tuple[str, Tuple[Tuple[str, str], ...]], int], Dict[str, Tuple[str, ...]]]:
    directions = catalog["directions"]
    halves = catalog["halves"]
    stair_shapes = catalog["stair_shapes"]
    mapping: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], int] = {}
    relevant: Dict[str, Tuple[str, ...]] = {}

    for spec in catalog["block_mappings"]:
        blocks = spec["blocks"] if "blocks" in spec else [spec["block"]]
        properties = tuple(spec["properties"])
        for block in blocks:
            relevant[block] = properties

        if "states" in spec:
            for block in blocks:
                for state in spec["states"]:
                    mapping[(block, state_key(state["properties"]))] = shape_ids[state["shape"]]
        elif "shape_template" in spec:
            for block in blocks:
                for facing in directions:
                    for half in halves:
                        for stair_shape in stair_shapes:
                            properties_dict = {
                                "facing": facing,
                                "half": half,
                                "shape": stair_shape,
                            }
                            shape_name = spec["shape_template"].format(**properties_dict)
                            mapping[(block, state_key(properties_dict))] = shape_ids[shape_name]
        elif "connection_template" in spec:
            for block in blocks:
                for mask in range(16):
                    properties_dict = {
                        direction: "true" if mask & (1 << bit) else "false"
                        for bit, direction in enumerate(directions)
                    }
                    shape_name = spec["connection_template"].format(connection=connection_name(mask, directions))
                    mapping[(block, state_key(properties_dict))] = shape_ids[shape_name]
        else:
            raise ValueError(f"unsupported block mapping spec: {spec}")

    return mapping, relevant


def py_repr(value: object) -> str:
    return repr(value)


def render_python(catalog: Dict[str, Any], shapes: Sequence[Shape]) -> str:
    shape_ids = {shape.name: index for index, shape in enumerate(shapes)}
    block_mapping, relevant = expand_block_mappings(catalog, shape_ids)
    empty_blocks = sorted(catalog["empty_blocks"])

    mapping_lines = []
    for (block, key), shape_id in sorted(block_mapping.items(), key=lambda item: (item[0][0], item[0][1])):
        mapping_lines.append(f"    ({block!r}, {key!r}): {shape_id},")

    relevant_lines = []
    for block, properties in sorted(relevant.items()):
        relevant_lines.append(f"    {block!r}: {properties!r},")

    shape_names_lines = [f"    {shape.name!r}," for shape in shapes]

    return f'''"""Generated block-state string to stable native shape-id mapping.

Generated by tools/generate_shape_catalog.py from catalog/shape_catalog.json.
Do not edit by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


SHAPE_CATALOG_VERSION = {catalog["shape_catalog_version"]}
BLOCK_SHAPE_MAPPING_VERSION = {catalog["block_shape_mapping_version"]}
CATALOG_VERSION = SHAPE_CATALOG_VERSION

DIRECTIONS = {tuple(catalog["directions"])!r}
HALVES = {tuple(catalog["halves"])!r}
STAIR_SHAPES = {tuple(catalog["stair_shapes"])!r}

EMPTY_BLOCKS = frozenset({empty_blocks!r})

BlockStateKey = Tuple[Tuple[str, str], ...]
BlockShapeKey = Tuple[str, BlockStateKey]


@dataclass(frozen=True)
class EncodedBlockRegion:
    shape_catalog_version: int
    block_shape_mapping_version: int
    side: int
    shape_ids: List[int]

    @property
    def catalog_version(self) -> int:
        return self.shape_catalog_version


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


def _state_key(*properties: Tuple[str, str]) -> BlockStateKey:
    return tuple(sorted(properties))


SHAPE_NAMES: List[str] = [
{chr(10).join(shape_names_lines)}
]
SHAPE_ID_BY_NAME: Dict[str, int] = {{name: shape_id for shape_id, name in enumerate(SHAPE_NAMES)}}
SHAPE_ID_BY_BLOCK_STATE: Dict[BlockShapeKey, int] = {{
{chr(10).join(mapping_lines)}
}}

SHAPE_EMPTY = SHAPE_ID_BY_NAME["empty"]
SHAPE_FULL_CUBE = SHAPE_ID_BY_NAME["full_cube"]
SHAPE_SLAB_BOTTOM = SHAPE_ID_BY_NAME["slab_bottom"]
SHAPE_SLAB_TOP = SHAPE_ID_BY_NAME["slab_top"]
SHAPE_COUNT = len(SHAPE_NAMES)

RELEVANT_PROPERTIES: Dict[str, Tuple[str, ...]] = {{
{chr(10).join(relevant_lines)}
}}


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
    shape_catalog_version: int = SHAPE_CATALOG_VERSION
    block_shape_mapping_version: int = BLOCK_SHAPE_MAPPING_VERSION
    cache: Dict[Optional[str], int] = field(default_factory=dict)

    @property
    def catalog_version(self) -> int:
        return self.shape_catalog_version

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
                f"Expected {{expected_count}} block strings for side={{side}}, "
                f"got {{len(block_strings)}}"
            )

        return EncodedBlockRegion(
            shape_catalog_version=self.shape_catalog_version,
            block_shape_mapping_version=self.block_shape_mapping_version,
            side=side,
            shape_ids=[self.shape_id(block_string) for block_string in block_strings],
        )


DEFAULT_CATALOG = BlockShapeCatalog()
'''


def render_header(catalog: Dict[str, Any], shapes: Sequence[Shape], box_count: int, face_count: int) -> str:
    return f"""// Generated by tools/generate_shape_catalog.py from catalog/shape_catalog.json.
// Do not edit by hand.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace minescript_miner {{

inline constexpr int SHAPE_CATALOG_VERSION = {catalog["shape_catalog_version"]};
inline constexpr int GEOMETRY_CATALOG_VERSION = {catalog["geometry_catalog_version"]};
inline constexpr int GEOMETRY_SHAPE_CATALOG_VERSION = SHAPE_CATALOG_VERSION;
inline constexpr std::size_t GEOMETRY_SHAPE_COUNT = {len(shapes)};
inline constexpr std::size_t GEOMETRY_BOX_COUNT = {box_count};
inline constexpr std::size_t GEOMETRY_FACE_COUNT = {face_count};

inline constexpr std::int32_t SHAPE_EMPTY = 0;
inline constexpr std::int32_t SHAPE_FULL_CUBE = 1;
inline constexpr std::int32_t SHAPE_SLAB_BOTTOM = 2;
inline constexpr std::int32_t SHAPE_SLAB_TOP = 3;

enum class PlaneAxis : std::int32_t {{
    X = 0,
    Y = 1,
    Z = 2,
}};

struct Aabb16 {{
    std::uint8_t min_x;
    std::uint8_t min_y;
    std::uint8_t min_z;
    std::uint8_t max_x;
    std::uint8_t max_y;
    std::uint8_t max_z;
}};

struct RectFace16 {{
    PlaneAxis axis;
    std::uint8_t coord;
    std::uint8_t u_min;
    std::uint8_t u_max;
    std::uint8_t v_min;
    std::uint8_t v_max;
    std::int8_t normal_sign;
}};

struct ShapeGeometry {{
    std::uint16_t box_offset;
    std::uint8_t box_count;
    std::uint16_t face_offset;
    std::uint8_t face_count;
}};

struct GeometryCatalog {{
    std::array<const char *, GEOMETRY_SHAPE_COUNT> shape_names;
    std::array<ShapeGeometry, GEOMETRY_SHAPE_COUNT> shapes;
    std::array<Aabb16, GEOMETRY_BOX_COUNT> boxes;
    std::array<RectFace16, GEOMETRY_FACE_COUNT> faces;
}};

const GeometryCatalog &geometry_catalog();
const ShapeGeometry &geometry_for_shape(std::int32_t shape_id);
std::int32_t geometry_catalog_shape_count();

const char *shape_id_name(std::int32_t shape_id);
std::int32_t shape_count();
const std::array<const char *, GEOMETRY_SHAPE_COUNT> &shape_names();

inline bool is_empty_shape(std::int32_t shape_id) {{
    return shape_id == SHAPE_EMPTY;
}}

}}
"""


def render_cpp(catalog: Dict[str, Any], shapes: Sequence[Shape]) -> str:
    shape_names = ",\n".join(f'    "{shape.name}"' for shape in shapes)
    directions = catalog["directions"]
    halves = catalog["halves"]
    stair_shapes = catalog["stair_shapes"]
    pane_center = next(spec["center"] for spec in catalog["shapes"] if spec.get("template") == "pane_{connection}")
    fence_center = next(spec["center"] for spec in catalog["shapes"] if spec.get("template") == "fence_{connection}")

    return f'''// Generated by tools/generate_shape_catalog.py from catalog/shape_catalog.json.
// Do not edit by hand.

#include "minescript_miner/geometry_catalog.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace minescript_miner {{

namespace {{

constexpr std::array<const char *, GEOMETRY_SHAPE_COUNT> SHAPE_NAME_TABLE = {{
{shape_names}
}};

enum Direction : std::uint8_t {{
    NORTH = 0,
    EAST = 1,
    SOUTH = 2,
    WEST = 3,
}};

enum StairShape : std::uint8_t {{
    STRAIGHT = 0,
    INNER_LEFT = 1,
    INNER_RIGHT = 2,
    OUTER_LEFT = 3,
    OUTER_RIGHT = 4,
}};

struct Range16 {{
    std::uint8_t min;
    std::uint8_t max;
}};

struct AxisPair {{
    PlaneAxis first;
    PlaneAxis second;
}};

struct BoxList {{
    std::array<Aabb16, 5> boxes{{}};
    std::uint8_t count = 0;
}};

struct SplitList {{
    std::array<std::uint8_t, 12> values{{}};
    std::uint8_t count = 0;
}};

struct CatalogBuilder {{
    GeometryCatalog catalog{{}};
    std::uint16_t shape_count = 0;
    std::uint16_t box_count = 0;
    std::uint16_t face_count = 0;
}};

constexpr Aabb16 box(
    std::uint8_t min_x,
    std::uint8_t min_y,
    std::uint8_t min_z,
    std::uint8_t max_x,
    std::uint8_t max_y,
    std::uint8_t max_z
) {{
    return {{min_x, min_y, min_z, max_x, max_y, max_z}};
}}

constexpr std::uint8_t min_u8(std::uint8_t a, std::uint8_t b) {{
    return a < b ? a : b;
}}

constexpr std::uint8_t max_u8(std::uint8_t a, std::uint8_t b) {{
    return a > b ? a : b;
}}

constexpr std::uint8_t axis_min(const Aabb16 &b, PlaneAxis axis) {{
    switch (axis) {{
        case PlaneAxis::X:
            return b.min_x;
        case PlaneAxis::Y:
            return b.min_y;
        case PlaneAxis::Z:
            return b.min_z;
    }}
    return 0;
}}

constexpr std::uint8_t axis_max(const Aabb16 &b, PlaneAxis axis) {{
    switch (axis) {{
        case PlaneAxis::X:
            return b.max_x;
        case PlaneAxis::Y:
            return b.max_y;
        case PlaneAxis::Z:
            return b.max_z;
    }}
    return 0;
}}

constexpr AxisPair uv_axes(PlaneAxis axis) {{
    switch (axis) {{
        case PlaneAxis::X:
            return {{PlaneAxis::Y, PlaneAxis::Z}};
        case PlaneAxis::Y:
            return {{PlaneAxis::X, PlaneAxis::Z}};
        case PlaneAxis::Z:
            return {{PlaneAxis::X, PlaneAxis::Y}};
    }}
    return {{PlaneAxis::X, PlaneAxis::Y}};
}}

constexpr bool overlaps_1d(std::uint8_t a_min, std::uint8_t a_max, std::uint8_t b_min, std::uint8_t b_max) {{
    return a_min < b_max && b_min < a_max;
}}

constexpr void add_box(BoxList &shape, Aabb16 b) {{
    shape.boxes[shape.count] = b;
    ++shape.count;
}}

constexpr void add_split(SplitList &splits, std::uint8_t value, std::uint8_t min_value, std::uint8_t max_value) {{
    if (value > min_value && value < max_value) {{
        splits.values[splits.count] = value;
        ++splits.count;
    }}
}}

constexpr void sort_unique(SplitList &splits) {{
    for (std::uint8_t i = 1; i < splits.count; ++i) {{
        const std::uint8_t value = splits.values[i];
        std::uint8_t j = i;
        while (j > 0 && splits.values[j - 1] > value) {{
            splits.values[j] = splits.values[j - 1];
            --j;
        }}
        splits.values[j] = value;
    }}

    std::uint8_t write_index = 0;
    for (std::uint8_t read_index = 0; read_index < splits.count; ++read_index) {{
        if (write_index == 0 || splits.values[read_index] != splits.values[write_index - 1]) {{
            splits.values[write_index] = splits.values[read_index];
            ++write_index;
        }}
    }}
    splits.count = write_index;
}}

constexpr bool midpoint_inside(std::uint8_t min_value, std::uint8_t max_value, std::uint16_t midpoint_times_2) {{
    return static_cast<std::uint16_t>(min_value) * 2 < midpoint_times_2 &&
           midpoint_times_2 < static_cast<std::uint16_t>(max_value) * 2;
}}

constexpr bool outside_occupied(
    const RectFace16 &face,
    std::uint16_t u_midpoint_times_2,
    std::uint16_t v_midpoint_times_2,
    const BoxList &shape
) {{
    const AxisPair axes = uv_axes(face.axis);
    for (std::uint8_t i = 0; i < shape.count; ++i) {{
        const Aabb16 &b = shape.boxes[i];
        const bool crosses_face =
            face.normal_sign > 0
                ? axis_min(b, face.axis) <= face.coord && axis_max(b, face.axis) > face.coord
                : axis_min(b, face.axis) < face.coord && axis_max(b, face.axis) >= face.coord;
        if (!crosses_face) {{
            continue;
        }}

        if (midpoint_inside(axis_min(b, axes.first), axis_max(b, axes.first), u_midpoint_times_2) &&
            midpoint_inside(axis_min(b, axes.second), axis_max(b, axes.second), v_midpoint_times_2)) {{
            return true;
        }}
    }}
    return false;
}}

constexpr void add_face(CatalogBuilder &builder, RectFace16 face) {{
    builder.catalog.faces[builder.face_count] = face;
    ++builder.face_count;
}}

constexpr void add_face_cells(CatalogBuilder &builder, const RectFace16 &face, const BoxList &shape) {{
    const AxisPair axes = uv_axes(face.axis);
    SplitList u_splits{{}};
    SplitList v_splits{{}};
    u_splits.values[u_splits.count++] = face.u_min;
    u_splits.values[u_splits.count++] = face.u_max;
    v_splits.values[v_splits.count++] = face.v_min;
    v_splits.values[v_splits.count++] = face.v_max;

    for (std::uint8_t i = 0; i < shape.count; ++i) {{
        const Aabb16 &b = shape.boxes[i];
        if (axis_min(b, face.axis) > face.coord || axis_max(b, face.axis) < face.coord) {{
            continue;
        }}

        const std::uint8_t box_u_min = axis_min(b, axes.first);
        const std::uint8_t box_u_max = axis_max(b, axes.first);
        const std::uint8_t box_v_min = axis_min(b, axes.second);
        const std::uint8_t box_v_max = axis_max(b, axes.second);
        if (!overlaps_1d(face.u_min, face.u_max, box_u_min, box_u_max) ||
            !overlaps_1d(face.v_min, face.v_max, box_v_min, box_v_max)) {{
            continue;
        }}

        add_split(u_splits, box_u_min, face.u_min, face.u_max);
        add_split(u_splits, box_u_max, face.u_min, face.u_max);
        add_split(v_splits, box_v_min, face.v_min, face.v_max);
        add_split(v_splits, box_v_max, face.v_min, face.v_max);
    }}

    sort_unique(u_splits);
    sort_unique(v_splits);

    for (std::uint8_t u_index = 0; u_index + 1 < u_splits.count; ++u_index) {{
        for (std::uint8_t v_index = 0; v_index + 1 < v_splits.count; ++v_index) {{
            const std::uint8_t u_min = u_splits.values[u_index];
            const std::uint8_t u_max = u_splits.values[u_index + 1];
            const std::uint8_t v_min = v_splits.values[v_index];
            const std::uint8_t v_max = v_splits.values[v_index + 1];
            if (u_max <= u_min || v_max <= v_min) {{
                continue;
            }}

            const std::uint16_t u_midpoint_times_2 = static_cast<std::uint16_t>(u_min) + u_max;
            const std::uint16_t v_midpoint_times_2 = static_cast<std::uint16_t>(v_min) + v_max;
            if (outside_occupied(face, u_midpoint_times_2, v_midpoint_times_2, shape)) {{
                continue;
            }}

            add_face(builder, {{face.axis, face.coord, u_min, u_max, v_min, v_max, face.normal_sign}});
        }}
    }}
}}

constexpr void add_box_faces(CatalogBuilder &builder, const Aabb16 &b, const BoxList &shape) {{
    add_face_cells(builder, {{PlaneAxis::X, b.min_x, b.min_y, b.max_y, b.min_z, b.max_z, -1}}, shape);
    add_face_cells(builder, {{PlaneAxis::X, b.max_x, b.min_y, b.max_y, b.min_z, b.max_z, 1}}, shape);
    add_face_cells(builder, {{PlaneAxis::Y, b.min_y, b.min_x, b.max_x, b.min_z, b.max_z, -1}}, shape);
    add_face_cells(builder, {{PlaneAxis::Y, b.max_y, b.min_x, b.max_x, b.min_z, b.max_z, 1}}, shape);
    add_face_cells(builder, {{PlaneAxis::Z, b.min_z, b.min_x, b.max_x, b.min_y, b.max_y, -1}}, shape);
    add_face_cells(builder, {{PlaneAxis::Z, b.max_z, b.min_x, b.max_x, b.min_y, b.max_y, 1}}, shape);
}}

constexpr void add_shape(CatalogBuilder &builder, const BoxList &shape) {{
    const std::uint16_t box_offset = builder.box_count;
    const std::uint16_t face_offset = builder.face_count;

    for (std::uint8_t i = 0; i < shape.count; ++i) {{
        builder.catalog.boxes[builder.box_count] = shape.boxes[i];
        ++builder.box_count;
    }}
    for (std::uint8_t i = 0; i < shape.count; ++i) {{
        add_box_faces(builder, shape.boxes[i], shape);
    }}

    builder.catalog.shapes[builder.shape_count] = {{
        box_offset,
        shape.count,
        face_offset,
        static_cast<std::uint8_t>(builder.face_count - face_offset),
    }};
    ++builder.shape_count;
}}

constexpr Range16 half_bounds_1d(Direction direction, bool front, bool x_axis) {{
    switch (direction) {{
        case NORTH:
            return x_axis ? Range16{{0, 16}} : (front ? Range16{{0, 8}} : Range16{{8, 16}});
        case SOUTH:
            return x_axis ? Range16{{0, 16}} : (front ? Range16{{8, 16}} : Range16{{0, 8}});
        case EAST:
            return x_axis ? (front ? Range16{{8, 16}} : Range16{{0, 8}}) : Range16{{0, 16}};
        case WEST:
            return x_axis ? (front ? Range16{{0, 8}} : Range16{{8, 16}}) : Range16{{0, 16}};
    }}
    return {{0, 16}};
}}

constexpr Direction lateral_direction(Direction direction, bool left) {{
    if (left) {{
        switch (direction) {{
            case NORTH:
                return WEST;
            case EAST:
                return NORTH;
            case SOUTH:
                return EAST;
            case WEST:
                return SOUTH;
        }}
    }}

    switch (direction) {{
        case NORTH:
            return EAST;
        case EAST:
            return SOUTH;
        case SOUTH:
            return WEST;
        case WEST:
            return NORTH;
    }}
    return NORTH;
}}

constexpr Aabb16 stair_quadrant(Direction direction, bool left, bool front, std::uint8_t y_min, std::uint8_t y_max) {{
    const Range16 front_x = half_bounds_1d(direction, front, true);
    const Range16 front_z = half_bounds_1d(direction, front, false);
    const Direction lateral = lateral_direction(direction, left);
    const Range16 lateral_x = half_bounds_1d(lateral, true, true);
    const Range16 lateral_z = half_bounds_1d(lateral, true, false);
    return box(
        max_u8(front_x.min, lateral_x.min),
        y_min,
        max_u8(front_z.min, lateral_z.min),
        min_u8(front_x.max, lateral_x.max),
        y_max,
        min_u8(front_z.max, lateral_z.max)
    );
}}

constexpr BoxList stair_boxes(Direction direction, bool top, StairShape stair_shape) {{
    BoxList shape{{}};
    std::uint8_t y_min = 8;
    std::uint8_t y_max = 16;
    if (top) {{
        add_box(shape, box(0, 8, 0, 16, 16, 16));
        y_min = 0;
        y_max = 8;
    }} else {{
        add_box(shape, box(0, 0, 0, 16, 8, 16));
    }}

    switch (stair_shape) {{
        case STRAIGHT:
            add_box(shape, stair_quadrant(direction, true, true, y_min, y_max));
            add_box(shape, stair_quadrant(direction, false, true, y_min, y_max));
            break;
        case INNER_LEFT:
            add_box(shape, stair_quadrant(direction, true, true, y_min, y_max));
            add_box(shape, stair_quadrant(direction, false, true, y_min, y_max));
            add_box(shape, stair_quadrant(direction, true, false, y_min, y_max));
            break;
        case INNER_RIGHT:
            add_box(shape, stair_quadrant(direction, true, true, y_min, y_max));
            add_box(shape, stair_quadrant(direction, false, true, y_min, y_max));
            add_box(shape, stair_quadrant(direction, false, false, y_min, y_max));
            break;
        case OUTER_LEFT:
            add_box(shape, stair_quadrant(direction, true, true, y_min, y_max));
            break;
        case OUTER_RIGHT:
            add_box(shape, stair_quadrant(direction, false, true, y_min, y_max));
            break;
    }}
    return shape;
}}

constexpr BoxList connection_boxes(int mask, std::uint8_t center_min, std::uint8_t center_max) {{
    BoxList shape{{}};
    add_box(shape, box(center_min, 0, center_min, center_max, 16, center_max));
    if ((mask & 1) != 0) {{
        add_box(shape, box(center_min, 0, 0, center_max, 16, center_min));
    }}
    if ((mask & 2) != 0) {{
        add_box(shape, box(center_max, 0, center_min, 16, 16, center_max));
    }}
    if ((mask & 4) != 0) {{
        add_box(shape, box(center_min, 0, center_max, center_max, 16, 16));
    }}
    if ((mask & 8) != 0) {{
        add_box(shape, box(0, 0, center_min, center_min, 16, center_max));
    }}
    return shape;
}}

constexpr GeometryCatalog build_geometry_catalog() {{
    CatalogBuilder builder{{}};
    builder.catalog.shape_names = SHAPE_NAME_TABLE;

    add_shape(builder, BoxList{{}});

    BoxList full_cube{{}};
    add_box(full_cube, box(0, 0, 0, 16, 16, 16));
    add_shape(builder, full_cube);

    BoxList slab_bottom{{}};
    add_box(slab_bottom, box(0, 0, 0, 16, 8, 16));
    add_shape(builder, slab_bottom);

    BoxList slab_top{{}};
    add_box(slab_top, box(0, 8, 0, 16, 16, 16));
    add_shape(builder, slab_top);

    for (std::uint8_t direction = 0; direction < {len(directions)}; ++direction) {{
        for (std::uint8_t half = 0; half < {len(halves)}; ++half) {{
            for (std::uint8_t stair_shape = 0; stair_shape < {len(stair_shapes)}; ++stair_shape) {{
                add_shape(
                    builder,
                    stair_boxes(
                        static_cast<Direction>(direction),
                        half == 1,
                        static_cast<StairShape>(stair_shape)
                    )
                );
            }}
        }}
    }}

    for (int mask = 0; mask < 16; ++mask) {{
        add_shape(builder, connection_boxes(mask, {pane_center[0]}, {pane_center[1]}));
    }}

    for (int mask = 0; mask < 16; ++mask) {{
        add_shape(builder, connection_boxes(mask, {fence_center[0]}, {fence_center[1]}));
    }}

    return builder.catalog;
}}

constexpr GeometryCatalog CATALOG = build_geometry_catalog();

constexpr std::uint32_t used_box_count() {{
    std::uint32_t total = 0;
    for (const ShapeGeometry &shape : CATALOG.shapes) {{
        total += shape.box_count;
    }}
    return total;
}}

constexpr std::uint32_t used_face_count() {{
    std::uint32_t total = 0;
    for (const ShapeGeometry &shape : CATALOG.shapes) {{
        total += shape.face_count;
    }}
    return total;
}}

static_assert(used_box_count() == GEOMETRY_BOX_COUNT, "geometry box count changed");
static_assert(used_face_count() == GEOMETRY_FACE_COUNT, "geometry face count changed");

}}  // namespace

const GeometryCatalog &geometry_catalog() {{
    return CATALOG;
}}

const ShapeGeometry &geometry_for_shape(std::int32_t shape_id) {{
    if (shape_id < 0 || static_cast<std::size_t>(shape_id) >= CATALOG.shapes.size()) {{
        return CATALOG.shapes[static_cast<std::size_t>(SHAPE_FULL_CUBE)];
    }}
    return CATALOG.shapes[static_cast<std::size_t>(shape_id)];
}}

std::int32_t geometry_catalog_shape_count() {{
    return static_cast<std::int32_t>(CATALOG.shapes.size());
}}

const std::array<const char *, GEOMETRY_SHAPE_COUNT> &shape_names() {{
    return CATALOG.shape_names;
}}

const char *shape_id_name(std::int32_t shape_id) {{
    if (shape_id < 0 || static_cast<std::size_t>(shape_id) >= CATALOG.shape_names.size()) {{
        return "unknown_shape";
    }}
    return CATALOG.shape_names[static_cast<std::size_t>(shape_id)];
}}

std::int32_t shape_count() {{
    return static_cast<std::int32_t>(CATALOG.shape_names.size());
}}

}}
'''


def generate() -> Dict[Path, str]:
    catalog = load_catalog()
    shapes = expand_shapes(catalog)
    faces = [faces_for_shape(shape) for shape in shapes]
    box_count = sum(len(shape.boxes) for shape in shapes)
    face_count = sum(len(shape_faces) for shape_faces in faces)
    return {
        PYTHON_TARGET: render_python(catalog, shapes),
        CPP_HEADER_TARGET: render_header(catalog, shapes, box_count, face_count),
        CPP_SOURCE_TARGET: render_cpp(catalog, shapes),
    }


def check_outputs(outputs: Dict[Path, str]) -> bool:
    ok = True
    for path, content in outputs.items():
        if not path.exists():
            print(f"missing generated file: {path}", file=sys.stderr)
            ok = False
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != content:
            print(f"generated file is out of date: {path}", file=sys.stderr)
            ok = False
    return ok


def write_outputs(outputs: Dict[Path, str]) -> None:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify generated files without writing")
    args = parser.parse_args(argv)

    outputs = generate()
    if args.check:
        return 0 if check_outputs(outputs) else 1

    write_outputs(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
