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
CPP_DATA_HEADER_TARGET = ROOT / "native" / "include" / "minescript_miner" / "geometry_catalog_data.hpp"

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


def render_data_header(shapes: Sequence[Shape]) -> str:
    shape_names = ",\n".join(f'    "{shape.name}"' for shape in shapes)

    box_lines: List[str] = []
    range_lines: List[str] = []
    offset = 0
    for shape in shapes:
        range_lines.append(f"    {{{offset}, {len(shape.boxes)}}},")
        for box_values in shape.boxes:
            box_lines.append("    {" + ", ".join(str(value) for value in box_values) + "},")
        offset += len(shape.boxes)

    return f"""// Generated by tools/generate_shape_catalog.py from catalog/shape_catalog.json.
// Data-only catalog tables. Do not edit by hand.

#pragma once

#include "minescript_miner/geometry_catalog.hpp"

#include <array>
#include <cstdint>

namespace minescript_miner::generated {{

struct ShapeBoxRange {{
    std::uint16_t offset;
    std::uint8_t count;
}};

inline constexpr std::array<const char *, GEOMETRY_SHAPE_COUNT> SHAPE_NAME_TABLE = {{
{shape_names}
}};

inline constexpr std::array<ShapeBoxRange, GEOMETRY_SHAPE_COUNT> SHAPE_BOX_RANGES = {{{{
{chr(10).join(range_lines)}
}}}};

inline constexpr std::array<Aabb16, GEOMETRY_BOX_COUNT> SHAPE_BOX_TABLE = {{{{
{chr(10).join(box_lines)}
}}}};

}}  // namespace minescript_miner::generated
"""


def generate() -> Dict[Path, str]:
    catalog = load_catalog()
    shapes = expand_shapes(catalog)
    faces = [faces_for_shape(shape) for shape in shapes]
    box_count = sum(len(shape.boxes) for shape in shapes)
    face_count = sum(len(shape_faces) for shape_faces in faces)
    return {
        PYTHON_TARGET: render_python(catalog, shapes),
        CPP_HEADER_TARGET: render_header(catalog, shapes, box_count, face_count),
        CPP_DATA_HEADER_TARGET: render_data_header(shapes),
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
