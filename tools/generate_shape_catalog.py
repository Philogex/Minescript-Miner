#!/usr/bin/env python3
"""Generate Java and C++ shape/geometry catalog files from JSON."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "catalog" / "shape_catalog.json"
CPP_CONTRACT_HEADER_TARGET = ROOT / "native" / "include" / "minecraft_miner" / "catalog" / "catalog_contract.hpp"
CPP_HEADER_TARGET = ROOT / "native" / "include" / "minecraft_miner" / "catalog" / "geometry_catalog.hpp"
CPP_DATA_HEADER_TARGET = ROOT / "native" / "include" / "minecraft_miner" / "catalog" / "geometry_catalog_data.hpp"
JAVA_CONTRACT_TARGET = ROOT / "fabric/src/main/java/dev/philogex/miner/catalog/GeneratedCatalog.java"
JAVA_MAPPING_TARGET = ROOT / "fabric/src/main/resources/catalog/block_shapes.tsv"

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


def stair_quadrant(
    direction: str,
    side: str,
    front: bool,
    y_min: int,
    y_max: int,
    units: int,
) -> Box:
    front_x = half_bounds_1d(direction, front, True, units)
    front_z = half_bounds_1d(direction, front, False, units)
    lateral = lateral_direction(direction, side)
    lateral_x = half_bounds_1d(lateral, True, True, units)
    lateral_z = half_bounds_1d(lateral, True, False, units)
    return (
        max(front_x[0], lateral_x[0]),
        y_min,
        max(front_z[0], lateral_z[0]),
        min(front_x[1], lateral_x[1]),
        y_max,
        min(front_z[1], lateral_z[1]),
    )


def half_bounds_1d(
    direction: str,
    front: bool,
    x_axis: bool,
    units: int,
) -> Tuple[int, int]:
    half = units // 2
    if direction == "north":
        return (0, units) if x_axis else ((0, half) if front else (half, units))
    if direction == "south":
        return (0, units) if x_axis else ((half, units) if front else (0, half))
    if direction == "east":
        return ((half, units) if front else (0, half)) if x_axis else (0, units)
    if direction == "west":
        return ((0, half) if front else (half, units)) if x_axis else (0, units)
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


def stair_boxes(
    direction: str,
    half: str,
    stair_shape: str,
    units: int,
) -> Tuple[Box, ...]:
    boxes: List[Box] = []
    half_units = units // 2
    y_min, y_max = half_units, units
    if half == "bottom":
        boxes.append((0, 0, 0, units, half_units, units))
    else:
        boxes.append((0, half_units, 0, units, units, units))
        y_min, y_max = 0, half_units

    quadrants = {
        "straight": (("left", True), ("right", True)),
        "outer_left": (("left", True),),
        "outer_right": (("right", True),),
        "inner_left": (("left", True), ("right", True), ("left", False)),
        "inner_right": (("left", True), ("right", True), ("right", False)),
    }[stair_shape]
    for side, front in quadrants:
        boxes.append(stair_quadrant(direction, side, front, y_min, y_max, units))
    return tuple(boxes)


def connection_boxes(
    mask: int,
    center_min: int,
    center_max: int,
    units: int,
) -> Tuple[Box, ...]:
    boxes: List[Box] = [(center_min, 0, center_min, center_max, units, center_max)]
    if mask & 1:
        boxes.append((center_min, 0, 0, center_max, units, center_min))
    if mask & 2:
        boxes.append((center_max, 0, center_min, units, units, center_max))
    if mask & 4:
        boxes.append((center_min, 0, center_max, center_max, units, units))
    if mask & 8:
        boxes.append((0, 0, center_min, center_min, units, center_max))
    return tuple(boxes)


def button_box(face: str, facing: str, powered: str, units: int) -> Box:
    if units % 16 != 0:
        raise ValueError("button geometry requires a multiple of 16 units per block")
    scale = units // 16
    depth = scale if powered == "true" else 2 * scale
    x5, x6, x10, x11 = (value * scale for value in (5, 6, 10, 11))

    if face == "floor":
        if facing in ("north", "south"):
            return x5, 0, x6, x11, depth, x10
        return x6, 0, x5, x10, depth, x11

    if face == "ceiling":
        if facing in ("north", "south"):
            return x5, units - depth, x6, x11, units, x10
        return x6, units - depth, x5, x10, units, x11

    if facing == "north":
        return x5, x6, units - depth, x11, x10, units
    if facing == "south":
        return x5, x6, 0, x11, x10, depth
    if facing == "west":
        return units - depth, x6, x5, units, x10, x11
    if facing == "east":
        return 0, x6, x5, depth, x10, x11
    raise ValueError(f"unknown button facing: {facing}")


def expand_shapes(catalog: Dict[str, Any]) -> List[Shape]:
    directions = catalog["directions"]
    halves = catalog["halves"]
    stair_shapes = catalog["stair_shapes"]
    units = catalog["geometry_units_per_block"]
    if units <= 0 or units > 255 or units % 2 != 0:
        raise ValueError("geometry_units_per_block must be an even value in 1..255")
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
                                stair_boxes(direction, half, stair_shape, units),
                            )
                        )
        elif spec.get("family") == "connection":
            center_min, center_max = spec["center"]
            for mask in range(16):
                shapes.append(
                    Shape(
                        spec["template"].format(connection=connection_name(mask, directions)),
                        connection_boxes(mask, center_min, center_max, units),
                    )
                )
        elif spec.get("family") == "button":
            for face in spec["faces"]:
                for facing in directions:
                    for powered in ("true", "false"):
                        shapes.append(
                            Shape(
                                spec["template"].format(
                                    face=face,
                                    facing=facing,
                                    powered=powered,
                                ),
                                (button_box(face, facing, powered, units),),
                            )
                        )
        else:
            raise ValueError(f"unsupported shape spec: {spec}")

    return shapes


def validate_shapes(shapes: Sequence[Shape], units: int) -> None:
    seen_names = set()
    for shape in shapes:
        if shape.name in seen_names:
            raise ValueError(f"duplicate shape name: {shape.name}")
        seen_names.add(shape.name)

        for box in shape.boxes:
            if len(box) != 6:
                raise ValueError(f"{shape.name}: expected six AABB coordinates")
            min_x, min_y, min_z, max_x, max_y, max_z = box
            if not (
                0 <= min_x < max_x <= units
                and 0 <= min_y < max_y <= units
                and 0 <= min_z < max_z <= units
            ):
                raise ValueError(
                    f"{shape.name}: AABB {box} is outside the 0..{units} grid"
                )


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


def resolve_blocks(catalog: Dict[str, Any], spec: Dict[str, Any]) -> List[str]:
    if "block" in spec:
        return [spec["block"]]
    if "blocks" in spec:
        return list(spec["blocks"])
    if "block_group" in spec:
        return list(catalog["block_groups"][spec["block_group"]])
    raise ValueError(f"block mapping has no block source: {spec}")


def resolve_state_values(catalog: Dict[str, Any], spec: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    state_values = spec["state_values"]
    property_names = tuple(state_values)
    value_lists = []
    for property_name in property_names:
        values = state_values[property_name]
        if isinstance(values, str):
            if not values.startswith("$"):
                raise ValueError(f"state value reference must start with '$': {values}")
            values = catalog[values[1:]]
        value_lists.append(values)

    for values in itertools.product(*value_lists):
        yield dict(zip(property_names, values))


def expand_block_mappings(catalog: Dict[str, Any], shape_ids: Dict[str, int]) -> Tuple[Dict[Tuple[str, Tuple[Tuple[str, str], ...]], int], Dict[str, Tuple[str, ...]]]:
    directions = catalog["directions"]
    mapping: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], int] = {}
    relevant: Dict[str, Tuple[str, ...]] = {}

    for spec in catalog["block_mappings"]:
        blocks = resolve_blocks(catalog, spec)
        properties = tuple(spec["properties"])
        for block in blocks:
            if block in relevant:
                raise ValueError(f"duplicate block mapping: {block}")
            relevant[block] = properties

        if "states" in spec:
            for block in blocks:
                for state in spec["states"]:
                    mapping[(block, state_key(state["properties"]))] = shape_ids[state["shape"]]
        elif "shape_template" in spec:
            for block in blocks:
                for properties_dict in resolve_state_values(catalog, spec):
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


def render_contract_header(catalog: Dict[str, Any], shapes: Sequence[Shape], box_count: int, face_count: int) -> str:
    return f"""// Generated by tools/generate_shape_catalog.py from catalog/shape_catalog.json.
// Catalog compatibility constants. Do not edit by hand.

#pragma once

#include <cstddef>
#include <cstdint>

namespace minecraft_miner {{

inline constexpr int SHAPE_CATALOG_VERSION = {catalog["shape_catalog_version"]};
inline constexpr char SHAPE_CATALOG_SHA256[] = "{hashlib.sha256(SOURCE.read_bytes()).hexdigest()}";
inline constexpr int GEOMETRY_CATALOG_VERSION = {catalog["geometry_catalog_version"]};
inline constexpr int GEOMETRY_SHAPE_CATALOG_VERSION = SHAPE_CATALOG_VERSION;
inline constexpr int BLOCK_SHAPE_MAPPING_VERSION = {catalog["block_shape_mapping_version"]};
inline constexpr std::int32_t GEOMETRY_UNITS_PER_BLOCK = {catalog["geometry_units_per_block"]};
inline constexpr std::size_t GEOMETRY_SHAPE_COUNT = {len(shapes)};
inline constexpr std::size_t GEOMETRY_BOX_COUNT = {box_count};
inline constexpr std::size_t GEOMETRY_FACE_COUNT = {face_count};
inline constexpr int MAX_CUBE_SIDE = 39;

}}
"""


def render_header(catalog: Dict[str, Any], shapes: Sequence[Shape], box_count: int, face_count: int) -> str:
    return f"""// Generated by tools/generate_shape_catalog.py from catalog/shape_catalog.json.
// Do not edit by hand.

#pragma once

#include "minecraft_miner/catalog/catalog_contract.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace minecraft_miner {{

inline constexpr std::int32_t SHAPE_EMPTY = 0;
inline constexpr std::int32_t SHAPE_FULL_CUBE = 1;
inline constexpr std::int32_t SHAPE_SLAB_BOTTOM = 2;
inline constexpr std::int32_t SHAPE_SLAB_TOP = 3;

enum class PlaneAxis : std::int32_t {{
    X = 0,
    Y = 1,
    Z = 2,
}};

struct LocalAabb {{
    std::uint8_t min_x;
    std::uint8_t min_y;
    std::uint8_t min_z;
    std::uint8_t max_x;
    std::uint8_t max_y;
    std::uint8_t max_z;
}};

struct LocalRectFace {{
    PlaneAxis axis;
    std::uint8_t coord;
    std::uint8_t u_min;
    std::uint8_t u_max;
    std::uint8_t v_min;
    std::uint8_t v_max;
    std::int8_t normal_sign;
}};

struct ShapeGeometry {{
    std::uint16_t face_offset;
    std::uint8_t face_count;
}};

struct GeometryCatalog {{
    std::array<const char *, GEOMETRY_SHAPE_COUNT> shape_names;
    std::array<ShapeGeometry, GEOMETRY_SHAPE_COUNT> shapes;
    std::array<LocalRectFace, GEOMETRY_FACE_COUNT> faces;
}};

const GeometryCatalog &geometry_catalog();
const ShapeGeometry &geometry_for_shape(std::int32_t shape_id);
std::int32_t geometry_catalog_shape_count();

const char *shape_id_name(std::int32_t shape_id);
std::uint8_t shape_box_count(std::int32_t shape_id);
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

#include "minecraft_miner/catalog/geometry_catalog.hpp"

#include <array>
#include <cstdint>

namespace minecraft_miner::generated {{

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

inline constexpr std::array<LocalAabb, GEOMETRY_BOX_COUNT> SHAPE_BOX_TABLE = {{{{
{chr(10).join(box_lines)}
}}}};

}}  // namespace minecraft_miner::generated
"""


def render_java(catalog: Dict[str, Any], shapes: Sequence[Shape]) -> str:
    names = ",\n        ".join(json.dumps(shape.name) for shape in shapes)
    return f'''// Generated by tools/generate_shape_catalog.py. Do not edit.
package dev.philogex.miner.catalog;

public final class GeneratedCatalog {{
    private GeneratedCatalog() {{}}
    public static final int SHAPE_CATALOG_VERSION = {catalog["shape_catalog_version"]};
    public static final String SHAPE_CATALOG_SHA256 = "{hashlib.sha256(SOURCE.read_bytes()).hexdigest()}";
    public static final int BLOCK_SHAPE_MAPPING_VERSION = {catalog["block_shape_mapping_version"]};
    public static final int MAX_CUBE_SIDE = 39;
    public static final int SHAPE_EMPTY = 0;
    public static final int SHAPE_FULL_CUBE = 1;
    public static final String[] SHAPE_NAMES = {{
        {names}
    }};
}}
'''


def render_java_mapping(catalog: Dict[str, Any], shapes: Sequence[Shape]) -> str:
    mapping, relevant = expand_block_mappings(catalog, {shape.name: i for i, shape in enumerate(shapes)})
    lines = ["# Generated by tools/generate_shape_catalog.py. Do not edit."]
    lines += [f"empty\t{block}" for block in sorted(catalog["empty_blocks"])]
    lines += [f"properties\t{block}\t{','.join(properties)}" for block, properties in sorted(relevant.items())]
    for (block, state), shape_id in sorted(mapping.items()):
        properties = ','.join(f"{key}={value}" for key, value in state)
        lines.append(f"shape\t{block}[{properties}]\t{shape_id}")
    return '\n'.join(lines) + '\n'


def generate() -> Dict[Path, str]:
    catalog = load_catalog()
    shapes = expand_shapes(catalog)
    validate_shapes(shapes, catalog["geometry_units_per_block"])
    faces = [faces_for_shape(shape) for shape in shapes]
    box_count = sum(len(shape.boxes) for shape in shapes)
    face_count = sum(len(shape_faces) for shape_faces in faces)
    return {
        CPP_CONTRACT_HEADER_TARGET: render_contract_header(catalog, shapes, box_count, face_count),
        CPP_HEADER_TARGET: render_header(catalog, shapes, box_count, face_count),
        CPP_DATA_HEADER_TARGET: render_data_header(shapes),
        JAVA_CONTRACT_TARGET: render_java(catalog, shapes),
        JAVA_MAPPING_TARGET: render_java_mapping(catalog, shapes),
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
