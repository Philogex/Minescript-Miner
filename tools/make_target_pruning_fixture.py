#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path


FULL_CUBE_SHAPE_ID = 1


def parse_fixture(lines: list[str]):
    side = None
    position = None
    default_shape = 0
    shapes: dict[int, int] = {}
    block_indices: list[int] = []

    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "side":
            side = int(parts[1])
        elif parts[0] == "position":
            position = tuple(float(value) for value in parts[1:4])
        elif parts[0] == "default_shape":
            default_shape = int(parts[1])
        elif parts[0] in {"block", "target"}:
            index = int(parts[1])
            shape_id = int(parts[2])
            shapes[index] = shape_id
            if parts[0] == "block":
                block_indices.append(index)

    if side is None or position is None:
        raise ValueError("fixture must define side and position")

    return side, position, default_shape, shapes, block_indices


def index_to_offset(index: int, side: int) -> tuple[int, int, int]:
    return index % side, index // (side * side), (index // side) % side


def offset_to_index(offset: tuple[int, int, int], side: int) -> int:
    return offset[0] + offset[2] * side + offset[1] * side * side


def inside_cube(offset: tuple[int, int, int], side: int) -> bool:
    return all(0 <= component < side for component in offset)


def block_position(index: int, side: int, position: tuple[float, float, float]) -> tuple[int, int, int]:
    center = tuple(math.floor(component) for component in position)
    half = side // 2
    minimum = center[0] - half, center[1] - half, center[2] - half
    offset = index_to_offset(index, side)
    return minimum[0] + offset[0], minimum[1] + offset[1], minimum[2] + offset[2]


def is_full_cube(index: int, shapes: dict[int, int], default_shape: int) -> bool:
    return shapes.get(index, default_shape) == FULL_CUBE_SHAPE_ID


def visible_cube_faces(
    index: int,
    side: int,
    position: tuple[float, float, float],
) -> list[tuple[int, int]]:
    block_pos = block_position(index, side, position)
    faces: list[tuple[int, int]] = []
    if position[0] < block_pos[0]:
        faces.append((0, -1))
    elif position[0] > block_pos[0] + 1:
        faces.append((0, 1))

    if position[1] < block_pos[1]:
        faces.append((1, -1))
    elif position[1] > block_pos[1] + 1:
        faces.append((1, 1))

    if position[2] < block_pos[2]:
        faces.append((2, -1))
    elif position[2] > block_pos[2] + 1:
        faces.append((2, 1))
    return faces


def has_internal_full_cube_neighbor(
    index: int,
    axis: int,
    normal_sign: int,
    side: int,
    shapes: dict[int, int],
    default_shape: int,
) -> bool:
    offset = list(index_to_offset(index, side))
    offset[axis] += normal_sign
    neighbor = tuple(offset)
    return (
        inside_cube(neighbor, side)
        and is_full_cube(offset_to_index(neighbor, side), shapes, default_shape)
    )


def face_within_reach(
    index: int,
    axis: int,
    normal_sign: int,
    side: int,
    position: tuple[float, float, float],
    reach: float,
) -> bool:
    block_pos = block_position(index, side, position)
    minimum = [float(component) for component in block_pos]
    maximum = [float(component + 1) for component in block_pos]
    face_coord = block_pos[axis] + (1 if normal_sign > 0 else 0)
    minimum[axis] = float(face_coord)
    maximum[axis] = float(face_coord)

    distance_squared = 0.0
    for axis_index, value in enumerate(position):
        if value < minimum[axis_index]:
            distance = minimum[axis_index] - value
        elif value > maximum[axis_index]:
            distance = value - maximum[axis_index]
        else:
            distance = 0.0
        distance_squared += distance * distance
    return distance_squared <= reach * reach


def candidate_face_count(
    index: int,
    side: int,
    position: tuple[float, float, float],
    reach: float,
    shapes: dict[int, int],
    default_shape: int,
) -> int:
    if not is_full_cube(index, shapes, default_shape):
        return 0

    count = 0
    for axis, normal_sign in visible_cube_faces(index, side, position):
        if has_internal_full_cube_neighbor(index, axis, normal_sign, side, shapes, default_shape):
            continue
        if face_within_reach(index, axis, normal_sign, side, position, reach):
            count += 1
    return count


def reach_from_fixture(lines: list[str]) -> float:
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        parts = line.split()
        if parts and parts[0] == "reach":
            return float(parts[1])
    raise ValueError("fixture must define reach")


def expected_target_faces_from_fixture(lines: list[str]) -> int:
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        parts = line.split()
        if parts and parts[0] == "expect_target_faces":
            return int(parts[1])
    raise ValueError("fixture must define expect_target_faces")


def selected_occluders(
    lines: list[str],
    count: int,
) -> tuple[list[tuple[int, int]], int]:
    side, position, default_shape, shapes, block_indices = parse_fixture(lines)
    reach = reach_from_fixture(lines)
    candidates: list[tuple[int, int]] = []
    for index in block_indices:
        face_count = candidate_face_count(
            index,
            side,
            position,
            reach,
            shapes,
            default_shape,
        )
        if face_count:
            candidates.append((index, face_count))
    base_target_faces = expected_target_faces_from_fixture(lines)
    return (
        candidates[:count],
        base_target_faces + sum(face_count for _index, face_count in candidates[:count]),
    )


def write_fixture(
    source_lines: list[str],
    output: Path,
    selected: list[tuple[int, int]],
    expected_target_faces: int,
) -> None:
    selected_lookup = {index for index, _face_count in selected}
    selected_text = " ".join(str(index) for index, _face_count in selected)
    output_lines = [
        "# Derived from recorded_side_39.scan for target-pruning stress tests.",
        "# Converted occluder block indices: " + selected_text,
    ]

    inserted_expectations = False
    for raw_line in source_lines:
        if raw_line.startswith("# Recorded from"):
            continue
        if raw_line.startswith("expect_angle_") or raw_line.startswith("expect_min_clips"):
            continue
        if raw_line.startswith("expect_target_faces"):
            output_lines.append(f"expect_target_faces {expected_target_faces}")
            continue
        if raw_line.startswith("expect_world_faces"):
            output_lines.append(raw_line)
            continue
        if raw_line.startswith("expect_found"):
            output_lines.append(raw_line)
            inserted_expectations = True
            continue

        parts = raw_line.split()
        if len(parts) >= 3 and parts[0] == "block" and int(parts[1]) in selected_lookup:
            output_lines.append("target " + " ".join(parts[1:]))
        else:
            output_lines.append(raw_line)

    if not inserted_expectations:
        raise ValueError("source fixture must define expect_found")
    output.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a target-pruning fixture by converting visible occluders to targets."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()

    lines = args.source.read_text(encoding="utf-8").splitlines()
    selected, expected_target_faces = selected_occluders(lines, args.count)
    if len(selected) < args.count:
        raise ValueError(f"only found {len(selected)} visible occluders")

    write_fixture(lines, args.output, selected, expected_target_faces)

    print("converted_occluder_indices:", " ".join(str(index) for index, _face_count in selected))
    print("converted_occluder_count:", len(selected))
    print("expected_target_faces:", expected_target_faces)


if __name__ == "__main__":
    main()
