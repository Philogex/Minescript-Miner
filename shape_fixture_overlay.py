"""Visualize fixture shape AABB corners with particles in-game."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import minescript as m

from minescript_miner.adapter.shape_catalog import DEFAULT_CATALOG, SHAPE_NAMES
from shape_fixture_check import (
    BlockPos,
    build_fixture_area,
    build_shape_fixtures,
    default_base_position,
    read_fixture_blocks,
    wait_for_fixture_shapes,
    _fixture_positions,
)


Box = Tuple[float, float, float, float, float, float]
Point = Tuple[float, float, float]

OVERLAY_DURATION_SECONDS = 10.0
PARTICLE_INTERVAL_SECONDS = 2.0
PARTICLE = "minecraft:end_rod"


def _box(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> Box:
    return x1, y1, z1, x2, y2, z2


def _connection_boxes(name: str, *, post_width: float, arm_width: float) -> List[Box]:
    half_post = post_width / 2.0
    half_arm = arm_width / 2.0
    center_min = 0.5 - half_post
    center_max = 0.5 + half_post
    arm_min = 0.5 - half_arm
    arm_max = 0.5 + half_arm
    parts = set(name.split("_"))

    boxes = [_box(center_min, 0.0, center_min, center_max, 1.0, center_max)]
    if "north" in parts:
        boxes.append(_box(arm_min, 0.0, 0.0, arm_max, 1.0, center_min))
    if "east" in parts:
        boxes.append(_box(center_max, 0.0, arm_min, 1.0, 1.0, arm_max))
    if "south" in parts:
        boxes.append(_box(arm_min, 0.0, center_max, arm_max, 1.0, 1.0))
    if "west" in parts:
        boxes.append(_box(0.0, 0.0, arm_min, center_min, 1.0, arm_max))
    return boxes


def _half_bounds(direction: str, *, front: bool) -> Tuple[float, float, float, float]:
    if direction == "north":
        return (0.0, 1.0, 0.0, 0.5) if front else (0.0, 1.0, 0.5, 1.0)
    if direction == "south":
        return (0.0, 1.0, 0.5, 1.0) if front else (0.0, 1.0, 0.0, 0.5)
    if direction == "east":
        return (0.5, 1.0, 0.0, 1.0) if front else (0.0, 0.5, 0.0, 1.0)
    if direction == "west":
        return (0.0, 0.5, 0.0, 1.0) if front else (0.5, 1.0, 0.0, 1.0)
    raise ValueError(f"Unknown direction: {direction}")


def _quadrant_bounds(direction: str, side: str, *, front: bool) -> Tuple[float, float, float, float]:
    front_x1, front_x2, front_z1, front_z2 = _half_bounds(direction, front=front)
    left_direction = {
        "north": "west",
        "east": "north",
        "south": "east",
        "west": "south",
    }[direction]
    lateral_direction = left_direction if side == "left" else {
        "north": "east",
        "east": "south",
        "south": "west",
        "west": "north",
    }[direction]
    lat_x1, lat_x2, lat_z1, lat_z2 = _half_bounds(lateral_direction, front=True)
    return (
        max(front_x1, lat_x1),
        min(front_x2, lat_x2),
        max(front_z1, lat_z1),
        min(front_z2, lat_z2),
    )


def _stair_boxes(direction: str, half: str, stair_shape: str) -> List[Box]:
    if half == "bottom":
        boxes = [_box(0.0, 0.0, 0.0, 1.0, 0.5, 1.0)]
        y1, y2 = 0.5, 1.0
    else:
        boxes = [_box(0.0, 0.5, 0.0, 1.0, 1.0, 1.0)]
        y1, y2 = 0.0, 0.5

    quadrants = {
        "straight": (("left", True), ("right", True)),
        "outer_left": (("left", True),),
        "outer_right": (("right", True),),
        "inner_left": (("left", True), ("right", True), ("left", False)),
        "inner_right": (("left", True), ("right", True), ("right", False)),
    }[stair_shape]
    for side, front in quadrants:
        x1, x2, z1, z2 = _quadrant_bounds(direction, side, front=front)
        boxes.append(_box(x1, y1, z1, x2, y2, z2))
    return boxes


def boxes_for_shape_name(shape_name: str) -> List[Box]:
    if shape_name == "empty":
        return []
    if shape_name == "full_cube":
        return [_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]
    if shape_name == "slab_bottom":
        return [_box(0.0, 0.0, 0.0, 1.0, 0.5, 1.0)]
    if shape_name == "slab_top":
        return [_box(0.0, 0.5, 0.0, 1.0, 1.0, 1.0)]
    if shape_name.startswith("stairs_"):
        _, direction, half, *shape_parts = shape_name.split("_")
        return _stair_boxes(direction, half, "_".join(shape_parts))
    if shape_name.startswith("pane_"):
        connection = shape_name.removeprefix("pane_")
        return _connection_boxes(connection, post_width=2.0 / 16.0, arm_width=2.0 / 16.0)
    if shape_name.startswith("fence_"):
        connection = shape_name.removeprefix("fence_")
        return _connection_boxes(connection, post_width=4.0 / 16.0, arm_width=4.0 / 16.0)
    if shape_name == "carpet":
        return [_box(0.0, 0.0, 0.0, 1.0, 1.0 / 16.0, 1.0)]
    if shape_name == "torch":
        return [_box(6.0 / 16.0, 0.0, 6.0 / 16.0, 10.0 / 16.0, 10.0 / 16.0, 10.0 / 16.0)]
    if shape_name.startswith("ladder_"):
        direction = shape_name.removeprefix("ladder_")
        return {
            "north": [_box(0.0, 0.0, 13.0 / 16.0, 1.0, 1.0, 1.0)],
            "east": [_box(0.0, 0.0, 0.0, 3.0 / 16.0, 1.0, 1.0)],
            "south": [_box(0.0, 0.0, 0.0, 1.0, 1.0, 3.0 / 16.0)],
            "west": [_box(13.0 / 16.0, 0.0, 0.0, 1.0, 1.0, 1.0)],
        }[direction]
    if shape_name.startswith("button_"):
        _, face, facing, powered = shape_name.split("_")
        depth = 1.0 / 16.0 if powered == "true" else 2.0 / 16.0
        if face == "floor":
            if facing in ("north", "south"):
                return [_box(5 / 16, 0.0, 6 / 16, 11 / 16, depth, 10 / 16)]
            return [_box(6 / 16, 0.0, 5 / 16, 10 / 16, depth, 11 / 16)]
        if face == "ceiling":
            if facing in ("north", "south"):
                return [_box(5 / 16, 1.0 - depth, 6 / 16, 11 / 16, 1.0, 10 / 16)]
            return [_box(6 / 16, 1.0 - depth, 5 / 16, 10 / 16, 1.0, 11 / 16)]
        if facing == "north":
            return [_box(5 / 16, 6 / 16, 1.0 - depth, 11 / 16, 10 / 16, 1.0)]
        if facing == "south":
            return [_box(5 / 16, 6 / 16, 0.0, 11 / 16, 10 / 16, depth)]
        if facing == "west":
            return [_box(1.0 - depth, 6 / 16, 5 / 16, 1.0, 10 / 16, 11 / 16)]
        return [_box(0.0, 6 / 16, 5 / 16, depth, 10 / 16, 11 / 16)]
    return [_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)]


def box_corners(box: Box, block_pos: BlockPos) -> Set[Point]:
    x1, y1, z1, x2, y2, z2 = box
    bx, by, bz = block_pos
    return {
        (bx + x, by + y, bz + z)
        for x in (x1, x2)
        for y in (y1, y2)
        for z in (z1, z2)
    }


def corner_points(shape_id: int, block_pos: BlockPos) -> Set[Point]:
    shape_name = SHAPE_NAMES[shape_id]
    points: Set[Point] = set()
    for box in boxes_for_shape_name(shape_name):
        points.update(box_corners(box, block_pos))
    return points


def particle_command(point: Point) -> str:
    x, y, z = point
    return f"particle {PARTICLE} {x:.4f} {y:.4f} {z:.4f} 0 0 0 0 1 force"


def spawn_corner_particles(points: Iterable[Point]) -> int:
    count = 0
    for point in points:
        m.execute(particle_command(point))
        count += 1
    return count


def actual_shape_points(fixtures, positions: Sequence[BlockPos]) -> Set[Point]:
    block_strings = read_fixture_blocks(positions)
    points: Set[Point] = set()
    for pos in positions:
        block_string = block_strings[pos]
        shape_id = DEFAULT_CATALOG.shape_id(block_string)
        points.update(corner_points(shape_id, pos))
    return points


def main() -> None:
    fixtures = build_shape_fixtures()
    base = default_base_position()
    positions = _fixture_positions(base, len(fixtures))

    m.echo(f"Shape fixture overlay: building scenario near {base}")
    build_fixture_area(fixtures, positions)
    failures = wait_for_fixture_shapes(fixtures, positions)
    if failures:
        m.echo(f"Shape fixture overlay aborted: {len(failures)} mismatch(es)")
        for failure in failures[:8]:
            m.echo(failure)
        raise AssertionError("\n".join(failures))

    points = actual_shape_points(fixtures, positions)
    frames = max(1, int(OVERLAY_DURATION_SECONDS / PARTICLE_INTERVAL_SECONDS))
    m.echo(
        "Shape fixture overlay: "
        f"{len(points)} AABB corner points, {OVERLAY_DURATION_SECONDS:.0f}s"
    )
    for frame in range(frames):
        spawned = spawn_corner_particles(points)
        if frame == 0:
            m.echo(f"Spawned {spawned} corner particles per overlay frame")
        time.sleep(PARTICLE_INTERVAL_SECONDS)

    m.echo("Shape fixture overlay done")


if __name__ == "__main__":
    main()
