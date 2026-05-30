"""Build and verify in-game block fixtures for every known shape id."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import minescript as m

from minescript_miner.adapter.block_ids import (
    DEFAULT_CATALOG,
    DIRECTIONS,
    HALVES,
    SHAPE_COUNT,
    SHAPE_FULL_CUBE,
    SHAPE_ID_BY_NAME,
    SHAPE_NAMES,
    STAIR_SHAPES,
)
from shape_catalog_check import assert_shape_catalog_parity


BlockPos = Tuple[int, int, int]
DEBUG_REPORT = PROJECT_ROOT / "shape_fixture_check.log"


@dataclass(frozen=True)
class ShapeFixture:
    label: str
    block_state: str
    expected_shape_id: int
    mode: str = "exact"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _connection_properties(mask: int) -> str:
    return ",".join(
        f"{direction}={_bool_text(bool(mask & (1 << bit)))}"
        for bit, direction in enumerate(DIRECTIONS)
    )


def _shape_id(name: str) -> int:
    return SHAPE_ID_BY_NAME[name]


def build_shape_fixtures() -> List[ShapeFixture]:
    fixtures = [
        ShapeFixture("empty_air", "minecraft:air", _shape_id("empty")),
        ShapeFixture("fallback_stone", "minecraft:stone", SHAPE_FULL_CUBE, "fallback"),
        ShapeFixture(
            "oak_slab_double",
            "minecraft:oak_slab[type=double,waterlogged=false]",
            SHAPE_FULL_CUBE,
            "alias",
        ),
        ShapeFixture(
            "oak_slab_bottom",
            "minecraft:oak_slab[type=bottom,waterlogged=false]",
            _shape_id("oak_slab_bottom"),
        ),
        ShapeFixture(
            "oak_slab_top",
            "minecraft:oak_slab[type=top,waterlogged=false]",
            _shape_id("oak_slab_top"),
        ),
    ]

    for facing in DIRECTIONS:
        for half in HALVES:
            for stair_shape in STAIR_SHAPES:
                name = f"oak_stairs_{facing}_{half}_{stair_shape}"
                fixtures.append(
                    ShapeFixture(
                        name,
                        "minecraft:oak_stairs["
                        f"facing={facing},half={half},shape={stair_shape},"
                        "waterlogged=false"
                        "]",
                        _shape_id(name),
                    )
                )

    for mask in range(16):
        connection = _connection_properties(mask)
        name = SHAPE_NAMES[_shape_id("pane_none") + mask]
        fixtures.append(
            ShapeFixture(
                f"glass_{name}",
                f"minecraft:glass_pane[{connection},waterlogged=false]",
                _shape_id(name),
            )
        )
        fixtures.append(
            ShapeFixture(
                f"iron_bars_{name}",
                f"minecraft:iron_bars[{connection},waterlogged=false]",
                _shape_id(name),
                "alias",
            )
        )

    for mask in range(16):
        connection = _connection_properties(mask)
        name = SHAPE_NAMES[_shape_id("oak_fence_none") + mask]
        fixtures.append(
            ShapeFixture(
                name,
                f"minecraft:oak_fence[{connection},waterlogged=false]",
                _shape_id(name),
            )
        )

    return fixtures


def _fixture_positions(
    base: BlockPos,
    fixture_count: int,
    columns: int = 10,
    spacing: int = 3,
) -> List[BlockPos]:
    base_x, base_y, base_z = base
    return [
        (
            base_x + (index % columns) * spacing,
            base_y,
            base_z + (index // columns) * spacing,
        )
        for index in range(fixture_count)
    ]


def _bounds(positions: Sequence[BlockPos]) -> Tuple[BlockPos, BlockPos]:
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    zs = [pos[2] for pos in positions]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _command(command: str) -> None:
    m.execute(command)


def build_fixture_area(fixtures: Sequence[ShapeFixture], positions: Sequence[BlockPos]) -> None:
    min_pos, max_pos = _bounds(positions)
    m.await_loaded_region(min_pos[0], min_pos[2], max_pos[0], max_pos[2])
    _command(
        "fill "
        f"{min_pos[0] - 1} {min_pos[1]} {min_pos[2] - 1} "
        f"{max_pos[0] + 1} {max_pos[1]} {max_pos[2] + 1} "
        "minecraft:air"
    )
    time.sleep(0.1)

    for fixture, pos in zip(fixtures, positions):
        x, y, z = pos
        _command(f"setblock {x} {y} {z} {fixture.block_state}")

    # Let neighbor/block-state updates settle before reading the region back.
    time.sleep(0.5)


def read_fixture_blocks(positions: Sequence[BlockPos]) -> Dict[BlockPos, str]:
    min_pos, max_pos = _bounds(positions)
    m.await_loaded_region(min_pos[0], min_pos[2], max_pos[0], max_pos[2])
    region = m.get_block_region(min_pos, max_pos)
    return {pos: region.get_block(*pos) for pos in positions}


def assert_fixture_shapes(
    fixtures: Sequence[ShapeFixture],
    positions: Sequence[BlockPos],
) -> List[str]:
    block_strings = read_fixture_blocks(positions)
    failures: List[str] = []

    for fixture, pos in zip(fixtures, positions):
        block_string = block_strings[pos]
        actual_shape_id = DEFAULT_CATALOG.shape_id(block_string)
        if actual_shape_id == fixture.expected_shape_id:
            continue

        expected_name = SHAPE_NAMES[fixture.expected_shape_id]
        actual_name = (
            SHAPE_NAMES[actual_shape_id]
            if 0 <= actual_shape_id < len(SHAPE_NAMES)
            else f"unknown({actual_shape_id})"
        )
        failures.append(
            f"{fixture.label} at {pos}: expected {fixture.expected_shape_id} "
            f"{expected_name}, got {actual_shape_id} {actual_name}; "
            f"read {block_string!r}"
        )

    return failures


def wait_for_fixture_shapes(
    fixtures: Sequence[ShapeFixture],
    positions: Sequence[BlockPos],
    *,
    attempts: int = 30,
    delay_seconds: float = 0.2,
) -> List[str]:
    last_failures: List[str] = []
    for attempt in range(1, attempts + 1):
        last_failures = assert_fixture_shapes(fixtures, positions)
        if not last_failures:
            return []

        if attempt in (1, attempts):
            m.echo(
                "Shape fixture retry: "
                f"{len(last_failures)} mismatch(es), attempt {attempt}/{attempts}"
            )
        time.sleep(delay_seconds)

    return last_failures


def write_debug_report(
    fixtures: Sequence[ShapeFixture],
    positions: Sequence[BlockPos],
    failures: Sequence[str],
) -> None:
    try:
        block_strings = read_fixture_blocks(positions)
    except Exception as exc:
        block_strings = {}
        read_error = repr(exc)
    else:
        read_error = None

    with DEBUG_REPORT.open("w", encoding="utf-8") as report:
        report.write("shape_fixture_check debug report\n")
        report.write(f"failure_count: {len(failures)}\n")
        if read_error is not None:
            report.write(f"read_error: {read_error}\n")
        report.write("\nfailures:\n")
        for failure in failures:
            report.write(f"- {failure}\n")

        report.write("\nfixtures:\n")
        for fixture, pos in zip(fixtures, positions):
            block_string = block_strings.get(pos, "<not read>")
            actual_shape_id = DEFAULT_CATALOG.shape_id(block_string)
            actual_name = (
                SHAPE_NAMES[actual_shape_id]
                if 0 <= actual_shape_id < len(SHAPE_NAMES)
                else f"unknown({actual_shape_id})"
            )
            report.write(
                f"{pos} label={fixture.label} mode={fixture.mode} "
                f"expected={fixture.expected_shape_id}:{SHAPE_NAMES[fixture.expected_shape_id]} "
                f"actual={actual_shape_id}:{actual_name} "
                f"set={fixture.block_state!r} read={block_string!r}\n"
            )


def assert_fixture_coverage(fixtures: Iterable[ShapeFixture]) -> None:
    covered_shape_ids = {fixture.expected_shape_id for fixture in fixtures}
    missing = [
        f"{shape_id}:{shape_name}"
        for shape_id, shape_name in enumerate(SHAPE_NAMES)
        if shape_id not in covered_shape_ids
    ]
    if missing:
        raise AssertionError("Missing fixture coverage for " + ", ".join(missing))

    if len(SHAPE_NAMES) != SHAPE_COUNT:
        raise AssertionError(
            f"Shape count mismatch inside Python catalog: "
            f"SHAPE_COUNT={SHAPE_COUNT}, len(SHAPE_NAMES)={len(SHAPE_NAMES)}"
        )


def default_base_position() -> BlockPos:
    px, py, pz = m.player_position()
    return (math.floor(px) + 4, math.floor(py), math.floor(pz) + 4)


def main() -> None:
    catalog = assert_shape_catalog_parity()
    fixtures = build_shape_fixtures()
    assert_fixture_coverage(fixtures)

    base = default_base_position()
    positions = _fixture_positions(base, len(fixtures))
    m.echo(
        "Shape fixture check: "
        f"building {len(fixtures)} fixtures for {catalog['shape_count']} shapes "
        f"near {base}"
    )

    build_fixture_area(fixtures, positions)
    failures = wait_for_fixture_shapes(fixtures, positions)
    if failures:
        write_debug_report(fixtures, positions, failures)
        m.echo(f"Shape fixture FAIL: {len(failures)} mismatch(es)")
        for failure in failures[:8]:
            m.echo(failure)
        if len(failures) > 8:
            m.echo(f"... {len(failures) - 8} more")
        m.echo(f"Debug report: {DEBUG_REPORT}")
        raise AssertionError("\n".join(failures))

    exact_count = sum(1 for fixture in fixtures if fixture.mode == "exact")
    alias_count = sum(1 for fixture in fixtures if fixture.mode == "alias")
    fallback_count = sum(1 for fixture in fixtures if fixture.mode == "fallback")
    m.echo(
        "Shape fixture OK: "
        f"{len(fixtures)} cases, {exact_count} exact, "
        f"{alias_count} alias, {fallback_count} fallback"
    )


if __name__ == "__main__":
    main()
