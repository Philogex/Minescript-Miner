"""Record native scan fixtures from the current in-game world."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import minescript as m


PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PROJECT_DIR / "src"
TARGET_CONFIG = PROJECT_DIR / "targets.txt"
FIXTURE_DIR = PROJECT_DIR / "native" / "tests" / "fixtures"
SIDES = (5, 39)
TARGET_REACH = 4.8

for path in (PROJECT_DIR, SRC_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from minescript_miner.adapter.shape_catalog import DEFAULT_CATALOG, SHAPE_EMPTY
from minescript_miner.adapter.target_pipeline import block_id_literal, load_target_blocks
from minescript_miner.minescript.world import get_area


ScanPosition = Tuple[float, float, float]
Orientation = Tuple[float, float]


def float_literal(value: float) -> str:
    return repr(float(value))


def reach_for_side(side: int) -> int:
    if side <= 0 or side % 2 == 0:
        raise ValueError(f"side must be a positive odd number, got {side}")
    return (side - 1) // 2


def fixture_lines(
    *,
    side: int,
    position: ScanPosition,
    orientation: Orientation,
    reach: float = TARGET_REACH,
    shape_ids: Sequence[int],
    target_indices: Iterable[int],
) -> List[str]:
    expected_count = side * side * side
    if len(shape_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} shape ids for side={side}, got {len(shape_ids)}"
        )

    target_lookup = frozenset(target_indices)
    lines = [
        "# Recorded from the current Minecraft world.",
        "# Unlisted blocks use default_shape. Order is x fastest, then z, then y.",
        "fixture_version 1",
        f"shape_catalog_version {DEFAULT_CATALOG.shape_catalog_version}",
        f"side {side}",
        "position " + " ".join(float_literal(value) for value in position),
        "orientation_yaw_pitch "
        + " ".join(float_literal(value) for value in orientation),
        f"reach {float_literal(reach)}",
        f"default_shape {SHAPE_EMPTY}",
        "",
    ]

    for index, shape_id in enumerate(shape_ids):
        if index in target_lookup:
            lines.append(f"target {index} {shape_id}")
        elif shape_id != SHAPE_EMPTY:
            lines.append(f"block {index} {shape_id}")

    lines.append("")
    return lines


def record_fixture(
    side: int,
    position: ScanPosition,
    orientation: Orientation,
    target_blocks: frozenset[str],
) -> Tuple[Path, int, int]:
    reach = reach_for_side(side)
    block_strings = []
    target_indices = []

    for index, (_block_pos, block_string) in enumerate(
        get_area(position, reach, await_region=True)
    ):
        block_strings.append(block_string)
        if block_id_literal(block_string) in target_blocks:
            target_indices.append(index)

    encoded = DEFAULT_CATALOG.encode_region(side, block_strings)
    lines = fixture_lines(
        side=side,
        position=position,
        orientation=orientation,
        reach=TARGET_REACH,
        shape_ids=encoded.shape_ids,
        target_indices=target_indices,
    )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"recorded_side_{side}.scan"
    temporary_path = fixture_path.with_suffix(".scan.tmp")
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    temporary_path.replace(fixture_path)

    non_empty_count = sum(shape_id != SHAPE_EMPTY for shape_id in encoded.shape_ids)
    return fixture_path, non_empty_count, len(target_indices)


def main() -> None:
    px, py, pz = m.player_position()
    position = (px, py + 1.62, pz)
    orientation = tuple(m.player_orientation())
    target_blocks = load_target_blocks(TARGET_CONFIG)

    m.echo(
        "Recording native scan fixtures at "
        f"{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}"
    )
    if not target_blocks:
        m.echo(f"Warning: no targets configured in {TARGET_CONFIG.name}")

    for side in SIDES:
        m.echo(f"Reading side={side} scan...")
        fixture_path, non_empty_count, target_count = record_fixture(
            side,
            position,
            orientation,
            target_blocks,
        )
        m.echo(
            f"Wrote {fixture_path.name}: "
            f"{non_empty_count} non-empty blocks, {target_count} targets"
        )
        if target_count == 0:
            m.echo(f"Warning: recorded_side_{side}.scan contains no target")


if __name__ == "__main__":
    main()
