from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import minescript as m


PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PROJECT_DIR / "src"
TARGET_CONFIG = PROJECT_DIR / "targets.txt"

for path in (PROJECT_DIR, SRC_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from minescript_miner.minescript.io import ScanTimings, acquire_current_target


REACH = 4.8
WARMUP_RUNS = 5
MEASURED_RUNS = 50


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def timing_values(timing: ScanTimings) -> dict[str, float]:
    return {
        "await_region": timing.area.await_region_ms,
        "prune_positions": timing.area.prune_positions_ms,
        "get_block_region": timing.area.region_read_ms,
        "extract_pruned_blocks": timing.area.region_extract_ms,
        "rebuild_cube": timing.area.cube_rebuild_ms,
        "target_config": timing.target_config_ms,
        "target_match": timing.target_match_ms,
        "shape_encode": timing.shape_encode_ms,
        "native_call": timing.native_call_ms,
        "python_outside_native": timing.total_ms - timing.native_call_ms,
        "total": timing.total_ms,
    }


def scan(*, await_region: bool) -> ScanTimings:
    px, py, pz = m.player_position()
    yaw, pitch = m.player_orientation()
    timings = ScanTimings()
    acquire_current_target(
        (px, py + 1.62, pz),
        (yaw, pitch),
        REACH,
        target_config=TARGET_CONFIG,
        await_region=await_region,
        timings=timings,
    )
    return timings


def run() -> None:
    initial = scan(await_region=True)
    m.echo(
        "Pipeline initial await_loaded_region: "
        f"{initial.area.await_region_ms:.4f} ms"
    )

    for _ in range(WARMUP_RUNS):
        scan(await_region=False)

    samples: dict[str, list[float]] = {}
    for _ in range(MEASURED_RUNS):
        for name, value in timing_values(scan(await_region=False)).items():
            samples.setdefault(name, []).append(value)

    m.echo(f"Pipeline benchmark: {MEASURED_RUNS} scans, reach={REACH}")
    for name, values in samples.items():
        m.echo(
            f"{name}: median={statistics.median(values):.4f} ms, "
            f"p95={percentile(values, 0.95):.4f} ms"
        )


if __name__ == "__main__":
    run()
