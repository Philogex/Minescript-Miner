from __future__ import annotations

import math
import os
import statistics
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import minescript as m


PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PROJECT_DIR / "src"
TARGET_CONFIG = PROJECT_DIR / "targets.txt"

for path in (PROJECT_DIR, SRC_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from minescript_miner.adapter.target_pipeline import load_target_blocks
from minescript_miner.minescript.scanner import (
    ScanTimings,
    acquire_current_target,
)
from minescript_miner.minescript.runtime import query


REACH = 4.8
WARMUP_RUNS = 10
MEASURED_RUNS = 100


def percentile_nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("percentile fraction must be in (0, 1]")

    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def timing_values(timing: ScanTimings) -> dict[str, float]:
    return {
        "await_region": timing.area.await_region_ms,
        "prune_positions": timing.area.prune_positions_ms,
        "get_block_region": timing.area.region_read_ms,
        "extract_pruned_blocks": timing.area.region_extract_ms,
        "rebuild_cube": timing.area.cube_rebuild_ms,
        "area_total": timing.area.total_ms,
        "target_config": timing.target_config_ms,
        "target_match": timing.target_match_ms,
        "shape_encode": timing.shape_encode_ms,
        "native_call": timing.native_call_ms,
        "python_outside_native": timing.total_ms - timing.native_call_ms,
        "total": timing.total_ms,
    }


def scan(*, await_region: bool, target_blocks: frozenset[str]) -> ScanTimings:
    px, py, pz = query(m.player_position)
    yaw, pitch = query(m.player_orientation)
    timings = ScanTimings()
    acquire_current_target(
        (px, py + 1.62, pz),
        (yaw, pitch),
        REACH,
        target_blocks=target_blocks,
        await_region=await_region,
        timings=timings,
    )
    return timings


def collect_samples(
    scan_function: Callable[..., ScanTimings],
    *,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, list[float]]:
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be positive")

    for _ in range(warmup_runs):
        scan_function(await_region=False)

    samples: dict[str, list[float]] = {}
    for _ in range(measured_runs):
        for name, value in timing_values(
            scan_function(await_region=False)
        ).items():
            samples.setdefault(name, []).append(value)
    return samples


def summarize_samples(
    samples: dict[str, list[float]],
) -> dict[str, tuple[float, float]]:
    return {
        name: (
            statistics.median(values),
            percentile_nearest_rank(values, 0.95),
        )
        for name, values in samples.items()
    }


def run() -> None:
    target_blocks = load_target_blocks(TARGET_CONFIG)
    scan_loaded_targets = partial(
        scan,
        target_blocks=target_blocks,
    )

    initial = scan_loaded_targets(await_region=True)
    m.echo(
        "Pipeline initial await_loaded_region: "
        f"{initial.area.await_region_ms:.4f} ms"
    )

    samples = collect_samples(
        scan_loaded_targets,
        warmup_runs=WARMUP_RUNS,
        measured_runs=MEASURED_RUNS,
    )

    m.echo(
        "Pipeline benchmark: "
        f"warmup={WARMUP_RUNS}, measured={MEASURED_RUNS}, "
        f"await_region=False, reach={REACH}"
    )
    if not any(samples["native_call"]):
        m.echo(
            "Pipeline warning: no configured target was found; "
            "native_call was not exercised."
        )
    for name, (median_ms, p95_ms) in summarize_samples(samples).items():
        m.echo(
            f"{name}: median={median_ms:.4f} ms, "
            f"p95={p95_ms:.4f} ms"
        )


if __name__ == "__main__":
    run()
