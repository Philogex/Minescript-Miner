#!/usr/bin/env python3
"""Plot native aim path generator output for manual inspection."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from minescript_miner.aim import AimConfig, load_aim_config
from minescript_miner.adapter.native_bridge import (
    AimPoint,
    Orientation,
    TargetMetrics,
    generate_minimum_jerk_aim_path,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "aim-path" / "aim_path.png"
PathGenerator = Callable[
    [Orientation, TargetMetrics, AimConfig, float],
    tuple[AimPoint, ...],
]


@dataclass(frozen=True)
class GeneratedPath:
    name: str
    points: tuple[AimPoint, ...]


@dataclass(frozen=True)
class VelocitySegment:
    start_ms: float
    end_ms: float
    velocity_deg_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate native aim paths from synthetic target metrics and plot "
            "velocity plus remaining target delta over time."
        ),
    )
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--start-pitch", type=float, default=0.0)
    parser.add_argument("--target-yaw", type=float, default=25.0)
    parser.add_argument("--target-pitch", type=float, default=-8.0)
    parser.add_argument("--width-yaw", type=float, default=1.5)
    parser.add_argument("--width-pitch", type=float, default=1.0)
    parser.add_argument("--distance", type=float, default=4.0)
    parser.add_argument(
        "--angular-step-deg",
        type=float,
        default=0.15,
        help="Minecraft orientation quantization step in degrees.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "aim_config.txt",
        help="Aim config file.",
    )
    parser.add_argument(
        "--generator",
        action="append",
        choices=sorted(GENERATORS),
        help=(
            "Native path generator to plot. Can be passed more than once. "
            "Defaults to all registered generators."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG path to write. Use '-' to skip writing.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window.",
    )
    return parser.parse_args()


def native_minimum_jerk(
    start_orientation: Orientation,
    target: TargetMetrics,
    config: AimConfig,
    angular_step_deg: float,
) -> tuple[AimPoint, ...]:
    return generate_minimum_jerk_aim_path(
        start_orientation,
        target,
        angular_step_deg,
        config.fitts_a_ms,
        config.fitts_b_ms,
        config.min_duration_ms,
        config.max_duration_ms,
        config.sample_hz,
    )


GENERATORS: dict[str, PathGenerator] = {
    "minimum_jerk": native_minimum_jerk,
}


def shortest_yaw_delta(source: float, target: float) -> float:
    return ((target - source + 180.0) % 360.0) - 180.0


def unwrap_yaws(points: Sequence[AimPoint]) -> list[float]:
    if not points:
        return []

    unwrapped = [points[0].yaw]
    for point in points[1:]:
        unwrapped.append(unwrapped[-1] + shortest_yaw_delta(unwrapped[-1], point.yaw))
    return unwrapped


def angular_velocity_segments(
    points: Sequence[AimPoint],
) -> list[VelocitySegment]:
    if len(points) < 2:
        return []

    yaws = unwrap_yaws(points)
    pitches = [point.pitch for point in points]
    times = [point.t_ms for point in points]
    segments = []
    for index in range(1, len(points)):
        dt_s = (times[index] - times[index - 1]) / 1000.0
        if dt_s <= 0.0:
            continue
        dyaw = yaws[index] - yaws[index - 1]
        dpitch = pitches[index] - pitches[index - 1]
        segments.append(
            VelocitySegment(
                start_ms=times[index - 1],
                end_ms=times[index],
                velocity_deg_s=math.hypot(dyaw, dpitch) / dt_s,
            )
        )
    return segments


def target_delta(
    points: Sequence[AimPoint],
    target: TargetMetrics,
) -> tuple[list[float], list[float]]:
    times = []
    deltas = []
    for point in points:
        dyaw = shortest_yaw_delta(point.yaw, target.yaw)
        dpitch = target.pitch - point.pitch
        times.append(point.t_ms)
        deltas.append(math.hypot(dyaw, dpitch))
    return times, deltas


def target_axis_deltas(
    points: Sequence[AimPoint],
    target: TargetMetrics,
) -> tuple[list[float], list[float], list[float]]:
    times = []
    yaw_deltas = []
    pitch_deltas = []
    for point in points:
        times.append(point.t_ms)
        yaw_deltas.append(shortest_yaw_delta(point.yaw, target.yaw))
        pitch_deltas.append(target.pitch - point.pitch)
    return times, yaw_deltas, pitch_deltas


def generate_paths(
    generator_names: Sequence[str],
    start_orientation: Orientation,
    target: TargetMetrics,
    config: AimConfig,
    angular_step_deg: float,
) -> list[GeneratedPath]:
    paths = []
    for name in generator_names:
        generator = GENERATORS[name]
        paths.append(
            GeneratedPath(
                name=name,
                points=generator(
                    start_orientation,
                    target,
                    config,
                    angular_step_deg,
                ),
            )
        )
    return paths


def print_summary(
    paths: Sequence[GeneratedPath],
    target: TargetMetrics,
    config: AimConfig,
    angular_step_deg: float,
) -> None:
    print(
        "Target metrics: "
        f"yaw={target.yaw:.3f}, pitch={target.pitch:.3f}, "
        f"width_yaw={target.width_yaw:.3f}, "
        f"width_pitch={target.width_pitch:.3f}, "
        f"distance={target.distance:.3f}"
    )
    print(
        "Aim config: "
        f"sample_hz={config.sample_hz}, "
        f"fitts=({config.fitts_a_ms:.3f}, {config.fitts_b_ms:.3f}), "
        f"duration=[{config.min_duration_ms:.3f}, {config.max_duration_ms:.3f}], "
        f"angular_step_deg={angular_step_deg:.6f}"
    )
    for path in paths:
        if not path.points:
            print(f"{path.name}: no path samples generated")
            continue
        _delta_times, deltas = target_delta(path.points, target)
        velocity_segments = angular_velocity_segments(path.points)
        duration = path.points[-1].t_ms - path.points[0].t_ms
        final_delta = deltas[-1] if deltas else math.nan
        max_velocity = (
            max(segment.velocity_deg_s for segment in velocity_segments)
            if velocity_segments
            else 0.0
        )
        print(
            f"{path.name}: "
            f"samples={len(path.points)}, "
            f"duration_ms={duration:.3f}, "
            f"final_delta_deg={final_delta:.6f}, "
            f"max_velocity_deg_s={max_velocity:.3f}"
        )


def plot_paths(
    paths: Sequence[GeneratedPath],
    target: TargetMetrics,
    *,
    output: Path | None,
    show: bool,
) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=False,
        constrained_layout=True,
    )
    figure.suptitle("Native Aim Path Generators")

    velocity_axis = axes[0]
    delta_axis = axes[1]
    plotted_any = False
    for path in paths:
        if not path.points:
            continue

        velocity_segments = angular_velocity_segments(path.points)
        if velocity_segments:
            velocity_axis.hlines(
                [segment.velocity_deg_s for segment in velocity_segments],
                [segment.start_ms for segment in velocity_segments],
                [segment.end_ms for segment in velocity_segments],
                linewidth=2.0,
                label=path.name,
            )
            velocity_axis.scatter(
                [
                    (segment.start_ms + segment.end_ms) / 2.0
                    for segment in velocity_segments
                ],
                [segment.velocity_deg_s for segment in velocity_segments],
                s=20,
            )

        delta_times, deltas = target_delta(path.points, target)
        delta_axis.plot(
            delta_times,
            deltas,
            marker="o",
            label=f"{path.name} total",
        )
        axis_times, yaw_deltas, pitch_deltas = target_axis_deltas(path.points, target)
        delta_axis.plot(
            axis_times,
            [abs(value) for value in yaw_deltas],
            linestyle="--",
            alpha=0.65,
            label=f"{path.name} yaw",
        )
        delta_axis.plot(
            axis_times,
            [abs(value) for value in pitch_deltas],
            linestyle=":",
            alpha=0.65,
            label=f"{path.name} pitch",
        )
        plotted_any = True

    velocity_axis.set_title("Angular Velocity")
    velocity_axis.set_xlabel("time [ms]")
    velocity_axis.set_ylabel("velocity [deg/s]")
    velocity_axis.grid(True, alpha=0.3)
    handles, labels = velocity_axis.get_legend_handles_labels()
    if handles:
        velocity_axis.legend(handles, labels)

    delta_axis.set_title("Delta To Target")
    delta_axis.set_xlabel("time [ms]")
    delta_axis.set_ylabel("remaining delta [deg]")
    delta_axis.grid(True, alpha=0.3)
    handles, labels = delta_axis.get_legend_handles_labels()
    if handles:
        delta_axis.legend(handles, labels)

    if not plotted_any:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "no path samples",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        print(f"Wrote {output}")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_aim_config(args.config)
    target = TargetMetrics(
        yaw=args.target_yaw,
        pitch=args.target_pitch,
        width_yaw=args.width_yaw,
        width_pitch=args.width_pitch,
        distance=args.distance,
    )
    generator_names = args.generator if args.generator is not None else sorted(GENERATORS)
    paths = generate_paths(
        generator_names,
        (args.start_yaw, args.start_pitch),
        target,
        config,
        args.angular_step_deg,
    )
    print_summary(paths, target, config, args.angular_step_deg)

    output = None if str(args.output) == "-" else args.output
    if output is None and not args.show:
        return
    plot_paths(paths, target, output=output, show=args.show)


if __name__ == "__main__":
    main()
