#!/usr/bin/env python3
"""Precompute Balabit mouse trajectory feature references.

The Balabit data set does not include explicit UI targets.  For this analysis
each pause-separated movement segment treats its final cursor position as the
intended endpoint.  That keeps the kinematic and geometric features useful as a
human-motion reference, while Fitts values remain model-based approximations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from minescript_miner.adapter.native_bridge import AimPoint, TargetMetrics
from minescript_miner.aim_analysis import AimPathFeatures, compute_aim_path_features


DEFAULT_DATASET = PROJECT_ROOT.parent / "Mouse-Dynamics-Challenge"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "build" / "aim-analysis" / "balabit"
FEATURE_NAMES = tuple(AimPathFeatures.__dataclass_fields__)


@dataclass(frozen=True)
class MouseSample:
    t_ms: float
    x: float
    y: float
    button: str
    state: str


@dataclass(frozen=True)
class SegmentFeatures:
    split: str
    user: str
    session: str
    segment_index: int
    sample_count: int
    start_ms: float
    end_ms: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    distance_px: float
    fitts_mt: float
    fitts_id: float
    fitts_predicted_mt: float
    fitts_residual: float
    fitts_residual_ratio: float
    sub_peak_count: int
    sub_primary_amp_ratio: float
    sub_correction_onset: float
    sub_interpeak_cv: float
    sub_peak_speed_ratio: float
    geo_path_efficiency: float
    geo_max_deviation: float
    geo_angular_dev_at_peak: float
    geo_curvature_integral: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Balabit Mouse Dynamics Challenge sessions once and write "
            "feature CSV plus compact summary JSON for later plotting."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path containing training_files/ and test_files/.",
    )
    parser.add_argument(
        "--split",
        choices=("training_files", "test_files", "all"),
        default="training_files",
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for features.csv and summary.json.",
    )
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--min-distance-px", type=float, default=16.0)
    parser.add_argument("--pause-threshold-ms", type=float, default=250.0)
    parser.add_argument(
        "--target-width-px",
        type=float,
        default=16.0,
        help="Approximate endpoint width used for Fitts features.",
    )
    parser.add_argument("--fitts-a-ms", type=float, default=0.0)
    parser.add_argument("--fitts-b-ms", type=float, default=100.0)
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Optional cap for quick experiments. 0 means no cap.",
    )
    return parser.parse_args()


def iter_session_paths(dataset: Path, split: str) -> Iterable[tuple[str, Path]]:
    splits = ("training_files", "test_files") if split == "all" else (split,)
    for split_name in splits:
        split_dir = dataset / split_name
        if not split_dir.is_dir():
            continue
        yield from (
            (split_name, path)
            for path in sorted(split_dir.glob("user*/session_*"))
            if path.is_file()
        )


def _row_time_ms(row: dict[str, str]) -> float | None:
    for key in ("client timestamp", "record timestamp"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            return float(raw) * 1000.0
        except ValueError:
            continue
    return None


def read_session(path: Path) -> list[MouseSample]:
    samples: list[MouseSample] = []
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            t_ms = _row_time_ms(row)
            if t_ms is None:
                continue
            try:
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, ValueError):
                continue
            sample = MouseSample(
                t_ms=t_ms,
                x=x,
                y=y,
                button=row.get("button", ""),
                state=row.get("state", ""),
            )
            if samples and math.isclose(sample.t_ms, samples[-1].t_ms):
                samples[-1] = sample
            elif not samples or sample.t_ms > samples[-1].t_ms:
                samples.append(sample)
    return samples


def split_segments(
    samples: Sequence[MouseSample],
    *,
    pause_threshold_ms: float,
) -> Iterable[list[MouseSample]]:
    current: list[MouseSample] = []
    previous_time: float | None = None
    for sample in samples:
        if (
            current
            and previous_time is not None
            and sample.t_ms - previous_time > pause_threshold_ms
        ):
            yield current
            current = []
        current.append(sample)
        previous_time = sample.t_ms
    if current:
        yield current


def segment_distance(segment: Sequence[MouseSample]) -> float:
    return math.hypot(segment[-1].x - segment[0].x, segment[-1].y - segment[0].y)


def segment_to_points(segment: Sequence[MouseSample]) -> tuple[AimPoint, ...]:
    start_time = segment[0].t_ms
    return tuple(
        AimPoint(sample.x, sample.y, sample.t_ms - start_time)
        for sample in segment
    )


def analyze_segment(
    split: str,
    session_path: Path,
    segment_index: int,
    segment: Sequence[MouseSample],
    *,
    target_width_px: float,
    fitts_a_ms: float,
    fitts_b_ms: float,
) -> SegmentFeatures:
    points = segment_to_points(segment)
    target = TargetMetrics(
        yaw=segment[-1].x,
        pitch=segment[-1].y,
        width_yaw=target_width_px,
        width_pitch=target_width_px,
        distance=0.0,
    )
    features = compute_aim_path_features(
        points,
        target,
        fitts_a_ms=fitts_a_ms,
        fitts_b_ms=fitts_b_ms,
        fallback_width_deg=target_width_px,
        wrap_yaw=False,
    )
    return SegmentFeatures(
        split=split,
        user=session_path.parent.name,
        session=session_path.name,
        segment_index=segment_index,
        sample_count=len(segment),
        start_ms=segment[0].t_ms,
        end_ms=segment[-1].t_ms,
        start_x=segment[0].x,
        start_y=segment[0].y,
        end_x=segment[-1].x,
        end_y=segment[-1].y,
        distance_px=segment_distance(segment),
        **asdict(features),
    )


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(rows: Sequence[SegmentFeatures]) -> dict[str, object]:
    features: dict[str, object] = {}
    for name in FEATURE_NAMES:
        values = [
            float(getattr(row, name))
            for row in rows
            if math.isfinite(float(getattr(row, name)))
        ]
        sorted_values = sorted(values)
        features[name] = {
            "count": len(values),
            "nan_count": len(rows) - len(values),
            "mean": statistics.fmean(values) if values else math.nan,
            "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "p05": percentile(sorted_values, 0.05),
            "p25": percentile(sorted_values, 0.25),
            "p50": percentile(sorted_values, 0.50),
            "p75": percentile(sorted_values, 0.75),
            "p95": percentile(sorted_values, 0.95),
        }
    return features


def write_features_csv(path: Path, rows: Sequence[SegmentFeatures]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(SegmentFeatures.__dataclass_fields__)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    rows: list[SegmentFeatures] = []
    scanned_sessions = 0
    skipped_segments = 0
    for split, path in iter_session_paths(args.dataset, args.split):
        scanned_sessions += 1
        samples = read_session(path)
        for segment_index, segment in enumerate(
            split_segments(samples, pause_threshold_ms=args.pause_threshold_ms)
        ):
            if len(segment) < args.min_samples:
                skipped_segments += 1
                continue
            if segment_distance(segment) < args.min_distance_px:
                skipped_segments += 1
                continue
            rows.append(
                analyze_segment(
                    split,
                    path,
                    segment_index,
                    segment,
                    target_width_px=args.target_width_px,
                    fitts_a_ms=args.fitts_a_ms,
                    fitts_b_ms=args.fitts_b_ms,
                )
            )
            if args.max_segments > 0 and len(rows) >= args.max_segments:
                break
        if args.max_segments > 0 and len(rows) >= args.max_segments:
            break

    output_dir = args.output_dir
    features_path = output_dir / "features.csv"
    summary_path = output_dir / "summary.json"
    write_features_csv(features_path, rows)
    summary = {
        "source": "balabit/Mouse-Dynamics-Challenge",
        "dataset": str(args.dataset),
        "split": args.split,
        "assumptions": {
            "target": "pause-separated segment endpoint",
            "coordinate_space": "screen pixels",
            "wrap_yaw": False,
            "duplicate_timestamps": "last sample wins",
        },
        "parameters": {
            "min_samples": args.min_samples,
            "min_distance_px": args.min_distance_px,
            "pause_threshold_ms": args.pause_threshold_ms,
            "target_width_px": args.target_width_px,
            "fitts_a_ms": args.fitts_a_ms,
            "fitts_b_ms": args.fitts_b_ms,
        },
        "session_count": scanned_sessions,
        "segment_count": len(rows),
        "skipped_segment_count": skipped_segments,
        "features": summarize(rows),
    }
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=True)
        file.write("\n")

    print(f"Wrote {features_path}")
    print(f"Wrote {summary_path}")
    print(
        "Analyzed "
        f"{len(rows)} segments from {scanned_sessions} sessions "
        f"({skipped_segments} skipped)."
    )


if __name__ == "__main__":
    main()
