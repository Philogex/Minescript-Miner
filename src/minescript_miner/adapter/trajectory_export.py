"""Export generated aim paths as explicitly synthetic Minecraft DAQ sessions."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from .native_bridge import AimPoint, Orientation, TargetMetrics


DAQ_SCHEMA_VERSION = 1
METADATA_SCHEMA_VERSION = 1


def _session_id() -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()


def _session_directory(output_root: Path, session_id: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    directory = output_root / f"synthetic-{timestamp}-{session_id[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _csv_writer(
    path: Path, fieldnames: Sequence[str]
) -> tuple[csv.DictWriter, TextIO]:
    file = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return writer, file


def _orientation_steps(previous: AimPoint, current: AimPoint, step_deg: float) -> tuple[float, float]:
    yaw_delta = ((current.yaw - previous.yaw + 180.0) % 360.0) - 180.0
    return yaw_delta / step_deg, (current.pitch - previous.pitch) / step_deg


def write_synthetic_trajectory_session(
    output_root: Path,
    points: Sequence[AimPoint],
    start_orientation: Orientation,
    target_metrics: TargetMetrics,
    *,
    generator: str,
    angular_step_deg: float,
    generator_config: Mapping[str, object] | None = None,
) -> Path:
    """Write one generated path in the DAQ CSV shape plus synthetic metadata.

    This is an analysis export, not an observation of Minecraft input. The
    metadata file documents every placeholder and derived field so downstream
    analysis can filter it independently from real DAQ recordings.
    """

    if not points:
        raise ValueError("cannot export an empty aim path")
    if angular_step_deg <= 0.0:
        raise ValueError("angular_step_deg must be positive")

    session_id = _session_id()
    session_directory = _session_directory(Path(output_root), session_id)
    event_time_ns = time.time_ns()
    final_t_ms = points[-1].t_ms
    target_block = target_metrics.target_block or (0, 0, 0)
    hit_point = target_metrics.hit_point
    neighbors = [
        {"dx": dx, "dy": dy, "dz": dz, "state": state}
        for dx, dy, dz, state in target_metrics.neighbors
    ]

    event_fields = (
        "schema_version",
        "session_id",
        "event_id",
        "event_time_ns",
        "target_x",
        "target_y",
        "target_z",
        "face_id",
        "hit_x",
        "hit_y",
        "hit_z",
        "block_state_before",
        "block_state_after",
        "neighbors_json",
    )
    state_fields = (
        "schema_version",
        "session_id",
        "event_id",
        "sample_time_ns",
        "event_time_ns",
        "relative_ms",
        "yaw",
        "pitch",
        "player_x",
        "player_y",
        "player_z",
        "fov",
        "gui_scale",
        "fps_estimate",
        "sensitivity",
    )
    mouse_fields = (
        "schema_version",
        "session_id",
        "event_id",
        "sample_time_ns",
        "event_time_ns",
        "relative_ms",
        "mouse_dx",
        "mouse_dy",
    )

    event_writer, event_file = _csv_writer(
        session_directory / "events.csv", event_fields
    )
    state_writer, state_file = _csv_writer(
        session_directory / "state_samples.csv", state_fields
    )
    mouse_writer, mouse_file = _csv_writer(
        session_directory / "mouse_trajectory.csv", mouse_fields
    )
    try:
        event_writer.writerow(
            {
                "schema_version": DAQ_SCHEMA_VERSION,
                "session_id": session_id,
                "event_id": 1,
                "event_time_ns": event_time_ns,
                "target_x": target_block[0],
                "target_y": target_block[1],
                "target_z": target_block[2],
                "face_id": target_metrics.face_id or "",
                "hit_x": "" if hit_point is None else hit_point[0],
                "hit_y": "" if hit_point is None else hit_point[1],
                "hit_z": "" if hit_point is None else hit_point[2],
                "block_state_before": target_metrics.block_state_before
                or "synthetic:unobserved",
                "block_state_after": "synthetic:complete",
                "neighbors_json": json.dumps(neighbors, separators=(",", ":")),
            }
        )
        for point in points:
            relative_ms = point.t_ms - final_t_ms
            state_writer.writerow(
                {
                    "schema_version": DAQ_SCHEMA_VERSION,
                    "session_id": session_id,
                    "event_id": 1,
                    "sample_time_ns": event_time_ns + round(relative_ms * 1_000_000),
                    "event_time_ns": event_time_ns,
                    "relative_ms": relative_ms,
                    "yaw": point.yaw,
                    "pitch": point.pitch,
                    "player_x": "nan",
                    "player_y": "nan",
                    "player_z": "nan",
                    "fov": "nan",
                    "gui_scale": 0,
                    "fps_estimate": "nan",
                    "sensitivity": "nan",
                }
            )
        for previous, current in zip(points, points[1:]):
            relative_ms = current.t_ms - final_t_ms
            mouse_dx, mouse_dy = _orientation_steps(
                previous, current, angular_step_deg
            )
            mouse_writer.writerow(
                {
                    "schema_version": DAQ_SCHEMA_VERSION,
                    "session_id": session_id,
                    "event_id": 1,
                    "sample_time_ns": event_time_ns + round(relative_ms * 1_000_000),
                    "event_time_ns": event_time_ns,
                    "relative_ms": relative_ms,
                    "mouse_dx": mouse_dx,
                    "mouse_dy": mouse_dy,
                }
            )
    finally:
        event_file.close()
        state_file.close()
        mouse_file.close()

    metadata = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "source": "minescript-miner-synthetic",
        "generator": generator,
        "generator_config": dict(generator_config or {}),
        "path": {
            "point_count": len(points),
            "duration_ms": final_t_ms - points[0].t_ms,
            "start_orientation": {
                "yaw": start_orientation[0],
                "pitch": start_orientation[1],
            },
            "target_metrics": {
                "yaw": target_metrics.yaw,
                "pitch": target_metrics.pitch,
                "width_yaw": target_metrics.width_yaw,
                "width_pitch": target_metrics.width_pitch,
                "distance": target_metrics.distance,
            },
            "angular_step_deg": angular_step_deg,
        },
        "synthetic_fields": {
            "target_block_and_hit": (
                "solver result" if target_metrics.target_block and hit_point else "placeholder"
            ),
            "block_state_before": (
                "scanner region" if target_metrics.block_state_before else "unobserved"
            ),
            "neighbors": (
                "scanner region; unavailable entries are synthetic:unobserved"
                if target_metrics.neighbors
                else "unobserved"
            ),
            "player_position": "unobserved nan",
            "fov_gui_scale_fps_sensitivity": "unobserved nan or sentinel",
            "mouse_dx_dy": (
                "orientation-step equivalents derived from consecutive aim points; "
                "not raw MouseHandler input"
            ),
        },
    }
    with (session_directory / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
        file.write("\n")
    return session_directory
