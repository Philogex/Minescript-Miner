"""Pure Python adapter between encoded Python data and the native module."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import _minescript_miner_native as native


ScanPosition = Tuple[float, float, float]
Orientation = Tuple[float, float]


@dataclass(frozen=True)
class TargetMetrics:
    yaw: float
    pitch: float
    width_yaw: float
    width_pitch: float
    distance: float


@dataclass(frozen=True)
class AimPoint:
    yaw: float
    pitch: float
    t_ms: float


def _uint16_payload(values: Sequence[int]):
    if isinstance(values, array):
        if values.typecode != "H":
            raise TypeError(
                f"Expected array('H') for compact uint16 payload, "
                f"got array({values.typecode!r})"
            )
        return values.tobytes()
    if isinstance(values, (bytes, bytearray)):
        return bytes(values)
    return values


def acquire_target(
    position: ScanPosition,
    orientation: Orientation,
    shape_catalog_version: int,
    side: int,
    reach: float,
    shape_ids: Sequence[int],
    target_indices: Sequence[int],
) -> Optional[Orientation]:
    """Return the nearest visible target as Minecraft yaw and pitch."""

    result = native.acquire_target(
        position,
        orientation,
        shape_catalog_version,
        side,
        reach,
        _uint16_payload(shape_ids),
        _uint16_payload(target_indices),
    )
    if result is None:
        return None

    yaw, pitch = result
    return float(yaw), float(pitch)


def acquire_target_metrics(
    position: ScanPosition,
    orientation: Orientation,
    shape_catalog_version: int,
    side: int,
    reach: float,
    shape_ids: Sequence[int],
    target_indices: Sequence[int],
) -> Optional[TargetMetrics]:
    """Return target orientation plus local visible aim width and distance."""

    result = native.acquire_target_metrics(
        position,
        orientation,
        shape_catalog_version,
        side,
        reach,
        _uint16_payload(shape_ids),
        _uint16_payload(target_indices),
    )
    if result is None:
        return None

    yaw, pitch, width_yaw, width_pitch, distance = result
    return TargetMetrics(
        float(yaw),
        float(pitch),
        float(width_yaw),
        float(width_pitch),
        float(distance),
    )


def _target_metrics_payload(metrics: TargetMetrics):
    return (
        metrics.yaw,
        metrics.pitch,
        metrics.width_yaw,
        metrics.width_pitch,
        metrics.distance,
    )


def generate_minimum_jerk_aim_path(
    start_orientation: Orientation,
    target_metrics: TargetMetrics,
    angular_step_deg: float,
    fitts_a_ms: float,
    fitts_b_ms: float,
    min_duration_ms: float,
    max_duration_ms: float,
    sample_hz: int,
) -> Tuple[AimPoint, ...]:
    """Return a native-generated minimum-jerk aim path.

    The current native implementation is intentionally a placeholder with the
    final API shape; it returns valid path samples but not the final motion
    model yet.
    """

    result = native.generate_minimum_jerk_aim_path(
        start_orientation,
        _target_metrics_payload(target_metrics),
        float(angular_step_deg),
        float(fitts_a_ms),
        float(fitts_b_ms),
        float(min_duration_ms),
        float(max_duration_ms),
        int(sample_hz),
    )
    return tuple(
        AimPoint(float(yaw), float(pitch), float(t_ms))
        for yaw, pitch, t_ms in result
    )
