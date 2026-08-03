"""Pure Python adapter between encoded Python data and the native module."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import secrets
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
    target_block: Tuple[int, int, int] | None = None
    face_id: str | None = None
    hit_point: Tuple[float, float, float] | None = None
    block_state_before: str | None = None
    neighbors: Tuple[Tuple[int, int, int, str], ...] = ()
    effective_width: float = 0.0
    visible_components: Tuple[Tuple[ScanPosition, ...], ...] = ()


@dataclass(frozen=True)
class AimPoint:
    yaw: float
    pitch: float
    t_ms: float


@dataclass(frozen=True)
class GeometryFeedbackDiagnostics:
    motor_target_yaw: float
    motor_target_pitch: float
    applied_margin_steps: float
    anchor_component_index: int
    directional_width_steps: float
    s_enter_steps: float
    s_anchor_steps: float
    s_exit_steps: float
    primary_endpoint_steps: float
    first_feedback_observation_ms: float
    first_feedback_latency_ms: float
    feedback_check_count: int
    correction_count: int
    first_visible_entry_ms: float
    first_safe_entry_ms: float
    visible_entry_count: int
    visible_exit_count: int
    safe_entry_count: int
    safe_exit_count: int
    final_visible: bool
    final_safe: bool


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
    """Return the target plus its complete visible convex components."""

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

    (
        yaw,
        pitch,
        width_yaw,
        width_pitch,
        distance,
        effective_width,
        target_block,
        face_id,
        hit_point,
        visible_components,
    ) = result
    return TargetMetrics(
        yaw=float(yaw),
        pitch=float(pitch),
        width_yaw=float(width_yaw),
        width_pitch=float(width_pitch),
        distance=float(distance),
        effective_width=float(effective_width),
        target_block=tuple(int(coordinate) for coordinate in target_block),
        face_id=str(face_id) or None,
        hit_point=tuple(float(coordinate) for coordinate in hit_point),
        visible_components=tuple(
            tuple(
                tuple(float(coordinate) for coordinate in direction)
                for direction in component
            )
            for component in visible_components
        ),
    )


def _target_metrics_payload(metrics: TargetMetrics):
    return (
        metrics.yaw,
        metrics.pitch,
        metrics.width_yaw,
        metrics.width_pitch,
        metrics.distance,
        metrics.effective_width,
    )


def _resolved_seed(seed: int | None) -> int:
    resolved = secrets.randbits(64) if seed is None else int(seed)
    if not 0 <= resolved <= (1 << 64) - 1:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    return resolved


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


def generate_sigmadrift_aim_path(
    start_orientation: Orientation,
    target_metrics: TargetMetrics,
    angular_step_deg: float,
    config_values: Sequence[float],
    seed: int | None = None,
) -> Tuple[AimPoint, ...]:
    """Return a native-generated SigmaDrift aim path."""

    result = native.generate_sigmadrift_aim_path(
        start_orientation,
        _target_metrics_payload(target_metrics),
        float(angular_step_deg),
        tuple(float(value) for value in config_values),
        _resolved_seed(seed),
    )
    return tuple(
        AimPoint(float(yaw), float(pitch), float(t_ms))
        for yaw, pitch, t_ms in result
    )


def generate_geometry_feedback_sigmadrift_aim_path(
    start_orientation: Orientation,
    target_metrics: TargetMetrics,
    angular_step_deg: float,
    config_values: Sequence[float],
    feedback_config_values: Sequence[float],
    seed: int | None = None,
) -> Tuple[AimPoint, ...]:
    """Return a native-generated geometry-feedback SigmaDrift path."""

    result = native.generate_geometry_feedback_sigmadrift_aim_path(
        start_orientation,
        _target_metrics_payload(target_metrics),
        target_metrics.visible_components,
        float(angular_step_deg),
        tuple(float(value) for value in config_values),
        tuple(feedback_config_values),
        _resolved_seed(seed),
    )
    return tuple(
        AimPoint(float(yaw), float(pitch), float(t_ms))
        for yaw, pitch, t_ms in result
    )


def generate_geometry_feedback_sigmadrift_aim_path_with_diagnostics(
    start_orientation: Orientation,
    target_metrics: TargetMetrics,
    angular_step_deg: float,
    config_values: Sequence[float],
    feedback_config_values: Sequence[float],
    seed: int | None = None,
) -> tuple[Tuple[AimPoint, ...], GeometryFeedbackDiagnostics]:
    """Return a geometry-feedback path plus exact controller diagnostics."""

    result, raw_diagnostics = (
        native.generate_geometry_feedback_sigmadrift_aim_path_with_diagnostics(
            start_orientation,
            _target_metrics_payload(target_metrics),
            target_metrics.visible_components,
            float(angular_step_deg),
            tuple(float(value) for value in config_values),
            tuple(feedback_config_values),
            _resolved_seed(seed),
        )
    )
    path = tuple(
        AimPoint(float(yaw), float(pitch), float(t_ms))
        for yaw, pitch, t_ms in result
    )
    diagnostics = GeometryFeedbackDiagnostics(
        motor_target_yaw=float(raw_diagnostics["motor_target_yaw"]),
        motor_target_pitch=float(raw_diagnostics["motor_target_pitch"]),
        applied_margin_steps=float(raw_diagnostics["applied_margin_steps"]),
        anchor_component_index=int(raw_diagnostics["anchor_component_index"]),
        directional_width_steps=float(
            raw_diagnostics["directional_width_steps"]
        ),
        s_enter_steps=float(raw_diagnostics["s_enter_steps"]),
        s_anchor_steps=float(raw_diagnostics["s_anchor_steps"]),
        s_exit_steps=float(raw_diagnostics["s_exit_steps"]),
        primary_endpoint_steps=float(
            raw_diagnostics["primary_endpoint_steps"]
        ),
        first_feedback_observation_ms=float(
            raw_diagnostics["first_feedback_observation_ms"]
        ),
        first_feedback_latency_ms=float(
            raw_diagnostics["first_feedback_latency_ms"]
        ),
        feedback_check_count=int(raw_diagnostics["feedback_check_count"]),
        correction_count=int(raw_diagnostics["correction_count"]),
        first_visible_entry_ms=float(raw_diagnostics["first_visible_entry_ms"]),
        first_safe_entry_ms=float(raw_diagnostics["first_safe_entry_ms"]),
        visible_entry_count=int(raw_diagnostics["visible_entry_count"]),
        visible_exit_count=int(raw_diagnostics["visible_exit_count"]),
        safe_entry_count=int(raw_diagnostics["safe_entry_count"]),
        safe_exit_count=int(raw_diagnostics["safe_exit_count"]),
        final_visible=bool(raw_diagnostics["final_visible"]),
        final_safe=bool(raw_diagnostics["final_safe"]),
    )
    return path, diagnostics
