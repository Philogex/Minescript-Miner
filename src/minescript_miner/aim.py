"""Aim path configuration and generation helpers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Union

from minescript_miner.adapter.native_bridge import (
    AimPoint,
    Orientation,
    TargetMetrics,
    generate_minimum_jerk_aim_path as _generate_minimum_jerk_aim_path,
    generate_sigmadrift_aim_path as _generate_sigmadrift_aim_path,
)


DEFAULT_AIM_CONFIG = Path("aim_config.txt")
DEFAULT_FALLBACK_ANGULAR_STEP_DEG = 0.15
SUPPORTED_AIM_MODELS = frozenset({"minimum_jerk", "sigmadrift"})
IMPLEMENTED_AIM_MODELS = frozenset({"minimum_jerk", "sigmadrift"})


@dataclass(frozen=True)
class MinimumJerkConfig:
    fitts_a_ms: float = 80.0
    fitts_b_ms: float = 110.0
    min_duration_ms: float = 60.0
    max_duration_ms: float = 450.0
    sample_hz: int = 120
    correction_probability: float = 0.6
    max_corrections: int = 1


@dataclass(frozen=True)
class SigmaDriftConfig:
    # Parameter set follows ck0i/SigmaDrift's motor_synergy::config.
    fitts_a: float = 50.0
    fitts_b: float = 150.0
    target_width: float = 20.0
    undershoot_min: float = 0.92
    undershoot_max: float = 0.97
    peak_time_ratio: float = 0.35
    primary_sigma_min: float = 0.18
    primary_sigma_max: float = 0.28
    overshoot_prob: float = 0.15
    overshoot_min: float = 1.02
    overshoot_max: float = 1.08
    correction_sigma_min: float = 0.12
    correction_sigma_max: float = 0.20
    second_correction_prob: float = 0.25
    curvature_scale: float = 0.025
    ou_theta: float = 3.5
    ou_sigma: float = 1.2
    tremor_freq_min: float = 8.0
    tremor_freq_max: float = 12.0
    tremor_amp_min: float = 0.15
    tremor_amp_max: float = 0.55
    sdn_k: float = 0.04
    sample_dt_mean: float = 7.8
    gamma_shape: float = 3.5


@dataclass(frozen=True)
class AimConfig:
    aim_model: str = "minimum_jerk"
    fallback_angular_step_deg: float = DEFAULT_FALLBACK_ANGULAR_STEP_DEG
    minimum_jerk: MinimumJerkConfig = field(default_factory=MinimumJerkConfig)
    sigmadrift: SigmaDriftConfig = field(default_factory=SigmaDriftConfig)


def _parse_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc
    return parsed


def _parse_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    return parsed


def _parse_str(value: str, _name: str) -> str:
    return value


Parser = Callable[[str, str], object]

GLOBAL_PARSERS: Mapping[str, Parser] = {
    "aim_model": _parse_str,
    "fallback_angular_step_deg": _parse_float,
}

MINIMUM_JERK_PARSERS: Mapping[str, Parser] = {
    "fitts_a_ms": _parse_float,
    "fitts_b_ms": _parse_float,
    "min_duration_ms": _parse_float,
    "max_duration_ms": _parse_float,
    "sample_hz": _parse_int,
    "correction_probability": _parse_float,
    "max_corrections": _parse_int,
}

SIGMADRIFT_PARSERS: Mapping[str, Parser] = {
    "fitts_a": _parse_float,
    "fitts_b": _parse_float,
    "target_width": _parse_float,
    "undershoot_min": _parse_float,
    "undershoot_max": _parse_float,
    "peak_time_ratio": _parse_float,
    "primary_sigma_min": _parse_float,
    "primary_sigma_max": _parse_float,
    "overshoot_prob": _parse_float,
    "overshoot_min": _parse_float,
    "overshoot_max": _parse_float,
    "correction_sigma_min": _parse_float,
    "correction_sigma_max": _parse_float,
    "second_correction_prob": _parse_float,
    "curvature_scale": _parse_float,
    "ou_theta": _parse_float,
    "ou_sigma": _parse_float,
    "tremor_freq_min": _parse_float,
    "tremor_freq_max": _parse_float,
    "tremor_amp_min": _parse_float,
    "tremor_amp_max": _parse_float,
    "sdn_k": _parse_float,
    "sample_dt_mean": _parse_float,
    "gamma_shape": _parse_float,
}

SECTION_PARSERS: Mapping[str, Mapping[str, Parser]] = {
    "global": GLOBAL_PARSERS,
    "minimum_jerk": MINIMUM_JERK_PARSERS,
    "sigmadrift": SIGMADRIFT_PARSERS,
}


def _section_display(section: str) -> str:
    return "top level" if section == "global" else f"{section} block"


def _parse_config_values(path: Path) -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {
        "global": {},
        "minimum_jerk": {},
        "sigmadrift": {},
    }
    section = "global"

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith("["):
            section = line[:-1].strip()
            if section not in SECTION_PARSERS or section == "global":
                raise ValueError(
                    f"{path}:{line_number}: unknown aim config block {section!r}"
                )
            continue
        if line == "]":
            section = "global"
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected 'name: value'")

        name, raw_value = line.split(":", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        parsers = SECTION_PARSERS[section]
        if name not in parsers:
            legacy_parser = MINIMUM_JERK_PARSERS.get(name)
            if section == "global" and legacy_parser is not None:
                values["minimum_jerk"][name] = legacy_parser(raw_value, name)
                continue
            raise ValueError(
                f"{path}:{line_number}: unknown aim config key {name!r} "
                f"in {_section_display(section)}"
            )
        values[section][name] = parsers[name](raw_value, name)

    return values


def _build_config(values: Mapping[str, Mapping[str, object]]) -> AimConfig:
    return AimConfig(
        **values["global"],
        minimum_jerk=MinimumJerkConfig(**values["minimum_jerk"]),
        sigmadrift=SigmaDriftConfig(**values["sigmadrift"]),
    )


def _validate_range_order(
    lower: float,
    upper: float,
    lower_name: str,
    upper_name: str,
) -> None:
    if upper < lower:
        raise ValueError(f"{upper_name} must be >= {lower_name}")


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_config(config: AimConfig) -> None:
    if config.aim_model not in SUPPORTED_AIM_MODELS:
        raise ValueError(f"unsupported aim_model {config.aim_model!r}")
    if config.fallback_angular_step_deg <= 0.0:
        raise ValueError("fallback_angular_step_deg must be positive")

    minimum = config.minimum_jerk
    if minimum.sample_hz <= 0:
        raise ValueError("minimum_jerk.sample_hz must be positive")
    _validate_range_order(
        minimum.min_duration_ms,
        minimum.max_duration_ms,
        "minimum_jerk.min_duration_ms",
        "minimum_jerk.max_duration_ms",
    )
    _validate_probability(
        minimum.correction_probability,
        "minimum_jerk.correction_probability",
    )
    if minimum.max_corrections < 0:
        raise ValueError("minimum_jerk.max_corrections must be >= 0")

    sigma = config.sigmadrift
    if sigma.target_width <= 0.0:
        raise ValueError("sigmadrift.target_width must be positive")
    if sigma.sample_dt_mean <= 0.0:
        raise ValueError("sigmadrift.sample_dt_mean must be positive")
    if sigma.gamma_shape <= 0.0:
        raise ValueError("sigmadrift.gamma_shape must be positive")
    for name in ("overshoot_prob", "second_correction_prob"):
        _validate_probability(getattr(sigma, name), f"sigmadrift.{name}")
    for lower, upper in (
        ("undershoot_min", "undershoot_max"),
        ("primary_sigma_min", "primary_sigma_max"),
        ("overshoot_min", "overshoot_max"),
        ("correction_sigma_min", "correction_sigma_max"),
        ("tremor_freq_min", "tremor_freq_max"),
        ("tremor_amp_min", "tremor_amp_max"),
    ):
        _validate_range_order(
            getattr(sigma, lower),
            getattr(sigma, upper),
            f"sigmadrift.{lower}",
            f"sigmadrift.{upper}",
        )


def load_aim_config(path: Union[str, Path] = DEFAULT_AIM_CONFIG) -> AimConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AimConfig()

    config = _build_config(_parse_config_values(config_path))
    _validate_config(config)
    return config


def sensitivity_to_angular_step_deg(sensitivity: float) -> float:
    return ((sensitivity * 0.6 + 0.2) ** 3) * 1.2


def _sigmadrift_payload(config: SigmaDriftConfig) -> tuple[float, ...]:
    return (
        config.fitts_a,
        config.fitts_b,
        config.target_width,
        config.undershoot_min,
        config.undershoot_max,
        config.peak_time_ratio,
        config.primary_sigma_min,
        config.primary_sigma_max,
        config.overshoot_prob,
        config.overshoot_min,
        config.overshoot_max,
        config.correction_sigma_min,
        config.correction_sigma_max,
        config.second_correction_prob,
        config.curvature_scale,
        config.ou_theta,
        config.ou_sigma,
        config.tremor_freq_min,
        config.tremor_freq_max,
        config.tremor_amp_min,
        config.tremor_amp_max,
        config.sdn_k,
        config.sample_dt_mean,
        config.gamma_shape,
    )


def generate_aim_path(
    start_orientation: Orientation,
    target: TargetMetrics,
    config: AimConfig | None = None,
    *,
    angular_step_deg: float,
    synthetic_export_root: Path | None = None,
    seed: int | None = None,
) -> tuple[AimPoint, ...]:
    resolved_config = config if config is not None else load_aim_config()
    if resolved_config.aim_model not in IMPLEMENTED_AIM_MODELS:
        raise ValueError(
            f"aim_model {resolved_config.aim_model!r} is configured but not implemented yet"
        )
    if resolved_config.aim_model == "minimum_jerk":
        minimum = resolved_config.minimum_jerk
        path = _generate_minimum_jerk_aim_path(
            start_orientation,
            target,
            angular_step_deg,
            minimum.fitts_a_ms,
            minimum.fitts_b_ms,
            minimum.min_duration_ms,
            minimum.max_duration_ms,
            minimum.sample_hz,
        )
    elif resolved_config.aim_model == "sigmadrift":
        path = _generate_sigmadrift_aim_path(
            start_orientation,
            target,
            angular_step_deg,
            _sigmadrift_payload(resolved_config.sigmadrift),
            seed,
        )
    else:
        raise ValueError(f"unsupported aim_model {resolved_config.aim_model!r}")

    if synthetic_export_root is not None:
        from minescript_miner.adapter.trajectory_export import (
            write_synthetic_trajectory_session,
        )

        write_synthetic_trajectory_session(
            synthetic_export_root,
            path,
            start_orientation,
            target,
            generator=resolved_config.aim_model,
            angular_step_deg=angular_step_deg,
            generator_config=asdict(resolved_config),
        )
    return path


def execute_aim_path(
    path: tuple[AimPoint, ...],
    *,
    sleep: Callable[[float], None] = time.sleep,
    is_active: Callable[[], bool] | None = None,
    settle_delay_s: float = 0.0,
) -> bool:
    from minescript_miner.minescript.io import set_orientation

    previous_t_ms: float | None = None
    last_point: AimPoint | None = None
    for point in path:
        if is_active is not None and not is_active():
            return False
        if previous_t_ms is not None:
            delay_s = max(0.0, (point.t_ms - previous_t_ms) / 1000.0)
            if delay_s > 0.0:
                sleep(delay_s)
            if is_active is not None and not is_active():
                return False
        set_orientation(point.yaw, point.pitch)
        previous_t_ms = point.t_ms
        last_point = point

    if last_point is None:
        return False
    if settle_delay_s > 0.0:
        sleep(settle_delay_s)
        if is_active is not None and not is_active():
            return False
        set_orientation(last_point.yaw, last_point.pitch)
    return True
