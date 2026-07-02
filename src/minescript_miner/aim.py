"""Aim path configuration and generation helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from minescript_miner.adapter.native_bridge import (
    AimPoint,
    Orientation,
    TargetMetrics,
    generate_minimum_jerk_aim_path as _generate_minimum_jerk_aim_path,
)


DEFAULT_AIM_CONFIG = Path("aim_config.txt")
DEFAULT_FALLBACK_ANGULAR_STEP_DEG = 0.15
SUPPORTED_AIM_MODELS = frozenset({"minimum_jerk"})


@dataclass(frozen=True)
class AimConfig:
    aim_model: str = "minimum_jerk"
    fallback_angular_step_deg: float = DEFAULT_FALLBACK_ANGULAR_STEP_DEG
    fitts_a_ms: float = 80.0
    fitts_b_ms: float = 110.0
    min_duration_ms: float = 60.0
    max_duration_ms: float = 450.0
    sample_hz: int = 120
    correction_probability: float = 0.6
    max_corrections: int = 1


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


def load_aim_config(path: Union[str, Path] = DEFAULT_AIM_CONFIG) -> AimConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AimConfig()

    values = {}
    parsers = {
        "aim_model": str,
        "fallback_angular_step_deg": _parse_float,
        "fitts_a_ms": _parse_float,
        "fitts_b_ms": _parse_float,
        "min_duration_ms": _parse_float,
        "max_duration_ms": _parse_float,
        "sample_hz": _parse_int,
        "correction_probability": _parse_float,
        "max_corrections": _parse_int,
    }
    for line_number, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(
                f"{config_path}:{line_number}: expected 'name: value'"
            )
        name, raw_value = line.split(":", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        if name not in parsers:
            raise ValueError(f"{config_path}:{line_number}: unknown aim config key {name!r}")
        parser = parsers[name]
        if parser is str:
            values[name] = raw_value
        else:
            values[name] = parser(raw_value, name)

    config = AimConfig(**values)
    if config.aim_model not in SUPPORTED_AIM_MODELS:
        raise ValueError(f"unsupported aim_model {config.aim_model!r}")
    if config.fallback_angular_step_deg <= 0.0:
        raise ValueError("fallback_angular_step_deg must be positive")
    if config.sample_hz <= 0:
        raise ValueError("sample_hz must be positive")
    if config.max_duration_ms < config.min_duration_ms:
        raise ValueError("max_duration_ms must be >= min_duration_ms")
    if not 0.0 <= config.correction_probability <= 1.0:
        raise ValueError("correction_probability must be in [0, 1]")
    if config.max_corrections < 0:
        raise ValueError("max_corrections must be >= 0")
    return config


def sensitivity_to_angular_step_deg(sensitivity: float) -> float:
    return ((sensitivity * 0.6 + 0.2) ** 3) * 1.2


def generate_aim_path(
    start_orientation: Orientation,
    target: TargetMetrics,
    config: AimConfig | None = None,
    *,
    angular_step_deg: float,
) -> tuple[AimPoint, ...]:
    resolved_config = config if config is not None else load_aim_config()
    if resolved_config.aim_model == "minimum_jerk":
        return _generate_minimum_jerk_aim_path(
            start_orientation,
            target,
            angular_step_deg,
            resolved_config.fitts_a_ms,
            resolved_config.fitts_b_ms,
            resolved_config.min_duration_ms,
            resolved_config.max_duration_ms,
            resolved_config.sample_hz,
        )
    raise ValueError(f"unsupported aim_model {resolved_config.aim_model!r}")
