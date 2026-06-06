"""Minescript-facing camera movement helpers."""

import time

import minescript as m


def ease_in_out(t: float) -> float:
    if t < 1.0:
        return (t * t) / 2.0
    t -= 1.0
    return -(t * (t - 2.0) - 1.0) / 2.0


def smooth_rotate_to(
    target_yaw: float,
    target_pitch: float,
    duration: float,
    step: float = 0.02,
) -> None:
    yaw_start, pitch_start = m.player_orientation()
    yaw_delta = ((target_yaw - yaw_start + 180.0) % 360.0) - 180.0
    pitch_delta = target_pitch - pitch_start

    steps = max(1, int(duration / step))
    for index in range(steps + 1):
        t = index / steps * 2.0
        factor = ease_in_out(t)
        m.player_set_orientation(
            (yaw_start + yaw_delta * factor) % 360.0,
            pitch_start + pitch_delta * factor,
        )
        time.sleep(step)
