import time

import minescript as m

# ------------------------------
# aiming
# ------------------------------

def ease_in_out(t: float) -> float:
    if t < 1.0:
        return (t * t) / 2.0
    t -= 1.0
    return -(t * (t - 2.0) - 1.0) / 2.0

def smooth_rotate_to(target_yaw: float, target_pitch: float, duration: float, step: float = 0.02):
    yaw0, pitch0 = m.player_orientation()

    dyaw = ((target_yaw - yaw0 + 180.0) % 360.0) - 180.0
    dpitch = target_pitch - pitch0

    steps = max(1, int(duration / step))
    for i in range(steps + 1):
        t = i / steps * 2.0
        f = ease_in_out(t)
        y = yaw0 + dyaw * f
        p = pitch0 + dpitch * f
        m.player_set_orientation(y % 360.0, p)
        time.sleep(step)