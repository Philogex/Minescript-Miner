from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import minescript as m


PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = PROJECT_DIR / "src"
TARGET_CONFIG = PROJECT_DIR / "targets.txt"

for path in (PROJECT_DIR, SRC_DIR):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from minescript_miner.adapter.target_pipeline import (
    block_id_literal,
    load_target_blocks,
)
from minescript_miner.minescript.io import (
    player_orientation,
    set_orientation,
)
from minescript_miner.minescript.scanner import (
    acquire_current_target,
)
from minescript_miner.minescript.runtime import query


TOGGLE_KEY = "o"
REACH = 4.5
ROTATION_DURATION = 0.3
IDLE_DELAY = 0.25
BREAK_POLL_DELAY = 0.05
LOG_SCAN_TIMINGS = os.environ.get(
    "MINESCRIPT_MINER_LOG_TIMINGS", ""
).lower() in {"1", "true", "yes"}

active = threading.Event()


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
    yaw_start, pitch_start = player_orientation()
    yaw_delta = ((target_yaw - yaw_start + 180.0) % 360.0) - 180.0
    pitch_delta = target_pitch - pitch_start

    steps = max(1, int(duration / step))
    for index in range(steps + 1):
        t = index / steps * 2.0
        factor = ease_in_out(t)
        set_orientation(
            (yaw_start + yaw_delta * factor) % 360.0,
            pitch_start + pitch_delta * factor,
        )
        time.sleep(step)


def listen_keys() -> None:
    with m.EventQueue() as queue:
        queue.register_key_listener()
        while True:
            event = queue.get()
            if (
                event.type == m.EventType.KEY
                and event.action == 1
                and (event.key == TOGGLE_KEY or event.key == ord(TOGGLE_KEY.upper()))
            ):
                if active.is_set():
                    active.clear()
                    m.player_press_attack(False)
                else:
                    active.set()
                m.echo(f"Miner active: {active.is_set()}")


def targeted_block_is_configured(targeted_block, target_blocks) -> bool:
    return (
        targeted_block is not None
        and block_id_literal(targeted_block.type) in target_blocks
    )


def mine_targeted_block(target_blocks) -> None:
    targeted = query(m.player_get_targeted_block, REACH)
    if not targeted_block_is_configured(targeted, target_blocks):
        extended_target = query(m.player_get_targeted_block, 20.0)
        targeted_type = None if targeted is None else targeted.type
        extended_description = (
            None
            if extended_target is None
            else f"{extended_target.type} at {extended_target.distance:.3f}"
        )
        m.log(
            "Miner aim verification missed target: "
            f"reach={targeted_type}, extended={extended_description}"
        )
        return

    target_position = tuple(targeted.position)
    target_type = block_id_literal(targeted.type)
    m.player_press_attack(True)
    try:
        while active.is_set():
            current = query(m.player_get_targeted_block, REACH)
            if current is None:
                break
            if tuple(current.position) != target_position:
                break
            if block_id_literal(current.type) != target_type:
                break
            time.sleep(BREAK_POLL_DELAY)
    finally:
        m.player_press_attack(False)


def run() -> None:
    threading.Thread(target=listen_keys, daemon=True).start()

    target_blocks = load_target_blocks(TARGET_CONFIG)
    if not target_blocks:
        m.echo(f"Miner stopped: no targets configured in {TARGET_CONFIG.name}")
        return

    m.echo(f"Miner ready: press '{TOGGLE_KEY}' to toggle")
    await_region = True
    try:
        while True:
            if not active.wait(timeout=IDLE_DELAY):
                continue

            px, py, pz = query(m.player_position)
            yaw, pitch = query(m.player_orientation)
            start = time.perf_counter()
            target_orientation = acquire_current_target(
                (px, py + 1.62, pz),
                (yaw, pitch),
                REACH,
                target_blocks=target_blocks,
                await_region=await_region,
            )
            if LOG_SCAN_TIMINGS:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                m.log(f"Miner scan time: {elapsed_ms:.4f} ms")
            await_region = False
            if target_orientation is None:
                time.sleep(IDLE_DELAY)
                continue

            if not active.is_set():
                continue

            smooth_rotate_to(
                target_orientation[0],
                target_orientation[1],
                duration=ROTATION_DURATION,
            )
            if active.is_set():
                mine_targeted_block(target_blocks)
    finally:
        m.player_press_attack(False)


if __name__ == "__main__":
    run()
