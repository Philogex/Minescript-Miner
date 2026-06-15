import os
import sys
from pathlib import Path

import minescript as m

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
TARGET_CONFIG = Path(PROJECT_DIR) / "targets.txt"

for path in (PROJECT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from minescript_miner.minescript.io import (
    acquire_current_target,
    load_target_blocks,
)
from geometry_catalog_check import assert_geometry_catalog_parity


def latest_native_log_value(key):
    log_path = Path(os.environ.get("MINESCRIPT_MINER_NATIVE_LOG", "minescript_miner_native_scan.log"))
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    prefix = f"{key}:"
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip()
    return None


catalog = assert_geometry_catalog_parity()
m.echo(
    "Geometry catalog OK: "
    f"version {catalog['version']}, {catalog['shape_count']} shapes"
)

px, py, pz = m.player_position()
yaw, pitch = m.player_orientation()

m.echo(f"python orientation yaw/pitch: {yaw}, {pitch}")
configured_targets = load_target_blocks(TARGET_CONFIG)
direction = acquire_current_target(
    (px, py + 1.62, pz),
    (yaw, pitch),
    target_blocks=configured_targets,
)
m.echo(f"native direction: {direction}")

target_blocks = latest_native_log_value("target_block_count")
target_face_count = latest_native_log_value("target_face_count")
target_faces = latest_native_log_value("first_target_face_indices")
target_angles = latest_native_log_value("first_target_face_center_angles_rad")
if target_blocks is not None:
    m.echo(f"native target blocks: {target_blocks}")
if target_face_count is not None:
    m.echo(f"native target face count: {target_face_count}")
if target_faces is not None:
    m.echo(f"native target faces: {target_faces}")
if target_angles is not None:
    m.echo(f"native target face center angles: {target_angles}")
