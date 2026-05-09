import os
import sys

import minescript as m

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")

for path in (PROJECT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from minescript_miner.minescript_adapter import scan_region_debug

px, py, pz = m.player_position()
yaw, pitch = m.player_orientation()

m.echo(f"python orientation yaw/pitch: {yaw}, {pitch}")
direction = scan_region_debug((px, py + 1.62, pz), (yaw, pitch))
m.echo(f"native direction: {direction}")
