from visibility_scanner.scanner import scan_targets, scan_target
from visibility_scanner.world_scanners import get_area, get_line
import aim.player_aim

import threading
import time

import minescript as m


# ------------------------------
# control handler
# ------------------------------

m.set_default_executor(m.script_loop)

active = False

def listen_keys():
    global active
    with m.EventQueue() as eq:
        eq.register_key_listener()
        while True:
            event = eq.get()
            if event.type == m.EventType.KEY:
                if (event.key == 'o' or event.key == 79) and event.action == 1:
                    active = not active
                    m.echo(f"Script active: {active}")
            event = None

threading.Thread(target=listen_keys, daemon=True).start()
m.echo("Press 'o' to activate/exit.")


# ------------------------------
# params
# ------------------------------

target_ids = [
    "minecraft:diamond_ore",
    "minecraft:deepslate_diamond_ore"
]

reach = 4.8
previous_target = m.player_position()


# ------------------------------
# main loop
# ------------------------------

while True:
    if not active:
        prev_block = None
    while not active:
        time.sleep(0.5)

    px, py, pz = m.player_position()
    occluders = get_area(position=(px, py + 1.62, pz))

    aim_result = scan_targets(
        position=(px, py + 1.62, pz), target_ids=target_ids, occluders=occluders, previous_target=previous_target
    )

    if aim_result is not None:
        previous_target = aim_result.centroid_world_pos
        x, y, z = aim_result.world_pos
        aim.player_aim.smooth_rotate_to(aim_result.target_angle[0], aim_result.target_angle[1], duration=0.3)
        prev_block = m.getblock(x, y, z)
        m.player_press_attack(True)
        while prev_block == m.getblock(x, y, z) and active:
            time.sleep(0.05)
        m.player_press_attack(False)
    else:
        time.sleep(0.5)

'''
# minimal usage example for scan_target()
start = (10, -58.5, 20) # use central position, or the ground will occlude
end = (10, -58.5, 10) # same here

occluders = get_line(position=start, target=end)
aim_result = scan_target(position=start, target=end, occluders=occluders)
'''