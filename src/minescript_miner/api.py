import _minescript_miner_native as native


def hello() -> str:
    return native.hello()


def scan_region_debug(position, orientation, side, type_ids, state_ids):
    return native.scan_region_debug(position, orientation, side, type_ids, state_ids)
