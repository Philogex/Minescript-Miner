import _minescript_miner_native as native


def hello() -> str:
    return native.hello()


def scan_region_debug(position, orientation, block_ids):
    return native.scan_region_debug(position, orientation, block_ids)
