import _minescript_miner_native as native


def hello() -> str:
    return native.hello()


def shape_catalog_debug():
    return native.shape_catalog_debug()


def scan_region_debug(position, orientation, catalog_version, side, shape_ids):
    return native.scan_region_debug(
        position,
        orientation,
        catalog_version,
        side,
        shape_ids,
    )
