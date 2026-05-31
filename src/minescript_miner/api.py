import _minescript_miner_native as native


def hello() -> str:
    return native.hello()


def geometry_catalog_debug():
    return native.geometry_catalog_debug()


def acquire_target(position, orientation, shape_catalog_version, side, shape_ids):
    return native.acquire_target(
        position,
        orientation,
        shape_catalog_version,
        side,
        shape_ids,
    )
