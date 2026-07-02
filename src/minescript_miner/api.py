import _minescript_miner_native as native
from minescript_miner.adapter.native_bridge import (
    TargetMetrics,
    acquire_target_metrics as bridged_acquire_target_metrics,
)


def hello() -> str:
    return native.hello()


def geometry_catalog_debug():
    return native.geometry_catalog_debug()


def acquire_target(position, orientation, shape_catalog_version, side, reach, shape_ids, target_indices):
    return native.acquire_target(
        position,
        orientation,
        shape_catalog_version,
        side,
        reach,
        shape_ids,
        target_indices,
    )


def acquire_target_metrics(position, orientation, shape_catalog_version, side, reach, shape_ids, target_indices):
    return bridged_acquire_target_metrics(
        position,
        orientation,
        shape_catalog_version,
        side,
        reach,
        shape_ids,
        target_indices,
    )


def get_angle_to_block(block_pos):
    from minescript_miner.queries import get_angle_to_block as query_angle

    return query_angle(block_pos)


def can_see_block(source, block_pos):
    from minescript_miner.queries import can_see_block as query_visibility

    return query_visibility(source, block_pos)
