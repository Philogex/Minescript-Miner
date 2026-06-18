"""Small Minescript IO boundary used by the query layer."""

from __future__ import annotations

import minescript as m

from minescript_miner.minescript.runtime import query


def await_loaded_region(min_x: int, min_z: int, max_x: int, max_z: int) -> None:
    query(m.await_loaded_region, min_x, min_z, max_x, max_z)


def player_position():
    return query(m.player_position)


def read_block_region(pos1, pos2):
    return query(m.get_block_region, pos1, pos2)
