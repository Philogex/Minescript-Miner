#pragma once

#include <cstdint>

namespace minescript_miner {

struct BlockOffset {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;
};

struct BlockPos {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;
};

constexpr BlockOffset index_to_offset(std::uint16_t index, std::int32_t side) {
    return {
        static_cast<std::int32_t>(index % side),
        static_cast<std::int32_t>(index / (side * side)),
        static_cast<std::int32_t>((index / side) % side),
    };
}

constexpr std::uint16_t offset_to_index(BlockOffset offset, std::int32_t side) {
    return static_cast<std::uint16_t>(offset.x + offset.z * side + offset.y * side * side);
}

constexpr BlockPos cube_min_pos(BlockPos center, std::int32_t side) {
    const std::int32_t half = side / 2;
    return {center.x - half, center.y - half, center.z - half};
}

constexpr BlockPos index_to_block_pos(std::uint16_t index, std::int32_t side, BlockPos center) {
    const BlockOffset offset = index_to_offset(index, side);
    const BlockPos min_pos = cube_min_pos(center, side);
    return {min_pos.x + offset.x, min_pos.y + offset.y, min_pos.z + offset.z};
}

}  // namespace minescript_miner
