#pragma once

#include "minescript_miner/geometry_catalog.hpp"

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

struct WorldPoint16 {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;
};

struct WorldRectFace16 {
    PlaneAxis axis = PlaneAxis::X;
    std::int8_t normal_sign = 0;
    WorldPoint16 p0{};
    WorldPoint16 p1{};
    WorldPoint16 p2{};
    WorldPoint16 p3{};
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

constexpr std::int32_t world16(std::int32_t block_coord, std::uint8_t local_coord) {
    return block_coord * 16 + static_cast<std::int32_t>(local_coord);
}

constexpr WorldRectFace16 face_to_world(const RectFace16 &face, BlockPos block_pos) {
    switch (face.axis) {
        case PlaneAxis::X: {
            const std::int32_t x = world16(block_pos.x, face.coord);
            const std::int32_t y0 = world16(block_pos.y, face.u_min);
            const std::int32_t y1 = world16(block_pos.y, face.u_max);
            const std::int32_t z0 = world16(block_pos.z, face.v_min);
            const std::int32_t z1 = world16(block_pos.z, face.v_max);
            return {face.axis, face.normal_sign, {x, y0, z0}, {x, y1, z0}, {x, y1, z1}, {x, y0, z1}};
        }
        case PlaneAxis::Y: {
            const std::int32_t y = world16(block_pos.y, face.coord);
            const std::int32_t x0 = world16(block_pos.x, face.u_min);
            const std::int32_t x1 = world16(block_pos.x, face.u_max);
            const std::int32_t z0 = world16(block_pos.z, face.v_min);
            const std::int32_t z1 = world16(block_pos.z, face.v_max);
            return {face.axis, face.normal_sign, {x0, y, z0}, {x1, y, z0}, {x1, y, z1}, {x0, y, z1}};
        }
        case PlaneAxis::Z: {
            const std::int32_t z = world16(block_pos.z, face.coord);
            const std::int32_t x0 = world16(block_pos.x, face.u_min);
            const std::int32_t x1 = world16(block_pos.x, face.u_max);
            const std::int32_t y0 = world16(block_pos.y, face.v_min);
            const std::int32_t y1 = world16(block_pos.y, face.v_max);
            return {face.axis, face.normal_sign, {x0, y0, z}, {x1, y0, z}, {x1, y1, z}, {x0, y1, z}};
        }
    }
    return {};
}

}  // namespace minescript_miner
