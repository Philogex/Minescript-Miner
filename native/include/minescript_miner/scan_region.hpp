#pragma once

#include "minescript_miner/geometry_catalog.hpp"
#include "minescript_miner/vec.hpp"

#include <cstdint>
#include <limits>
#include <vector>

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

struct WorldPoint {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;
};

struct WorldRectFace {
    PlaneAxis axis = PlaneAxis::X;
    std::int8_t normal_sign = 0;
    WorldPoint p0{};
    WorldPoint p1{};
    WorldPoint p2{};
    WorldPoint p3{};
};

struct WorldFace {
    WorldRectFace face{};
    Vec3 center{};
};

struct TargetFaceCandidate {
    std::uint32_t world_face_index = 0;
    double center_angle = 0.0;
};

struct UInt16View {
    const std::uint16_t *data = nullptr;
    std::size_t size = 0;

    constexpr UInt16View() = default;
    constexpr UInt16View(const std::uint16_t *data, std::size_t size)
        : data(data), size(size) {}
    UInt16View(const std::vector<std::uint16_t> &values)
        : data(values.data()), size(values.size()) {}

    constexpr const std::uint16_t *begin() const {
        return data;
    }

    constexpr const std::uint16_t *end() const {
        return data + size;
    }

    constexpr const std::uint16_t &operator[](std::size_t index) const {
        return data[index];
    }
};

struct ScanRegionGeometry {
    std::vector<WorldFace> world_faces{};
    std::vector<TargetFaceCandidate> target_faces{};
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

constexpr std::int32_t world_grid_coordinate(
    std::int32_t block_coord,
    std::uint8_t local_coord
) {
    return block_coord * GEOMETRY_UNITS_PER_BLOCK +
           static_cast<std::int32_t>(local_coord);
}

// Does not consider non AABB shapes
constexpr WorldRectFace face_to_world(const LocalRectFace &face, BlockPos block_pos) {
    switch (face.axis) {
        case PlaneAxis::X: {
            const std::int32_t x = world_grid_coordinate(block_pos.x, face.coord);
            const std::int32_t y0 = world_grid_coordinate(block_pos.y, face.u_min);
            const std::int32_t y1 = world_grid_coordinate(block_pos.y, face.u_max);
            const std::int32_t z0 = world_grid_coordinate(block_pos.z, face.v_min);
            const std::int32_t z1 = world_grid_coordinate(block_pos.z, face.v_max);
            return {face.axis, face.normal_sign, {x, y0, z0}, {x, y1, z0}, {x, y1, z1}, {x, y0, z1}};
        }
        case PlaneAxis::Y: {
            const std::int32_t y = world_grid_coordinate(block_pos.y, face.coord);
            const std::int32_t x0 = world_grid_coordinate(block_pos.x, face.u_min);
            const std::int32_t x1 = world_grid_coordinate(block_pos.x, face.u_max);
            const std::int32_t z0 = world_grid_coordinate(block_pos.z, face.v_min);
            const std::int32_t z1 = world_grid_coordinate(block_pos.z, face.v_max);
            return {face.axis, face.normal_sign, {x0, y, z0}, {x1, y, z0}, {x1, y, z1}, {x0, y, z1}};
        }
        case PlaneAxis::Z: {
            const std::int32_t z = world_grid_coordinate(block_pos.z, face.coord);
            const std::int32_t x0 = world_grid_coordinate(block_pos.x, face.u_min);
            const std::int32_t x1 = world_grid_coordinate(block_pos.x, face.u_max);
            const std::int32_t y0 = world_grid_coordinate(block_pos.y, face.v_min);
            const std::int32_t y1 = world_grid_coordinate(block_pos.y, face.v_max);
            return {face.axis, face.normal_sign, {x0, y0, z}, {x1, y0, z}, {x1, y1, z}, {x0, y1, z}};
        }
    }
    return {};
}

constexpr Vec3 world_point_to_vec3(WorldPoint point) {
    return {
        static_cast<double>(point.x) / GEOMETRY_UNITS_PER_BLOCK,
        static_cast<double>(point.y) / GEOMETRY_UNITS_PER_BLOCK,
        static_cast<double>(point.z) / GEOMETRY_UNITS_PER_BLOCK,
    };
}

constexpr Vec3 face_center(const WorldRectFace &face) {
    constexpr double center_denominator =
        2.0 * static_cast<double>(GEOMETRY_UNITS_PER_BLOCK);
    return {
        static_cast<double>(face.p0.x + face.p2.x) / center_denominator,
        static_cast<double>(face.p0.y + face.p2.y) / center_denominator,
        static_cast<double>(face.p0.z + face.p2.z) / center_denominator,
    };
}

constexpr Vec3 face_normal(const WorldRectFace &face) {
    switch (face.axis) {
        case PlaneAxis::X:
            return {static_cast<double>(face.normal_sign), 0.0, 0.0};
        case PlaneAxis::Y:
            return {0.0, static_cast<double>(face.normal_sign), 0.0};
        case PlaneAxis::Z:
            return {0.0, 0.0, static_cast<double>(face.normal_sign)};
    }
    return {};
}

ScanRegionGeometry build_scan_region_geometry(
    UInt16View shape_ids,
    UInt16View target_indices,
    const Vec3 &eye,
    const Vec3 &look_dir,
    std::int32_t side,
    double reach = std::numeric_limits<double>::infinity()
);

}  // namespace minescript_miner
