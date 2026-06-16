#include "minescript_miner/scan_region.hpp"

#include "minescript_miner/angle.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>

namespace minescript_miner {

static_assert(index_to_offset(0, 3).x == 0);
static_assert(index_to_offset(0, 3).y == 0);
static_assert(index_to_offset(0, 3).z == 0);

static_assert(index_to_offset(1, 3).x == 1);
static_assert(index_to_offset(3, 3).z == 1);
static_assert(index_to_offset(9, 3).y == 1);

static_assert(offset_to_index({2, 1, 2}, 3) == 17);
static_assert(index_to_block_pos(0, 3, {10, 64, -4}).x == 9);
static_assert(index_to_block_pos(0, 3, {10, 64, -4}).y == 63);
static_assert(index_to_block_pos(0, 3, {10, 64, -4}).z == -5);
static_assert(index_to_block_pos(26, 3, {10, 64, -4}).x == 11);
static_assert(index_to_block_pos(26, 3, {10, 64, -4}).y == 65);
static_assert(index_to_block_pos(26, 3, {10, 64, -4}).z == -3);

static_assert(GEOMETRY_UNITS_PER_BLOCK % 16 == 0);
constexpr std::uint8_t local_sixteenth(std::int32_t value) {
    return static_cast<std::uint8_t>(
        value * GEOMETRY_UNITS_PER_BLOCK / 16
    );
}

constexpr LocalRectFace X_FACE{
    PlaneAxis::X,
    local_sixteenth(8),
    local_sixteenth(2),
    local_sixteenth(6),
    local_sixteenth(3),
    local_sixteenth(7),
    1,
};
constexpr WorldRectFace X_WORLD = face_to_world(X_FACE, {10, 64, -4});
static_assert(face_p0(X_WORLD).x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(face_p0(X_WORLD).y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(2));
static_assert(face_p0(X_WORLD).z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(3));
static_assert(face_p2(X_WORLD).x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(face_p2(X_WORLD).y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(6));
static_assert(face_p2(X_WORLD).z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(7));

constexpr LocalRectFace Y_FACE{
    PlaneAxis::Y,
    local_sixteenth(8),
    local_sixteenth(2),
    local_sixteenth(6),
    local_sixteenth(3),
    local_sixteenth(7),
    1,
};
constexpr WorldRectFace Y_WORLD = face_to_world(Y_FACE, {10, 64, -4});
static_assert(face_p0(Y_WORLD).x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(2));
static_assert(face_p0(Y_WORLD).y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(face_p0(Y_WORLD).z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(3));
static_assert(face_p2(Y_WORLD).x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(6));
static_assert(face_p2(Y_WORLD).y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(face_p2(Y_WORLD).z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(7));

constexpr LocalRectFace Z_FACE{
    PlaneAxis::Z,
    local_sixteenth(8),
    local_sixteenth(2),
    local_sixteenth(6),
    local_sixteenth(3),
    local_sixteenth(7),
    1,
};
constexpr WorldRectFace Z_WORLD = face_to_world(Z_FACE, {10, 64, -4});
static_assert(face_p0(Z_WORLD).x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(2));
static_assert(face_p0(Z_WORLD).y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(3));
static_assert(face_p0(Z_WORLD).z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(face_p2(Z_WORLD).x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(6));
static_assert(face_p2(Z_WORLD).y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(7));
static_assert(face_p2(Z_WORLD).z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));

namespace {

constexpr bool offset_inside_cube(BlockOffset offset, std::int32_t side) {
    return offset.x >= 0 && offset.x < side &&
           offset.y >= 0 && offset.y < side &&
           offset.z >= 0 && offset.z < side;
}

constexpr BlockOffset face_neighbor_offset(const LocalRectFace &face) {
    switch (face.axis) {
        case PlaneAxis::X:
            return {face.normal_sign, 0, 0};
        case PlaneAxis::Y:
            return {0, face.normal_sign, 0};
        case PlaneAxis::Z:
            return {0, 0, face.normal_sign};
    }
    return {};
}

bool has_internal_full_cube_neighbor_face(
    UInt16View shape_ids,
    std::uint16_t block_index,
    std::int32_t side,
    const LocalRectFace &face
) {
    if (shape_ids[block_index] != SHAPE_FULL_CUBE) {
        return false;
    }

    const BlockOffset offset = index_to_offset(block_index, side);
    const BlockOffset delta = face_neighbor_offset(face);
    const BlockOffset neighbor{
        offset.x + delta.x,
        offset.y + delta.y,
        offset.z + delta.z,
    };
    if (!offset_inside_cube(neighbor, side)) {
        return false;
    }

    const std::uint16_t neighbor_index = offset_to_index(neighbor, side);
    return shape_ids[neighbor_index] == SHAPE_FULL_CUBE;
}

bool face_points_to_eye(const WorldRectFace &face, const Vec3 &eye) {
    const double coord =
        static_cast<double>(face.coord) / GEOMETRY_UNITS_PER_BLOCK;
    switch (face.axis) {
        case PlaneAxis::X:
            return face.normal_sign > 0 ? eye.x > coord : eye.x < coord;
        case PlaneAxis::Y:
            return face.normal_sign > 0 ? eye.y > coord : eye.y < coord;
        case PlaneAxis::Z:
            return face.normal_sign > 0 ? eye.z > coord : eye.z < coord;
    }
    return false;
}

double axis_distance(double value, double minimum, double maximum) {
    return std::max({minimum - value, 0.0, value - maximum});
}

bool face_within_reach(const WorldRectFace &face, const Vec3 &eye, double reach) {
    const Vec3 p0 = world_point_to_vec3(face_bounds_min(face));
    const Vec3 p2 = world_point_to_vec3(face_bounds_max(face));
    const double distance_x =
        axis_distance(eye.x, std::min(p0.x, p2.x), std::max(p0.x, p2.x));
    const double distance_y =
        axis_distance(eye.y, std::min(p0.y, p2.y), std::max(p0.y, p2.y));
    const double distance_z =
        axis_distance(eye.z, std::min(p0.z, p2.z), std::max(p0.z, p2.z));
    return distance_x * distance_x +
               distance_y * distance_y +
               distance_z * distance_z <=
           reach * reach;
}

std::size_t world_face_capacity(UInt16View shape_ids) {
    std::size_t capacity = 0;
    for (const std::uint16_t shape_id : shape_ids) {
        capacity += geometry_for_shape(shape_id).face_count;
    }
    return capacity;
}

std::size_t target_face_capacity(
    UInt16View shape_ids,
    UInt16View target_indices
) {
    std::size_t capacity = 0;
    for (const std::uint16_t target_index : target_indices) {
        capacity += geometry_for_shape(shape_ids[target_index]).face_count;
    }
    return capacity;
}

}  // namespace

ScanRegionGeometry build_scan_region_geometry(
    UInt16View shape_ids,
    UInt16View target_indices,
    const Vec3 &eye,
    const Vec3 &look_dir,
    std::int32_t side,
    double reach
) {
    const BlockPos center{
        static_cast<std::int32_t>(std::floor(eye.x)),
        static_cast<std::int32_t>(std::floor(eye.y)),
        static_cast<std::int32_t>(std::floor(eye.z)),
    };

    std::vector<std::uint8_t> target_lookup(shape_ids.size, 0);
    for (const std::uint16_t target_index : target_indices) {
        target_lookup[target_index] = 1;
    }

    ScanRegionGeometry geometry{};
    geometry.world_faces.reserve(world_face_capacity(shape_ids));
    geometry.target_faces.reserve(target_face_capacity(shape_ids, target_indices));

    const GeometryCatalog &catalog = geometry_catalog();
    for (std::size_t block_index = 0; block_index < shape_ids.size; ++block_index) {
        const std::uint16_t shape_id = shape_ids[block_index];
        const ShapeGeometry &shape = geometry_for_shape(shape_id);
        const auto block_index16 = static_cast<std::uint16_t>(block_index);
        const BlockPos block_pos = index_to_block_pos(block_index16, side, center);

        for (std::uint8_t face_index = 0; face_index < shape.face_count; ++face_index) {
            const LocalRectFace &local_face = catalog.faces[shape.face_offset + face_index];
            if (has_internal_full_cube_neighbor_face(shape_ids, block_index16, side, local_face)) {
                continue;
            }

            const WorldRectFace world_rect = face_to_world(local_face, block_pos);
            if (!face_points_to_eye(world_rect, eye)) {
                continue;
            }

            WorldFace world_face{
                world_rect,
                face_center(world_rect),
            };

            const std::uint32_t world_face_index =
                static_cast<std::uint32_t>(geometry.world_faces.size());
            geometry.world_faces.push_back(world_face);

            if (target_lookup[block_index] == 0 ||
                !face_within_reach(world_rect, eye, reach)) {
                continue;
            }
            geometry.target_faces.push_back({
                world_face_index,
                angle_to_point(look_dir, world_face.center - eye),
            });
        }
    }

    std::sort(geometry.target_faces.begin(), geometry.target_faces.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.center_angle != rhs.center_angle) {
            return lhs.center_angle < rhs.center_angle;
        }
        return lhs.world_face_index < rhs.world_face_index;
    });
    return geometry;
}

}  // namespace minescript_miner
