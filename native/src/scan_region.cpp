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
static_assert(X_WORLD.p0.x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(X_WORLD.p0.y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(2));
static_assert(X_WORLD.p0.z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(3));
static_assert(X_WORLD.p2.x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(X_WORLD.p2.y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(6));
static_assert(X_WORLD.p2.z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(7));

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
static_assert(Y_WORLD.p0.x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(2));
static_assert(Y_WORLD.p0.y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(Y_WORLD.p0.z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(3));
static_assert(Y_WORLD.p2.x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(6));
static_assert(Y_WORLD.p2.y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(Y_WORLD.p2.z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(7));

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
static_assert(Z_WORLD.p0.x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(2));
static_assert(Z_WORLD.p0.y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(3));
static_assert(Z_WORLD.p0.z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));
static_assert(Z_WORLD.p2.x == 10 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(6));
static_assert(Z_WORLD.p2.y == 64 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(7));
static_assert(Z_WORLD.p2.z == -4 * GEOMETRY_UNITS_PER_BLOCK + local_sixteenth(8));

namespace {

bool face_points_to_eye(const WorldFace &face, const Vec3 &eye) {
    return dot(face_normal(face.face), eye - face.center) > 0.0;
}

double axis_distance(double value, double minimum, double maximum) {
    return std::max({minimum - value, 0.0, value - maximum});
}

bool face_within_reach(const WorldRectFace &face, const Vec3 &eye, double reach) {
    const Vec3 p0 = world_point_to_vec3(face.p0);
    const Vec3 p2 = world_point_to_vec3(face.p2);
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

std::size_t world_face_capacity(const std::vector<std::uint16_t> &shape_ids) {
    std::size_t capacity = 0;
    for (const std::uint16_t shape_id : shape_ids) {
        capacity += geometry_for_shape(shape_id).face_count;
    }
    return capacity;
}

std::size_t target_face_capacity(
    const std::vector<std::uint16_t> &shape_ids,
    const std::vector<std::uint16_t> &target_indices
) {
    std::size_t capacity = 0;
    for (const std::uint16_t target_index : target_indices) {
        capacity += geometry_for_shape(shape_ids[target_index]).face_count;
    }
    return capacity;
}

}  // namespace

ScanRegionGeometry build_scan_region_geometry(
    const std::vector<std::uint16_t> &shape_ids,
    const std::vector<std::uint16_t> &target_indices,
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

    std::vector<std::uint8_t> target_lookup(shape_ids.size(), 0);
    for (const std::uint16_t target_index : target_indices) {
        target_lookup[target_index] = 1;
    }

    ScanRegionGeometry geometry{};
    geometry.world_faces.reserve(world_face_capacity(shape_ids));
    geometry.target_faces.reserve(target_face_capacity(shape_ids, target_indices));

    const GeometryCatalog &catalog = geometry_catalog();
    for (std::size_t block_index = 0; block_index < shape_ids.size(); ++block_index) {
        const std::uint16_t shape_id = shape_ids[block_index];
        const ShapeGeometry &shape = geometry_for_shape(shape_id);
        const BlockPos block_pos = index_to_block_pos(static_cast<std::uint16_t>(block_index), side, center);

        for (std::uint8_t face_index = 0; face_index < shape.face_count; ++face_index) {
            const LocalRectFace &local_face = catalog.faces[shape.face_offset + face_index];
            const WorldRectFace world_rect = face_to_world(local_face, block_pos);
            WorldFace world_face{
                world_rect,
                face_center(world_rect),
            };

            const std::uint32_t world_face_index =
                static_cast<std::uint32_t>(geometry.world_faces.size());
            geometry.world_faces.push_back(world_face);

            if (target_lookup[block_index] == 0 ||
                !face_points_to_eye(world_face, eye) ||
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
