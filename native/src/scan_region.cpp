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

constexpr RectFace16 X_FACE{PlaneAxis::X, 8, 2, 6, 3, 7, 1};
constexpr WorldRectFace16 X_WORLD = face_to_world(X_FACE, {10, 64, -4});
static_assert(X_WORLD.p0.x == 168);
static_assert(X_WORLD.p0.y == 1026);
static_assert(X_WORLD.p0.z == -61);
static_assert(X_WORLD.p2.x == 168);
static_assert(X_WORLD.p2.y == 1030);
static_assert(X_WORLD.p2.z == -57);

constexpr RectFace16 Y_FACE{PlaneAxis::Y, 8, 2, 6, 3, 7, 1};
constexpr WorldRectFace16 Y_WORLD = face_to_world(Y_FACE, {10, 64, -4});
static_assert(Y_WORLD.p0.x == 162);
static_assert(Y_WORLD.p0.y == 1032);
static_assert(Y_WORLD.p0.z == -61);
static_assert(Y_WORLD.p2.x == 166);
static_assert(Y_WORLD.p2.y == 1032);
static_assert(Y_WORLD.p2.z == -57);

constexpr RectFace16 Z_FACE{PlaneAxis::Z, 8, 2, 6, 3, 7, 1};
constexpr WorldRectFace16 Z_WORLD = face_to_world(Z_FACE, {10, 64, -4});
static_assert(Z_WORLD.p0.x == 162);
static_assert(Z_WORLD.p0.y == 1027);
static_assert(Z_WORLD.p0.z == -56);
static_assert(Z_WORLD.p2.x == 166);
static_assert(Z_WORLD.p2.y == 1031);
static_assert(Z_WORLD.p2.z == -56);

namespace {

bool face_points_to_eye(const WorldFace &face, const Vec3 &eye) {
    return dot(face_normal(face.face), eye - face.center) > 0.0;
}

double axis_distance(double value, double minimum, double maximum) {
    return std::max({minimum - value, 0.0, value - maximum});
}

bool face_within_reach(const WorldRectFace16 &face, const Vec3 &eye, double reach) {
    const Vec3 p0 = point16_to_world(face.p0);
    const Vec3 p2 = point16_to_world(face.p2);
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
            const RectFace16 &local_face = catalog.faces[shape.face_offset + face_index];
            const WorldRectFace16 world_rect = face_to_world(local_face, block_pos);
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
