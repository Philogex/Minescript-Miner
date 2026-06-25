#include "minecraft_miner/scanner/scan_region.hpp"

#include "minecraft_miner/aim/angle.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <limits>

namespace minecraft_miner {

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

constexpr std::uint16_t INVALID_FACE_INDEX = std::numeric_limits<std::uint16_t>::max();

struct FullCubeFaceLookup {
    std::array<std::uint16_t, 6> face_indices{};
};

struct FaceIndexList {
    std::array<std::uint16_t, 3> face_indices{};
    std::uint8_t count = 0;
};

constexpr bool offset_inside_cube(BlockOffset offset, std::int32_t side) {
    return offset.x >= 0 && offset.x < side &&
           offset.y >= 0 && offset.y < side &&
           offset.z >= 0 && offset.z < side;
}

constexpr std::uint8_t face_lookup_slot(PlaneAxis axis, std::int8_t normal_sign) {
    return static_cast<std::uint8_t>(
        static_cast<std::uint8_t>(axis) * 2 + (normal_sign > 0 ? 1 : 0)
    );
}

FullCubeFaceLookup make_full_cube_face_lookup(const GeometryCatalog &catalog) {
    FullCubeFaceLookup lookup{};
    lookup.face_indices.fill(INVALID_FACE_INDEX);

    const ShapeGeometry &shape = geometry_for_shape(SHAPE_FULL_CUBE);
    for (std::uint8_t face_index = 0; face_index < shape.face_count; ++face_index) {
        const auto catalog_face_index =
            static_cast<std::uint16_t>(shape.face_offset + face_index);
        const LocalRectFace &face = catalog.faces[catalog_face_index];
        lookup.face_indices[face_lookup_slot(face.axis, face.normal_sign)] =
            catalog_face_index;
    }
    return lookup;
}

const FullCubeFaceLookup &full_cube_face_lookup(const GeometryCatalog &catalog) {
    static const FullCubeFaceLookup lookup = make_full_cube_face_lookup(catalog);
    return lookup;
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

void append_visible_full_cube_face(
    FaceIndexList &faces,
    const FullCubeFaceLookup &lookup,
    PlaneAxis axis,
    std::int8_t normal_sign
) {
    const std::uint16_t face_index =
        lookup.face_indices[face_lookup_slot(axis, normal_sign)];
    if (face_index == INVALID_FACE_INDEX) {
        return;
    }
    faces.face_indices[faces.count] = face_index;
    ++faces.count;
}

FaceIndexList visible_full_cube_faces(
    const FullCubeFaceLookup &lookup,
    BlockPos block_pos,
    const Vec3 &eye
) {
    FaceIndexList faces{};
    if (eye.x < static_cast<double>(block_pos.x)) {
        append_visible_full_cube_face(faces, lookup, PlaneAxis::X, -1);
    } else if (eye.x > static_cast<double>(block_pos.x + 1)) {
        append_visible_full_cube_face(faces, lookup, PlaneAxis::X, 1);
    }

    if (eye.y < static_cast<double>(block_pos.y)) {
        append_visible_full_cube_face(faces, lookup, PlaneAxis::Y, -1);
    } else if (eye.y > static_cast<double>(block_pos.y + 1)) {
        append_visible_full_cube_face(faces, lookup, PlaneAxis::Y, 1);
    }

    if (eye.z < static_cast<double>(block_pos.z)) {
        append_visible_full_cube_face(faces, lookup, PlaneAxis::Z, -1);
    } else if (eye.z > static_cast<double>(block_pos.z + 1)) {
        append_visible_full_cube_face(faces, lookup, PlaneAxis::Z, 1);
    }
    return faces;
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

void append_cached_world_face_if_visible(
    const ScanRegionGeometry &geometry,
    UInt16View shape_ids,
    std::uint16_t block_index,
    const LocalRectFace &local_face,
    BlockPos block_pos,
    const Vec3 &eye,
    std::int32_t side
) {
    if (has_internal_full_cube_neighbor_face(shape_ids, block_index, side, local_face)) {
        return;
    }

    const WorldRectFace world_rect = face_to_world(local_face, block_pos);
    if (!face_points_to_eye(world_rect, eye)) {
        return;
    }

    append_world_face(geometry, world_rect);
}

}  // namespace

WorldFaceSpan faces_for_block(
    const ScanRegionGeometry &geometry,
    std::uint16_t block_index
) {
    if (!geometry.has_lazy_block_faces() ||
        static_cast<std::size_t>(block_index) >= geometry.shape_ids.size) {
        return {};
    }

    BlockFaceSpan &span = geometry.block_faces[block_index];
    if (span.initialized) {
        return {span.offset, span.count};
    }

    span.initialized = true;
    span.offset = static_cast<std::uint32_t>(geometry.world_faces.size());

    const GeometryCatalog &catalog = geometry_catalog();
    const FullCubeFaceLookup &full_cube_lookup = full_cube_face_lookup(catalog);
    const std::uint16_t shape_id = geometry.shape_ids[block_index];
    const ShapeGeometry &shape = geometry_for_shape(shape_id);
    const BlockPos block_pos =
        index_to_block_pos(block_index, geometry.side, geometry.center);

    if (shape_id == SHAPE_FULL_CUBE) {
        const FaceIndexList faces =
            visible_full_cube_faces(full_cube_lookup, block_pos, geometry.eye);
        for (std::uint8_t index = 0; index < faces.count; ++index) {
            append_cached_world_face_if_visible(
                geometry,
                geometry.shape_ids,
                block_index,
                catalog.faces[faces.face_indices[index]],
                block_pos,
                geometry.eye,
                geometry.side
            );
        }
    } else {
        for (std::uint8_t face_index = 0;
             face_index < shape.face_count;
             ++face_index) {
            const LocalRectFace &local_face =
                catalog.faces[shape.face_offset + face_index];
            append_cached_world_face_if_visible(
                geometry,
                geometry.shape_ids,
                block_index,
                local_face,
                block_pos,
                geometry.eye,
                geometry.side
            );
        }
    }

    span.count = static_cast<std::uint16_t>(
        geometry.world_faces.size() - span.offset
    );
    return {span.offset, span.count};
}

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

    ScanRegionGeometry geometry{};
    geometry.shape_ids = shape_ids;
    geometry.eye = eye;
    geometry.center = center;
    geometry.side = side;
    geometry.block_faces.resize(shape_ids.size);
    const std::size_t target_capacity =
        target_face_capacity(shape_ids, target_indices);
    const std::size_t face_capacity = std::max<std::size_t>(target_capacity, 16);
    geometry.world_faces.reserve(face_capacity);
    geometry.world_face_centers.reserve(face_capacity);
    geometry.target_faces.reserve(target_capacity);

    for (const std::uint16_t target_index : target_indices) {
        const WorldFaceSpan span = faces_for_block(geometry, target_index);
        for (std::uint16_t face_index = 0;
             face_index < span.count;
             ++face_index) {
            const std::uint32_t world_face_index =
                span.offset + face_index;
            const WorldRectFace &world_face =
                geometry.world_faces[world_face_index];
            if (!face_within_reach(world_face, eye, reach)) {
                continue;
            }
            geometry.target_faces.push_back({
                world_face_index,
                angle_to_point(
                    look_dir,
                    world_face_center(geometry, world_face_index) - eye
                ),
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

}  // namespace minecraft_miner
