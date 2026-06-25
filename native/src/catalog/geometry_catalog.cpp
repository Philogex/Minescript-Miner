#include "minecraft_miner/catalog/geometry_catalog.hpp"

#include "minecraft_miner/catalog/geometry_catalog_data.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace minecraft_miner {

namespace {

struct AxisPair {
    PlaneAxis first;
    PlaneAxis second;
};

struct BoxList {
    std::array<LocalAabb, 5> boxes{};
    std::uint8_t count = 0;
};

struct SplitList {
    std::array<std::uint8_t, 12> values{};
    std::uint8_t count = 0;
};

struct CatalogBuilder {
    GeometryCatalog catalog{};
    std::uint16_t shape_count = 0;
    std::uint16_t face_count = 0;
};

constexpr std::uint8_t axis_min(const LocalAabb &b, PlaneAxis axis) {
    switch (axis) {
        case PlaneAxis::X:
            return b.min_x;
        case PlaneAxis::Y:
            return b.min_y;
        case PlaneAxis::Z:
            return b.min_z;
    }
    return 0;
}

constexpr std::uint8_t axis_max(const LocalAabb &b, PlaneAxis axis) {
    switch (axis) {
        case PlaneAxis::X:
            return b.max_x;
        case PlaneAxis::Y:
            return b.max_y;
        case PlaneAxis::Z:
            return b.max_z;
    }
    return 0;
}

constexpr AxisPair uv_axes(PlaneAxis axis) {
    switch (axis) {
        case PlaneAxis::X:
            return {PlaneAxis::Y, PlaneAxis::Z};
        case PlaneAxis::Y:
            return {PlaneAxis::X, PlaneAxis::Z};
        case PlaneAxis::Z:
            return {PlaneAxis::X, PlaneAxis::Y};
    }
    return {PlaneAxis::X, PlaneAxis::Y};
}

constexpr bool overlaps_1d(std::uint8_t a_min, std::uint8_t a_max, std::uint8_t b_min, std::uint8_t b_max) {
    return a_min < b_max && b_min < a_max;
}

constexpr void add_box(BoxList &shape, LocalAabb b) {
    shape.boxes[shape.count] = b;
    ++shape.count;
}

constexpr void add_split(SplitList &splits, std::uint8_t value, std::uint8_t min_value, std::uint8_t max_value) {
    if (value > min_value && value < max_value) {
        splits.values[splits.count] = value;
        ++splits.count;
    }
}

constexpr void sort_unique(SplitList &splits) {
    for (std::uint8_t i = 1; i < splits.count; ++i) {
        const std::uint8_t value = splits.values[i];
        std::uint8_t j = i;
        while (j > 0 && splits.values[j - 1] > value) {
            splits.values[j] = splits.values[j - 1];
            --j;
        }
        splits.values[j] = value;
    }

    std::uint8_t write_index = 0;
    for (std::uint8_t read_index = 0; read_index < splits.count; ++read_index) {
        if (write_index == 0 || splits.values[read_index] != splits.values[write_index - 1]) {
            splits.values[write_index] = splits.values[read_index];
            ++write_index;
        }
    }
    splits.count = write_index;
}

constexpr bool midpoint_inside(std::uint8_t min_value, std::uint8_t max_value, std::uint16_t midpoint_times_2) {
    return static_cast<std::uint16_t>(min_value) * 2 < midpoint_times_2 &&
           midpoint_times_2 < static_cast<std::uint16_t>(max_value) * 2;
}

constexpr bool outside_occupied(
    const LocalRectFace &face,
    std::uint16_t u_midpoint_times_2,
    std::uint16_t v_midpoint_times_2,
    const BoxList &shape
) {
    const AxisPair axes = uv_axes(face.axis);
    for (std::uint8_t i = 0; i < shape.count; ++i) {
        const LocalAabb &b = shape.boxes[i];
        const bool crosses_face =
            face.normal_sign > 0
                ? axis_min(b, face.axis) <= face.coord && axis_max(b, face.axis) > face.coord
                : axis_min(b, face.axis) < face.coord && axis_max(b, face.axis) >= face.coord;
        if (!crosses_face) {
            continue;
        }

        if (midpoint_inside(axis_min(b, axes.first), axis_max(b, axes.first), u_midpoint_times_2) &&
            midpoint_inside(axis_min(b, axes.second), axis_max(b, axes.second), v_midpoint_times_2)) {
            return true;
        }
    }
    return false;
}

constexpr void add_face(CatalogBuilder &builder, LocalRectFace face) {
    builder.catalog.faces[builder.face_count] = face;
    ++builder.face_count;
}

constexpr void add_face_cells(CatalogBuilder &builder, const LocalRectFace &face, const BoxList &shape) {
    const AxisPair axes = uv_axes(face.axis);
    SplitList u_splits{};
    SplitList v_splits{};
    u_splits.values[u_splits.count++] = face.u_min;
    u_splits.values[u_splits.count++] = face.u_max;
    v_splits.values[v_splits.count++] = face.v_min;
    v_splits.values[v_splits.count++] = face.v_max;

    for (std::uint8_t i = 0; i < shape.count; ++i) {
        const LocalAabb &b = shape.boxes[i];
        if (axis_min(b, face.axis) > face.coord || axis_max(b, face.axis) < face.coord) {
            continue;
        }

        const std::uint8_t box_u_min = axis_min(b, axes.first);
        const std::uint8_t box_u_max = axis_max(b, axes.first);
        const std::uint8_t box_v_min = axis_min(b, axes.second);
        const std::uint8_t box_v_max = axis_max(b, axes.second);
        if (!overlaps_1d(face.u_min, face.u_max, box_u_min, box_u_max) ||
            !overlaps_1d(face.v_min, face.v_max, box_v_min, box_v_max)) {
            continue;
        }

        add_split(u_splits, box_u_min, face.u_min, face.u_max);
        add_split(u_splits, box_u_max, face.u_min, face.u_max);
        add_split(v_splits, box_v_min, face.v_min, face.v_max);
        add_split(v_splits, box_v_max, face.v_min, face.v_max);
    }

    sort_unique(u_splits);
    sort_unique(v_splits);

    for (std::uint8_t u_index = 0; u_index + 1 < u_splits.count; ++u_index) {
        for (std::uint8_t v_index = 0; v_index + 1 < v_splits.count; ++v_index) {
            const std::uint8_t u_min = u_splits.values[u_index];
            const std::uint8_t u_max = u_splits.values[u_index + 1];
            const std::uint8_t v_min = v_splits.values[v_index];
            const std::uint8_t v_max = v_splits.values[v_index + 1];
            if (u_max <= u_min || v_max <= v_min) {
                continue;
            }

            const std::uint16_t u_midpoint_times_2 = static_cast<std::uint16_t>(u_min) + u_max;
            const std::uint16_t v_midpoint_times_2 = static_cast<std::uint16_t>(v_min) + v_max;
            if (outside_occupied(face, u_midpoint_times_2, v_midpoint_times_2, shape)) {
                continue;
            }

            add_face(builder, {face.axis, face.coord, u_min, u_max, v_min, v_max, face.normal_sign});
        }
    }
}

constexpr void add_box_faces(CatalogBuilder &builder, const LocalAabb &b, const BoxList &shape) {
    add_face_cells(builder, {PlaneAxis::X, b.min_x, b.min_y, b.max_y, b.min_z, b.max_z, -1}, shape);
    add_face_cells(builder, {PlaneAxis::X, b.max_x, b.min_y, b.max_y, b.min_z, b.max_z, 1}, shape);
    add_face_cells(builder, {PlaneAxis::Y, b.min_y, b.min_x, b.max_x, b.min_z, b.max_z, -1}, shape);
    add_face_cells(builder, {PlaneAxis::Y, b.max_y, b.min_x, b.max_x, b.min_z, b.max_z, 1}, shape);
    add_face_cells(builder, {PlaneAxis::Z, b.min_z, b.min_x, b.max_x, b.min_y, b.max_y, -1}, shape);
    add_face_cells(builder, {PlaneAxis::Z, b.max_z, b.min_x, b.max_x, b.min_y, b.max_y, 1}, shape);
}

constexpr void add_shape(CatalogBuilder &builder, const BoxList &shape) {
    const std::uint16_t face_offset = builder.face_count;

    for (std::uint8_t i = 0; i < shape.count; ++i) {
        add_box_faces(builder, shape.boxes[i], shape);
    }

    builder.catalog.shapes[builder.shape_count] = {
        face_offset,
        static_cast<std::uint8_t>(builder.face_count - face_offset),
    };
    ++builder.shape_count;
}

constexpr BoxList shape_boxes(std::size_t shape_index) {
    BoxList shape{};
    const generated::ShapeBoxRange range = generated::SHAPE_BOX_RANGES[shape_index];
    for (std::uint8_t i = 0; i < range.count; ++i) {
        add_box(shape, generated::SHAPE_BOX_TABLE[range.offset + i]);
    }
    return shape;
}

constexpr GeometryCatalog build_geometry_catalog() {
    CatalogBuilder builder{};
    builder.catalog.shape_names = generated::SHAPE_NAME_TABLE;

    for (std::size_t shape_index = 0; shape_index < GEOMETRY_SHAPE_COUNT; ++shape_index) {
        add_shape(builder, shape_boxes(shape_index));
    }

    return builder.catalog;
}

constexpr GeometryCatalog CATALOG = build_geometry_catalog();

constexpr std::uint32_t used_box_count() {
    std::uint32_t total = 0;
    for (const generated::ShapeBoxRange &range : generated::SHAPE_BOX_RANGES) {
        total += range.count;
    }
    return total;
}

constexpr std::uint32_t used_face_count() {
    std::uint32_t total = 0;
    for (const ShapeGeometry &shape : CATALOG.shapes) {
        total += shape.face_count;
    }
    return total;
}

static_assert(used_box_count() == GEOMETRY_BOX_COUNT, "geometry box count changed");
static_assert(used_face_count() == GEOMETRY_FACE_COUNT, "geometry face count changed");

}  // namespace

const GeometryCatalog &geometry_catalog() {
    return CATALOG;
}

const ShapeGeometry &geometry_for_shape(std::int32_t shape_id) {
    if (shape_id < 0 || static_cast<std::size_t>(shape_id) >= CATALOG.shapes.size()) {
        return CATALOG.shapes[static_cast<std::size_t>(SHAPE_FULL_CUBE)];
    }
    return CATALOG.shapes[static_cast<std::size_t>(shape_id)];
}

std::int32_t geometry_catalog_shape_count() {
    return static_cast<std::int32_t>(CATALOG.shapes.size());
}

const std::array<const char *, GEOMETRY_SHAPE_COUNT> &shape_names() {
    return CATALOG.shape_names;
}

const char *shape_id_name(std::int32_t shape_id) {
    if (shape_id < 0 || static_cast<std::size_t>(shape_id) >= CATALOG.shape_names.size()) {
        return "unknown_shape";
    }
    return CATALOG.shape_names[static_cast<std::size_t>(shape_id)];
}

std::uint8_t shape_box_count(std::int32_t shape_id) {
    if (shape_id < 0 || static_cast<std::size_t>(shape_id) >= generated::SHAPE_BOX_RANGES.size()) {
        return generated::SHAPE_BOX_RANGES[static_cast<std::size_t>(SHAPE_FULL_CUBE)].count;
    }
    return generated::SHAPE_BOX_RANGES[static_cast<std::size_t>(shape_id)].count;
}

std::int32_t shape_count() {
    return static_cast<std::int32_t>(CATALOG.shape_names.size());
}

}
