#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minescript_miner {

inline constexpr int GEOMETRY_CATALOG_VERSION = 1;

inline constexpr std::int32_t SHAPE_EMPTY = 0;
inline constexpr std::int32_t SHAPE_FULL_CUBE = 1;
inline constexpr std::int32_t SHAPE_SLAB_BOTTOM = 2;
inline constexpr std::int32_t SHAPE_SLAB_TOP = 3;

enum class PlaneAxis : std::int32_t {
    X = 0,
    Y = 1,
    Z = 2,
};

struct Aabb {
    double min_x;
    double min_y;
    double min_z;
    double max_x;
    double max_y;
    double max_z;
};

struct RectFace {
    PlaneAxis axis;
    double coord;
    double u_min;
    double u_max;
    double v_min;
    double v_max;
    int normal_sign;
};

struct ShapeGeometry {
    std::uint32_t box_offset;
    std::uint32_t box_count;
    std::uint32_t face_offset;
    std::uint32_t face_count;
};

struct GeometryCatalog {
    std::vector<std::string> shape_names;
    std::vector<ShapeGeometry> shapes;
    std::vector<Aabb> boxes;
    std::vector<RectFace> faces;
};

const GeometryCatalog &geometry_catalog();
const ShapeGeometry &geometry_for_shape(std::int32_t shape_id);
std::int32_t geometry_catalog_shape_count();

std::string shape_id_name(std::int32_t shape_id);
std::int32_t shape_count();
const std::vector<std::string> &shape_names();

inline bool is_empty_shape(std::int32_t shape_id) {
    return shape_id == SHAPE_EMPTY;
}

}
