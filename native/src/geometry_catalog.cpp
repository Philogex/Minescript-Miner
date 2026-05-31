#include "minescript_miner/geometry_catalog.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minescript_miner {

namespace {

constexpr double EPS = 1.0e-9;

struct GeneratedShape {
    std::string name;
    std::vector<Aabb> boxes;
    std::vector<RectFace> faces;
};

Aabb box(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z) {
    return {min_x, min_y, min_z, max_x, max_y, max_z};
}

double axis_min(const Aabb &b, PlaneAxis axis) {
    switch (axis) {
        case PlaneAxis::X:
            return b.min_x;
        case PlaneAxis::Y:
            return b.min_y;
        case PlaneAxis::Z:
            return b.min_z;
    }
    return 0.0;
}

double axis_max(const Aabb &b, PlaneAxis axis) {
    switch (axis) {
        case PlaneAxis::X:
            return b.max_x;
        case PlaneAxis::Y:
            return b.max_y;
        case PlaneAxis::Z:
            return b.max_z;
    }
    return 0.0;
}

bool contains_point(const Aabb &b, const std::array<double, 3> &point) {
    return point[0] > b.min_x + EPS && point[0] < b.max_x - EPS &&
           point[1] > b.min_y + EPS && point[1] < b.max_y - EPS &&
           point[2] > b.min_z + EPS && point[2] < b.max_z - EPS;
}

bool overlaps_1d(double a_min, double a_max, double b_min, double b_max) {
    return a_min < b_max - EPS && b_min < a_max - EPS;
}

std::array<PlaneAxis, 2> uv_axes(PlaneAxis axis) {
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

std::array<double, 3> point_on_face(const RectFace &face, double u, double v, double offset) {
    std::array<double, 3> point = {0.0, 0.0, 0.0};
    const auto axes = uv_axes(face.axis);
    point[static_cast<std::size_t>(face.axis)] = face.coord + offset;
    point[static_cast<std::size_t>(axes[0])] = u;
    point[static_cast<std::size_t>(axes[1])] = v;
    return point;
}

void add_split(std::vector<double> &values, double value, double min_value, double max_value) {
    if (value > min_value + EPS && value < max_value - EPS) {
        values.push_back(value);
    }
}

std::vector<double> sorted_unique(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    values.erase(
        std::unique(
            values.begin(),
            values.end(),
            [](double a, double b) {
                return std::abs(a - b) <= EPS;
            }
        ),
        values.end()
    );
    return values;
}

bool outside_occupied(const RectFace &face, double u, double v, const std::vector<Aabb> &boxes) {
    const double offset = static_cast<double>(face.normal_sign) * 1.0e-7;
    const auto point = point_on_face(face, u, v, offset);
    for (const Aabb &b : boxes) {
        if (contains_point(b, point)) {
            return true;
        }
    }
    return false;
}

void add_face_cells(const RectFace &face, const std::vector<Aabb> &boxes, std::vector<RectFace> &faces) {
    const auto axes = uv_axes(face.axis);
    std::vector<double> u_splits = {face.u_min, face.u_max};
    std::vector<double> v_splits = {face.v_min, face.v_max};

    for (const Aabb &b : boxes) {
        if (axis_min(b, face.axis) > face.coord + EPS || axis_max(b, face.axis) < face.coord - EPS) {
            continue;
        }

        const double box_u_min = axis_min(b, axes[0]);
        const double box_u_max = axis_max(b, axes[0]);
        const double box_v_min = axis_min(b, axes[1]);
        const double box_v_max = axis_max(b, axes[1]);
        if (!overlaps_1d(face.u_min, face.u_max, box_u_min, box_u_max) ||
            !overlaps_1d(face.v_min, face.v_max, box_v_min, box_v_max)) {
            continue;
        }

        add_split(u_splits, box_u_min, face.u_min, face.u_max);
        add_split(u_splits, box_u_max, face.u_min, face.u_max);
        add_split(v_splits, box_v_min, face.v_min, face.v_max);
        add_split(v_splits, box_v_max, face.v_min, face.v_max);
    }

    u_splits = sorted_unique(std::move(u_splits));
    v_splits = sorted_unique(std::move(v_splits));

    for (std::size_t u_index = 0; u_index + 1 < u_splits.size(); ++u_index) {
        for (std::size_t v_index = 0; v_index + 1 < v_splits.size(); ++v_index) {
            const double u_min = u_splits[u_index];
            const double u_max = u_splits[u_index + 1];
            const double v_min = v_splits[v_index];
            const double v_max = v_splits[v_index + 1];
            if (u_max <= u_min + EPS || v_max <= v_min + EPS) {
                continue;
            }

            const double u_mid = (u_min + u_max) * 0.5;
            const double v_mid = (v_min + v_max) * 0.5;
            if (outside_occupied(face, u_mid, v_mid, boxes)) {
                continue;
            }

            faces.push_back({face.axis, face.coord, u_min, u_max, v_min, v_max, face.normal_sign});
        }
    }
}

void add_box_faces(const Aabb &b, const std::vector<Aabb> &boxes, std::vector<RectFace> &faces) {
    add_face_cells({PlaneAxis::X, b.min_x, b.min_y, b.max_y, b.min_z, b.max_z, -1}, boxes, faces);
    add_face_cells({PlaneAxis::X, b.max_x, b.min_y, b.max_y, b.min_z, b.max_z, 1}, boxes, faces);
    add_face_cells({PlaneAxis::Y, b.min_y, b.min_x, b.max_x, b.min_z, b.max_z, -1}, boxes, faces);
    add_face_cells({PlaneAxis::Y, b.max_y, b.min_x, b.max_x, b.min_z, b.max_z, 1}, boxes, faces);
    add_face_cells({PlaneAxis::Z, b.min_z, b.min_x, b.max_x, b.min_y, b.max_y, -1}, boxes, faces);
    add_face_cells({PlaneAxis::Z, b.max_z, b.min_x, b.max_x, b.min_y, b.max_y, 1}, boxes, faces);
}

GeneratedShape shape_from_boxes(std::string name, std::vector<Aabb> boxes) {
    std::vector<RectFace> faces;
    for (const Aabb &b : boxes) {
        add_box_faces(b, boxes, faces);
    }
    return {std::move(name), std::move(boxes), std::move(faces)};
}

std::string connection_mask_name(int mask) {
    static constexpr const char *directions[] = {"north", "east", "south", "west"};

    std::string name;
    for (int bit = 0; bit < 4; ++bit) {
        if ((mask & (1 << bit)) == 0) {
            continue;
        }
        if (!name.empty()) {
            name += "_";
        }
        name += directions[bit];
    }
    return name.empty() ? "none" : name;
}

std::pair<double, double> half_bounds_1d(const std::string &direction, bool front, bool x_axis) {
    if (direction == "north") {
        return x_axis ? std::pair{0.0, 1.0} : (front ? std::pair{0.0, 0.5} : std::pair{0.5, 1.0});
    }
    if (direction == "south") {
        return x_axis ? std::pair{0.0, 1.0} : (front ? std::pair{0.5, 1.0} : std::pair{0.0, 0.5});
    }
    if (direction == "east") {
        return x_axis ? (front ? std::pair{0.5, 1.0} : std::pair{0.0, 0.5}) : std::pair{0.0, 1.0};
    }
    if (direction == "west") {
        return x_axis ? (front ? std::pair{0.0, 0.5} : std::pair{0.5, 1.0}) : std::pair{0.0, 1.0};
    }
    throw std::logic_error("unknown stair direction");
}

std::string lateral_direction(const std::string &direction, const std::string &side) {
    if (side == "left") {
        if (direction == "north") {
            return "west";
        }
        if (direction == "east") {
            return "north";
        }
        if (direction == "south") {
            return "east";
        }
        if (direction == "west") {
            return "south";
        }
    } else {
        if (direction == "north") {
            return "east";
        }
        if (direction == "east") {
            return "south";
        }
        if (direction == "south") {
            return "west";
        }
        if (direction == "west") {
            return "north";
        }
    }
    throw std::logic_error("unknown stair direction/side");
}

Aabb stair_quadrant(const std::string &direction, const std::string &side, bool front, double y_min, double y_max) {
    const auto front_x = half_bounds_1d(direction, front, true);
    const auto front_z = half_bounds_1d(direction, front, false);
    const std::string lateral = lateral_direction(direction, side);
    const auto lateral_x = half_bounds_1d(lateral, true, true);
    const auto lateral_z = half_bounds_1d(lateral, true, false);
    return box(
        std::max(front_x.first, lateral_x.first),
        y_min,
        std::max(front_z.first, lateral_z.first),
        std::min(front_x.second, lateral_x.second),
        y_max,
        std::min(front_z.second, lateral_z.second)
    );
}

std::vector<Aabb> stair_boxes(const std::string &direction, const std::string &half, const std::string &stair_shape) {
    std::vector<Aabb> boxes;
    double y_min = 0.5;
    double y_max = 1.0;
    if (half == "bottom") {
        boxes.push_back(box(0.0, 0.0, 0.0, 1.0, 0.5, 1.0));
    } else {
        boxes.push_back(box(0.0, 0.5, 0.0, 1.0, 1.0, 1.0));
        y_min = 0.0;
        y_max = 0.5;
    }

    if (stair_shape == "straight") {
        boxes.push_back(stair_quadrant(direction, "left", true, y_min, y_max));
        boxes.push_back(stair_quadrant(direction, "right", true, y_min, y_max));
    } else if (stair_shape == "outer_left") {
        boxes.push_back(stair_quadrant(direction, "left", true, y_min, y_max));
    } else if (stair_shape == "outer_right") {
        boxes.push_back(stair_quadrant(direction, "right", true, y_min, y_max));
    } else if (stair_shape == "inner_left") {
        boxes.push_back(stair_quadrant(direction, "left", true, y_min, y_max));
        boxes.push_back(stair_quadrant(direction, "right", true, y_min, y_max));
        boxes.push_back(stair_quadrant(direction, "left", false, y_min, y_max));
    } else if (stair_shape == "inner_right") {
        boxes.push_back(stair_quadrant(direction, "left", true, y_min, y_max));
        boxes.push_back(stair_quadrant(direction, "right", true, y_min, y_max));
        boxes.push_back(stair_quadrant(direction, "right", false, y_min, y_max));
    } else {
        throw std::logic_error("unknown stair shape");
    }
    return boxes;
}

std::vector<Aabb> connection_boxes(int mask, double post_width, double arm_width) {
    const double half_post = post_width * 0.5;
    const double half_arm = arm_width * 0.5;
    const double center_min = 0.5 - half_post;
    const double center_max = 0.5 + half_post;
    const double arm_min = 0.5 - half_arm;
    const double arm_max = 0.5 + half_arm;

    std::vector<Aabb> boxes = {
        box(center_min, 0.0, center_min, center_max, 1.0, center_max),
    };
    if ((mask & 1) != 0) {
        boxes.push_back(box(arm_min, 0.0, 0.0, arm_max, 1.0, center_min));
    }
    if ((mask & 2) != 0) {
        boxes.push_back(box(center_max, 0.0, arm_min, 1.0, 1.0, arm_max));
    }
    if ((mask & 4) != 0) {
        boxes.push_back(box(arm_min, 0.0, center_max, arm_max, 1.0, 1.0));
    }
    if ((mask & 8) != 0) {
        boxes.push_back(box(0.0, 0.0, arm_min, center_min, 1.0, arm_max));
    }
    return boxes;
}

std::vector<GeneratedShape> generate_shapes() {
    static constexpr const char *directions[] = {"north", "east", "south", "west"};
    static constexpr const char *halves[] = {"bottom", "top"};
    static constexpr const char *stair_shapes[] = {
        "straight",
        "inner_left",
        "inner_right",
        "outer_left",
        "outer_right",
    };

    std::vector<GeneratedShape> generated;
    generated.push_back(shape_from_boxes("empty", {}));
    generated.push_back(shape_from_boxes("full_cube", {box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)}));
    generated.push_back(shape_from_boxes("slab_bottom", {box(0.0, 0.0, 0.0, 1.0, 0.5, 1.0)}));
    generated.push_back(shape_from_boxes("slab_top", {box(0.0, 0.5, 0.0, 1.0, 1.0, 1.0)}));

    for (const char *direction : directions) {
        for (const char *half : halves) {
            for (const char *stair_shape : stair_shapes) {
                generated.push_back(
                    shape_from_boxes(
                        std::string("stairs_") + direction + "_" + half + "_" + stair_shape,
                        stair_boxes(direction, half, stair_shape)
                    )
                );
            }
        }
    }

    for (int mask = 0; mask < 16; ++mask) {
        generated.push_back(
            shape_from_boxes("pane_" + connection_mask_name(mask), connection_boxes(mask, 2.0 / 16.0, 2.0 / 16.0))
        );
    }

    for (int mask = 0; mask < 16; ++mask) {
        generated.push_back(
            shape_from_boxes("fence_" + connection_mask_name(mask), connection_boxes(mask, 4.0 / 16.0, 4.0 / 16.0))
        );
    }

    return generated;
}

GeometryCatalog build_geometry_catalog() {
    GeometryCatalog catalog;
    const std::vector<GeneratedShape> generated = generate_shapes();
    catalog.shape_names.reserve(generated.size());
    catalog.shapes.reserve(generated.size());

    for (const GeneratedShape &shape : generated) {
        const auto box_offset = static_cast<std::uint32_t>(catalog.boxes.size());
        const auto face_offset = static_cast<std::uint32_t>(catalog.faces.size());
        const auto box_count = static_cast<std::uint32_t>(shape.boxes.size());
        const auto face_count = static_cast<std::uint32_t>(shape.faces.size());

        catalog.shape_names.push_back(shape.name);
        catalog.shapes.push_back({box_offset, box_count, face_offset, face_count});
        catalog.boxes.insert(catalog.boxes.end(), shape.boxes.begin(), shape.boxes.end());
        catalog.faces.insert(catalog.faces.end(), shape.faces.begin(), shape.faces.end());
    }

    return catalog;
}

}  // namespace

const GeometryCatalog &geometry_catalog() {
    static const GeometryCatalog catalog = build_geometry_catalog();
    return catalog;
}

const ShapeGeometry &geometry_for_shape(std::int32_t shape_id) {
    const GeometryCatalog &catalog = geometry_catalog();
    if (shape_id < 0 || static_cast<std::size_t>(shape_id) >= catalog.shapes.size()) {
        return catalog.shapes[static_cast<std::size_t>(SHAPE_FULL_CUBE)];
    }
    return catalog.shapes[static_cast<std::size_t>(shape_id)];
}

std::int32_t geometry_catalog_shape_count() {
    return static_cast<std::int32_t>(geometry_catalog().shapes.size());
}

const std::vector<std::string> &shape_names() {
    return geometry_catalog().shape_names;
}

std::string shape_id_name(std::int32_t shape_id) {
    const auto &names = shape_names();
    if (shape_id < 0 || static_cast<std::size_t>(shape_id) >= names.size()) {
        return "unknown_shape";
    }
    return names[static_cast<std::size_t>(shape_id)];
}

std::int32_t shape_count() {
    return static_cast<std::int32_t>(shape_names().size());
}

}
