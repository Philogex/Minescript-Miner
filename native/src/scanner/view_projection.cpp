#include "minecraft_miner/scanner/view_projection.hpp"

#include "minecraft_miner/geometry/clipping.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace minecraft_miner {

namespace {

struct ViewPolygon {
    std::array<ViewPoint, MAX_CLIP_VERTICES> points{};
    std::uint8_t count = 0;
};

bool normalize(const Vec3 &value, Vec3 &out) {
    const double squared_length = length_squared(value);
    if (squared_length <= 0.0) {
        return false;
    }
    out = value * (1.0 / std::sqrt(squared_length));
    return true;
}

bool same_view_point(const ViewPoint &lhs, const ViewPoint &rhs) {
    return lhs.x == rhs.x &&
           lhs.y == rhs.y &&
           lhs.depth == rhs.depth;
}

bool append_view_point(ViewPolygon &polygon, const ViewPoint &point) {
    if (polygon.count > 0 &&
        same_view_point(polygon.points[polygon.count - 1], point)) {
        return true;
    }
    if (polygon.count >= polygon.points.size()) {
        return false;
    }
    polygon.points[polygon.count++] = point;
    return true;
}

ViewPoint intersect_near_plane(const ViewPoint &from, const ViewPoint &to) {
    const double t =
        (PROJECTION_NEAR_DEPTH - from.depth) /
        (to.depth - from.depth);
    return {
        from.x + t * (to.x - from.x),
        from.y + t * (to.y - from.y),
        PROJECTION_NEAR_DEPTH,
    };
}

bool clip_view_polygon_to_near_plane(const ViewPolygon &input, ViewPolygon &out) {
    out = {};
    if (input.count == 0) {
        return true;
    }

    ViewPoint previous = input.points[input.count - 1];
    bool previous_inside = previous.depth >= PROJECTION_NEAR_DEPTH;

    for (std::uint8_t i = 0; i < input.count; ++i) {
        const ViewPoint current = input.points[i];
        const bool current_inside =
            current.depth >= PROJECTION_NEAR_DEPTH;
        if (previous_inside != current_inside &&
            !append_view_point(
                out,
                intersect_near_plane(previous, current)
            )) {
            return false;
        }
        if (current_inside && !append_view_point(out, current)) {
            return false;
        }

        previous = current;
        previous_inside = current_inside;
    }

    if (out.count > 1 &&
        same_view_point(out.points[0], out.points[out.count - 1])) {
        --out.count;
    }
    return true;
}

bool project_view_polygon_impl(const ViewPolygon &input, ProjectedFace &out) {
    ViewPolygon clipped{};
    if (!clip_view_polygon_to_near_plane(input, clipped) ||
        clipped.count < 3) {
        out = {};
        return false;
    }

    out = {};
    out.count = clipped.count;
    for (std::uint8_t i = 0; i < clipped.count; ++i) {
        out.points[i] = project_view_point(clipped.points[i]);
        if (!std::isfinite(out.points[i].point.x) ||
            !std::isfinite(out.points[i].point.y)) {
            out = {};
            return false;
        }
    }

    for (std::uint8_t i = 1; i + 1 < out.count; ++i) {
        if (make_inverse_depth_plane(
                out.points[0],
                out.points[i],
                out.points[i + 1],
                out.inverse_depth
            )) {
            return true;
        }
    }

    out = {};
    return false;
}

}  // namespace

bool make_view_basis(const Vec3 &forward, ViewBasis &out) {
    Vec3 normalized_forward{};
    if (!normalize(forward, normalized_forward)) {
        return false;
    }

    const Vec3 preferred_up =
        std::abs(normalized_forward.y) < 0.999
            ? Vec3{0.0, 1.0, 0.0}
            : Vec3{0.0, 0.0, 1.0};

    Vec3 right{};
    if (!normalize(cross(preferred_up, normalized_forward), right)) {
        return false;
    }

    out = {
        right,
        cross(normalized_forward, right),
        normalized_forward,
    };
    return true;
}

bool make_view_basis_toward(const Vec3 &eye, const Vec3 &target, ViewBasis &out) {
    return make_view_basis(target - eye, out);
}

bool make_inverse_depth_plane(
    const ProjectedPoint &a,
    const ProjectedPoint &b,
    const ProjectedPoint &c,
    InverseDepthPlane &out
) {
    if (a.depth <= 0.0 || b.depth <= 0.0 || c.depth <= 0.0) {
        return false;
    }

    if (orient2d(a.point, b.point, c.point) == Orientation::Collinear) {
        return false;
    }
    const double denominator =
        orient2d_determinant(a.point, b.point, c.point);
    if (denominator == 0.0) {
        return false;
    }

    const double inverse_a = 1.0 / a.depth;
    const double inverse_b = 1.0 / b.depth;
    const double inverse_c = 1.0 / c.depth;
    const double delta_b = inverse_b - inverse_a;
    const double delta_c = inverse_c - inverse_a;
    const double bx = b.point.x - a.point.x;
    const double by = b.point.y - a.point.y;
    const double cx = c.point.x - a.point.x;
    const double cy = c.point.y - a.point.y;

    out.x = (delta_b * cy - delta_c * by) / denominator;
    out.y = (bx * delta_c - cx * delta_b) / denominator;
    out.constant = inverse_a - out.x * a.point.x - out.y * a.point.y;
    return true;
}

bool project_view_polygon(
    const ViewPoint *points,
    std::uint8_t count,
    ProjectedFace &out
) {
    if (count > MAX_CLIP_VERTICES) {
        out = {};
        return false;
    }

    ViewPolygon view_polygon{};
    view_polygon.count = count;
    for (std::uint8_t i = 0; i < count; ++i) {
        view_polygon.points[i] = points[i];
    }
    return project_view_polygon_impl(view_polygon, out);
}

bool project_world_face(
    const WorldRectFace &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    ProjectedFace &out
) {
    const std::array<WorldPoint, 4> world_points{
        face_p0(face),
        face_p1(face),
        face_p2(face),
        face_p3(face),
    };
    ViewPolygon view_polygon{};
    view_polygon.count = static_cast<std::uint8_t>(world_points.size());

    for (std::size_t i = 0; i < world_points.size(); ++i) {
        view_polygon.points[i] =
            world_to_view(world_point_to_vec3(world_points[i]), eye, basis);
    }
    return project_view_polygon_impl(view_polygon, out);
}

namespace {

constexpr ViewBasis TEST_BASIS{
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};
constexpr ViewPoint TEST_VIEW_POINT = world_to_view({2.0, 4.0, 2.0}, {}, TEST_BASIS);
constexpr ProjectedPoint TEST_PROJECTED_POINT = project_view_point(TEST_VIEW_POINT);

static_assert(TEST_VIEW_POINT.x == 2.0);
static_assert(TEST_VIEW_POINT.y == 4.0);
static_assert(TEST_VIEW_POINT.depth == 2.0);
static_assert(TEST_PROJECTED_POINT.point.x == 1.0);
static_assert(TEST_PROJECTED_POINT.point.y == 2.0);

}  // namespace

}  // namespace minecraft_miner
