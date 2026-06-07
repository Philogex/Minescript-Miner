#include "minescript_miner/visibility.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace minescript_miner {

namespace {

struct ViewPolygon {
    std::array<ViewPoint, MAX_CLIP_VERTICES> points{};
    std::uint8_t count = 0;
};

struct ReachPolygon {
    std::array<Point2, MAX_REACH_FACE_VERTICES> points{};
    std::uint8_t count = 0;
};

struct LocalFace {
    double plane = 0.0;
    double u_min = 0.0;
    double u_max = 0.0;
    double v_min = 0.0;
    double v_max = 0.0;
    double eye_plane = 0.0;
    double eye_u = 0.0;
    double eye_v = 0.0;
};

constexpr std::array<Point2, MAX_REACH_FACE_VERTICES> UNIT_CIRCLE_16{{
    {1.0, 0.0},
    {0.9238795325112867, 0.3826834323650898},
    {0.7071067811865476, 0.7071067811865476},
    {0.3826834323650898, 0.9238795325112867},
    {0.0, 1.0},
    {-0.3826834323650898, 0.9238795325112867},
    {-0.7071067811865476, 0.7071067811865476},
    {-0.9238795325112867, 0.3826834323650898},
    {-1.0, 0.0},
    {-0.9238795325112867, -0.3826834323650898},
    {-0.7071067811865476, -0.7071067811865476},
    {-0.3826834323650898, -0.9238795325112867},
    {0.0, -1.0},
    {0.3826834323650898, -0.9238795325112867},
    {0.7071067811865476, -0.7071067811865476},
    {0.9238795325112867, -0.3826834323650898},
}};

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

bool project_view_polygon(const ViewPolygon &input, ProjectedFace &out) {
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

bool append_reach_point(ReachPolygon &polygon, Point2 point) {
    for (std::uint8_t i = 0; i < polygon.count; ++i) {
        if (polygon.points[i].x == point.x &&
            polygon.points[i].y == point.y) {
            return true;
        }
    }
    if (polygon.count >= polygon.points.size()) {
        return false;
    }
    polygon.points[polygon.count++] = point;
    return true;
}

double local_cross(Point2 a, Point2 b, Point2 point) {
    return (b.x - a.x) * (point.y - a.y) -
           (b.y - a.y) * (point.x - a.x);
}

bool convex_hull(ReachPolygon points, ReachPolygon &out) {
    out = {};
    if (points.count < 3) {
        return false;
    }

    std::sort(
        points.points.begin(),
        points.points.begin() + points.count,
        [](Point2 lhs, Point2 rhs) {
            return lhs.x < rhs.x ||
                   (lhs.x == rhs.x && lhs.y < rhs.y);
        }
    );

    std::array<Point2, MAX_REACH_FACE_VERTICES * 2> hull{};
    std::size_t count = 0;
    for (std::uint8_t i = 0; i < points.count; ++i) {
        while (count >= 2 &&
               local_cross(hull[count - 2], hull[count - 1], points.points[i]) <= 0.0) {
            --count;
        }
        hull[count++] = points.points[i];
    }

    const std::size_t lower_count = count;
    for (std::size_t i = points.count - 1; i > 0; --i) {
        const Point2 point = points.points[i - 1];
        while (count > lower_count &&
               local_cross(hull[count - 2], hull[count - 1], point) <= 0.0) {
            --count;
        }
        hull[count++] = point;
    }
    if (count > 1) {
        --count;
    }
    if (count < 3 || count > out.points.size()) {
        return false;
    }

    out.count = static_cast<std::uint8_t>(count);
    std::copy_n(hull.begin(), count, out.points.begin());
    return true;
}

LocalFace make_local_face(const WorldRectFace16 &face, const Vec3 &eye) {
    const Vec3 p0 = point16_to_world(face.p0);
    const Vec3 p2 = point16_to_world(face.p2);
    switch (face.axis) {
        case PlaneAxis::X:
            return {
                p0.x,
                std::min(p0.y, p2.y),
                std::max(p0.y, p2.y),
                std::min(p0.z, p2.z),
                std::max(p0.z, p2.z),
                eye.x,
                eye.y,
                eye.z,
            };
        case PlaneAxis::Y:
            return {
                p0.y,
                std::min(p0.x, p2.x),
                std::max(p0.x, p2.x),
                std::min(p0.z, p2.z),
                std::max(p0.z, p2.z),
                eye.y,
                eye.x,
                eye.z,
            };
        case PlaneAxis::Z:
            return {
                p0.z,
                std::min(p0.x, p2.x),
                std::max(p0.x, p2.x),
                std::min(p0.y, p2.y),
                std::max(p0.y, p2.y),
                eye.z,
                eye.x,
                eye.y,
            };
    }
    return {};
}

Vec3 local_to_world(const LocalFace &local, PlaneAxis axis, Point2 point) {
    switch (axis) {
        case PlaneAxis::X:
            return {local.plane, point.x, point.y};
        case PlaneAxis::Y:
            return {point.x, local.plane, point.y};
        case PlaneAxis::Z:
            return {point.x, point.y, local.plane};
    }
    return {};
}

double axis_distance(double value, double minimum, double maximum) {
    return std::max({minimum - value, 0.0, value - maximum});
}

bool append_edge_circle_intersections(
    ReachPolygon &points,
    double fixed,
    double varying_min,
    double varying_max,
    double center_fixed,
    double center_varying,
    double radius_squared,
    bool fixed_is_v
) {
    const double fixed_delta = fixed - center_fixed;
    const double varying_squared =
        radius_squared - fixed_delta * fixed_delta;
    if (varying_squared < 0.0) {
        return true;
    }

    const double varying_delta = std::sqrt(std::max(0.0, varying_squared));
    const double intersections[2]{
        center_varying - varying_delta,
        center_varying + varying_delta,
    };
    for (const double varying : intersections) {
        if (varying < varying_min || varying > varying_max) {
            continue;
        }
        const Point2 point =
            fixed_is_v ? Point2{varying, fixed} : Point2{fixed, varying};
        if (!append_reach_point(points, point)) {
            return false;
        }
    }
    return true;
}

bool make_reachable_face_polygon(
    const WorldRectFace16 &face,
    const Vec3 &eye,
    double reach,
    ReachPolygon &out,
    bool &full_face
) {
    out = {};
    full_face = false;
    if (std::isinf(reach) && reach > 0.0) {
        full_face = true;
        return true;
    }
    if (!(reach > 0.0) || !std::isfinite(reach)) {
        return false;
    }

    const double margin =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, reach);
    const double inner_reach = std::max(0.0, reach - margin);
    const double inner_reach_squared = inner_reach * inner_reach;
    const LocalFace local = make_local_face(face, eye);
    const double plane_delta = local.plane - local.eye_plane;
    const double plane_distance_squared = plane_delta * plane_delta;
    if (plane_distance_squared >= inner_reach_squared) {
        return false;
    }

    const double nearest_u =
        axis_distance(local.eye_u, local.u_min, local.u_max);
    const double nearest_v =
        axis_distance(local.eye_v, local.v_min, local.v_max);
    if (plane_distance_squared +
            nearest_u * nearest_u +
            nearest_v * nearest_v >
        inner_reach_squared) {
        return false;
    }

    const std::array<Point2, 4> corners{{
        {local.u_min, local.v_min},
        {local.u_max, local.v_min},
        {local.u_max, local.v_max},
        {local.u_min, local.v_max},
    }};
    full_face = std::all_of(
        corners.begin(),
        corners.end(),
        [&](Point2 corner) {
            const double du = corner.x - local.eye_u;
            const double dv = corner.y - local.eye_v;
            return plane_distance_squared + du * du + dv * dv <=
                   inner_reach_squared;
        }
    );
    if (full_face) {
        return true;
    }

    const double radius_squared =
        inner_reach_squared - plane_distance_squared;
    const double radius = std::sqrt(radius_squared);
    const bool circle_inside_face =
        local.eye_u - radius >= local.u_min &&
        local.eye_u + radius <= local.u_max &&
        local.eye_v - radius >= local.v_min &&
        local.eye_v + radius <= local.v_max;
    if (circle_inside_face) {
        out.count = static_cast<std::uint8_t>(UNIT_CIRCLE_16.size());
        for (std::size_t i = 0; i < UNIT_CIRCLE_16.size(); ++i) {
            out.points[i] = {
                local.eye_u + UNIT_CIRCLE_16[i].x * radius,
                local.eye_v + UNIT_CIRCLE_16[i].y * radius,
            };
        }
        return true;
    }

    ReachPolygon candidates{};
    for (const Point2 corner : corners) {
        const double du = corner.x - local.eye_u;
        const double dv = corner.y - local.eye_v;
        if (du * du + dv * dv <= radius_squared &&
            !append_reach_point(candidates, corner)) {
            return false;
        }
    }
    if (!append_edge_circle_intersections(
            candidates,
            local.v_min,
            local.u_min,
            local.u_max,
            local.eye_v,
            local.eye_u,
            radius_squared,
            true
        ) ||
        !append_edge_circle_intersections(
            candidates,
            local.v_max,
            local.u_min,
            local.u_max,
            local.eye_v,
            local.eye_u,
            radius_squared,
            true
        ) ||
        !append_edge_circle_intersections(
            candidates,
            local.u_min,
            local.v_min,
            local.v_max,
            local.eye_u,
            local.eye_v,
            radius_squared,
            false
        ) ||
        !append_edge_circle_intersections(
            candidates,
            local.u_max,
            local.v_min,
            local.v_max,
            local.eye_u,
            local.eye_v,
            radius_squared,
            false
        )) {
        return false;
    }
    return convex_hull(candidates, out);
}

bool project_world_triangle(
    const std::array<Vec3, 3> &triangle,
    const Vec3 &eye,
    const ViewBasis &basis,
    ProjectedFace &out
) {
    ViewPolygon view_polygon{};
    view_polygon.count = 3;
    for (std::uint8_t i = 0; i < view_polygon.count; ++i) {
        view_polygon.points[i] =
            world_to_view(triangle[i], eye, basis);
    }
    return project_view_polygon(view_polygon, out);
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

bool clip_projected_face_in_front(
    const ProjectedFace &candidate,
    const ProjectedFace &reference,
    Polygon2 &out
) {
    const LinearHalfPlane2 depth_difference =
        depth_difference_half_plane(candidate.inverse_depth, reference.inverse_depth);
    if (depth_difference.x == 0.0 &&
        depth_difference.y == 0.0 &&
        depth_difference.constant == 0.0) {
        out = {};
        return true;
    }
    return clip_half_plane(
        projected_face_polygon(candidate),
        depth_difference,
        out
    );
}

bool project_world_face(
    const WorldRectFace16 &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    ProjectedFace &out
) {
    const std::array<WorldPoint16, 4> world_points{face.p0, face.p1, face.p2, face.p3};
    ViewPolygon view_polygon{};
    view_polygon.count = static_cast<std::uint8_t>(world_points.size());

    for (std::size_t i = 0; i < world_points.size(); ++i) {
        view_polygon.points[i] =
            world_to_view(point16_to_world(world_points[i]), eye, basis);
    }
    return project_view_polygon(view_polygon, out);
}

bool project_reachable_world_face(
    const WorldRectFace16 &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    double reach,
    ProjectedFacePieces &out
) {
    out = {};
    ReachPolygon reachable{};
    bool full_face = false;
    if (!make_reachable_face_polygon(
            face,
            eye,
            reach,
            reachable,
            full_face
        )) {
        return false;
    }

    ProjectedFace full_projection{};
    if (!project_world_face(face, eye, basis, full_projection)) {
        return false;
    }
    if (full_face) {
        out.faces[0] = full_projection;
        out.count = 1;
        return true;
    }

    const LocalFace local = make_local_face(face, eye);
    for (std::uint8_t i = 1; i + 1 < reachable.count; ++i) {
        if (out.count >= out.faces.size()) {
            out = {};
            return false;
        }
        const std::array<Vec3, 3> world_triangle{{
            local_to_world(local, face.axis, reachable.points[0]),
            local_to_world(local, face.axis, reachable.points[i]),
            local_to_world(local, face.axis, reachable.points[i + 1]),
        }};
        ProjectedFace projection{};
        if (!project_world_triangle(
                world_triangle,
                eye,
                basis,
                projection
            )) {
            continue;
        }
        projection.inverse_depth = full_projection.inverse_depth;
        out.faces[out.count++] = projection;
    }
    return out.count > 0;
}

namespace {

constexpr ViewBasis TEST_BASIS{
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};
constexpr ViewPoint TEST_VIEW_POINT = world_to_view({2.0, 4.0, 2.0}, {}, TEST_BASIS);
constexpr ProjectedPoint TEST_PROJECTED_POINT = project_view_point(TEST_VIEW_POINT);
constexpr InverseDepthPlane TEST_NEAR_DEPTH{0.0, 0.0, 0.5};
constexpr InverseDepthPlane TEST_FAR_DEPTH{0.0, 0.0, 0.25};

static_assert(TEST_VIEW_POINT.x == 2.0);
static_assert(TEST_VIEW_POINT.y == 4.0);
static_assert(TEST_VIEW_POINT.depth == 2.0);
static_assert(TEST_PROJECTED_POINT.point.x == 1.0);
static_assert(TEST_PROJECTED_POINT.point.y == 2.0);
static_assert(depth_difference_at(TEST_NEAR_DEPTH, TEST_FAR_DEPTH, {0.0, 0.0}) == 0.25);
static_assert(is_in_front_at(TEST_NEAR_DEPTH, TEST_FAR_DEPTH, {0.0, 0.0}));

}  // namespace

}  // namespace minescript_miner
