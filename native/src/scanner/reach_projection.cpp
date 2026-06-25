#include "minescript_miner/scanner/reach_projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace minescript_miner {

namespace {

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
               local_cross(
                   hull[count - 2],
                   hull[count - 1],
                   points.points[i]
               ) <= 0.0) {
            --count;
        }
        hull[count++] = points.points[i];
    }

    const std::size_t lower_count = count;
    for (std::size_t i = points.count - 1; i > 0; --i) {
        const Point2 point = points.points[i - 1];
        while (count > lower_count &&
               local_cross(hull[count - 2], hull[count - 1], point) <=
                   0.0) {
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

LocalFace make_local_face(const WorldRectFace &face, const Vec3 &eye) {
    const double coord =
        static_cast<double>(face.coord) / GEOMETRY_UNITS_PER_BLOCK;
    const double u_min =
        static_cast<double>(face.u_min) / GEOMETRY_UNITS_PER_BLOCK;
    const double u_max =
        static_cast<double>(face.u_max) / GEOMETRY_UNITS_PER_BLOCK;
    const double v_min =
        static_cast<double>(face.v_min) / GEOMETRY_UNITS_PER_BLOCK;
    const double v_max =
        static_cast<double>(face.v_max) / GEOMETRY_UNITS_PER_BLOCK;
    switch (face.axis) {
        case PlaneAxis::X:
            return {
                coord,
                u_min,
                u_max,
                v_min,
                v_max,
                eye.x,
                eye.y,
                eye.z,
            };
        case PlaneAxis::Y:
            return {
                coord,
                u_min,
                u_max,
                v_min,
                v_max,
                eye.y,
                eye.x,
                eye.z,
            };
        case PlaneAxis::Z:
            return {
                coord,
                u_min,
                u_max,
                v_min,
                v_max,
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

    const double varying_delta =
        std::sqrt(std::max(0.0, varying_squared));
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
    const WorldRectFace &face,
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

}  // namespace

bool make_reachable_world_face_pieces(
    const WorldRectFace &face,
    const Vec3 &eye,
    double reach,
    ReachableWorldFacePieces &out
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

    if (full_face) {
        WorldFacePolygon &polygon = out.faces[0];
        polygon.points[0] = world_point_to_vec3(face_p0(face));
        polygon.points[1] = world_point_to_vec3(face_p1(face));
        polygon.points[2] = world_point_to_vec3(face_p2(face));
        polygon.points[3] = world_point_to_vec3(face_p3(face));
        polygon.count = 4;
        out.count = 1;
        return true;
    }

    const LocalFace local = make_local_face(face, eye);
    for (std::uint8_t i = 1; i + 1 < reachable.count; ++i) {
        if (out.count >= out.faces.size()) {
            out = {};
            return false;
        }
        WorldFacePolygon &triangle = out.faces[out.count++];
        triangle.points[0] =
            local_to_world(local, face.axis, reachable.points[0]);
        triangle.points[1] =
            local_to_world(local, face.axis, reachable.points[i]);
        triangle.points[2] =
            local_to_world(local, face.axis, reachable.points[i + 1]);
        triangle.count = 3;
    }
    return out.count > 0;
}

bool project_reachable_world_face(
    const WorldRectFace &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    double reach,
    ProjectedFacePieces &out
) {
    out = {};
    ReachableWorldFacePieces reachable{};
    if (!make_reachable_world_face_pieces(
            face,
            eye,
            reach,
            reachable
        )) {
        return false;
    }

    ProjectedFace full_projection{};
    if (!project_world_face(face, eye, basis, full_projection)) {
        return false;
    }

    for (std::uint8_t i = 0; i < reachable.count; ++i) {
        if (out.count >= out.faces.size()) {
            out = {};
            return false;
        }
        const WorldFacePolygon &world_polygon = reachable.faces[i];
        ProjectedFace projection{};
        std::array<ViewPoint, MAX_CLIP_VERTICES> view_points{};
        if (world_polygon.count > view_points.size()) {
            out = {};
            return false;
        }
        for (std::uint8_t point_index = 0;
             point_index < world_polygon.count;
             ++point_index) {
            view_points[point_index] = world_to_view(
                world_polygon.points[point_index],
                eye,
                basis
            );
        }
        if (!project_view_polygon(
                view_points.data(),
                world_polygon.count,
                projection
            )) {
            continue;
        }
        projection.inverse_depth = full_projection.inverse_depth;
        out.faces[out.count++] = projection;
    }
    return out.count > 0;
}

}  // namespace minescript_miner
