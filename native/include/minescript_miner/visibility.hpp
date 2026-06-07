#pragma once

#include "minescript_miner/clipping.hpp"
#include "minescript_miner/scan_region.hpp"
#include "minescript_miner/tri2.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace minescript_miner {

// Positive view-space depth prevents singular perspective projections.
inline constexpr double PROJECTION_NEAR_DEPTH = 1.0e-6;
inline constexpr std::size_t MAX_REACH_FACE_VERTICES = 16;
inline constexpr std::size_t MAX_REACH_FACE_PIECES =
    MAX_REACH_FACE_VERTICES - 2;

struct ViewBasis {
    Vec3 right{};
    Vec3 up{};
    Vec3 forward{};
};

struct ViewPoint {
    double x = 0.0;
    double y = 0.0;
    double depth = 0.0;
};

struct ProjectedPoint {
    Point2 point{};
    double depth = 0.0;
};

struct InverseDepthPlane {
    double x = 0.0;
    double y = 0.0;
    double constant = 0.0;
};

struct ProjectedFace {
    std::array<ProjectedPoint, MAX_CLIP_VERTICES> points{};
    std::uint8_t count = 0;
    InverseDepthPlane inverse_depth{};
};

struct ProjectedFacePieces {
    std::array<ProjectedFace, MAX_REACH_FACE_PIECES> faces{};
    std::uint8_t count = 0;
};

bool make_view_basis(const Vec3 &forward, ViewBasis &out);
bool make_view_basis_toward(const Vec3 &eye, const Vec3 &target, ViewBasis &out);

constexpr ViewPoint world_to_view(
    const Vec3 &world_point,
    const Vec3 &eye,
    const ViewBasis &basis
) {
    const Vec3 relative = world_point - eye;
    return {
        dot(relative, basis.right),
        dot(relative, basis.up),
        dot(relative, basis.forward),
    };
}

constexpr ProjectedPoint project_view_point(const ViewPoint &point) {
    return {
        {point.x / point.depth, point.y / point.depth},
        point.depth,
    };
}

bool make_inverse_depth_plane(
    const ProjectedPoint &a,
    const ProjectedPoint &b,
    const ProjectedPoint &c,
    InverseDepthPlane &out
);

constexpr double inverse_depth_at(const InverseDepthPlane &plane, Point2 point) {
    return plane.x * point.x + plane.y * point.y + plane.constant;
}

constexpr LinearHalfPlane2 depth_difference_half_plane(
    const InverseDepthPlane &candidate,
    const InverseDepthPlane &reference
) {
    return {
        candidate.x - reference.x,
        candidate.y - reference.y,
        candidate.constant - reference.constant,
    };
}

constexpr double depth_difference_at(
    const InverseDepthPlane &candidate,
    const InverseDepthPlane &reference,
    Point2 point
) {
    return half_plane_value(depth_difference_half_plane(candidate, reference), point);
}

constexpr bool is_in_front_at(
    const InverseDepthPlane &candidate,
    const InverseDepthPlane &reference,
    Point2 point
) {
    return depth_difference_at(candidate, reference, point) > 0.0;
}

constexpr Polygon2 projected_face_polygon(const ProjectedFace &face) {
    Polygon2 polygon{};
    polygon.count = face.count;
    for (std::uint8_t i = 0; i < face.count; ++i) {
        polygon.points[i] = face.points[i].point;
    }
    return polygon;
}

bool clip_projected_face_in_front(
    const ProjectedFace &candidate,
    const ProjectedFace &reference,
    Polygon2 &out
);

bool project_world_face(
    const WorldRectFace16 &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    ProjectedFace &out
);

bool project_reachable_world_face(
    const WorldRectFace16 &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    double reach,
    ProjectedFacePieces &out
);

}
