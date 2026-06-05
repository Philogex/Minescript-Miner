#pragma once

#include "minescript_miner/clipping.hpp"
#include "minescript_miner/scan_region.hpp"
#include "minescript_miner/tri2.hpp"

#include <array>

namespace minescript_miner {

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
    std::array<ProjectedPoint, 4> points{};
    InverseDepthPlane inverse_depth{};
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

constexpr bool is_in_front(const ViewPoint &point) {
    return point.depth > 0.0;
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
    return polygon_from_quad({
        face.points[0].point,
        face.points[1].point,
        face.points[2].point,
        face.points[3].point,
    });
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

}
