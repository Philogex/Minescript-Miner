#pragma once

#include "minescript_miner/geometry/tri2.hpp"
#include "minescript_miner/scanner/scan_region.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace minescript_miner {

// Positive view-space depth prevents singular perspective projections.
inline constexpr double PROJECTION_NEAR_DEPTH = 1.0e-6;
inline constexpr std::size_t MAX_CLIP_VERTICES = 8;

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

bool project_view_polygon(
    const ViewPoint *points,
    std::uint8_t count,
    ProjectedFace &out
);

bool project_world_face(
    const WorldRectFace &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    ProjectedFace &out
);

}
