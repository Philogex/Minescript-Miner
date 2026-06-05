#pragma once

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

struct ProjectedFace {
    std::array<ProjectedPoint, 4> points{};
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

bool project_world_face(
    const WorldRectFace16 &face,
    const Vec3 &eye,
    const ViewBasis &basis,
    ProjectedFace &out
);

}
