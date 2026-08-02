#pragma once

#include "minecraft_miner/geometry/tri2.hpp"
#include "minecraft_miner/geometry/vec.hpp"

#include <vector>

namespace minecraft_miner::aim {

using VisibleDirectionComponent = std::vector<Vec3>;
using VisibleDirectionComponents = std::vector<VisibleDirectionComponent>;

struct TargetProjection {
    Vec3 right{};
    Vec3 up{};
    Vec3 forward{};
};

struct ProjectedTargetComponent {
    std::vector<Point2> vertices{};
    double orientation = 0.0;
};

struct ProjectedTargetRegion {
    TargetProjection projection{};
    std::vector<ProjectedTargetComponent> components{};
};

bool make_target_projection(
    const Vec3 &center_direction,
    TargetProjection &out
);

bool project_target_direction(
    const TargetProjection &projection,
    const Vec3 &direction,
    Point2 &out
);

bool project_visible_target_region(
    const Vec3 &center_direction,
    const VisibleDirectionComponents &components,
    ProjectedTargetRegion &out
);

// The visible region is closed: points on a component boundary count as hits.
bool point_in_visible_region(
    const ProjectedTargetRegion &region,
    const Vec3 &direction
);

bool point_in_visible_region_with_margin(
    const ProjectedTargetRegion &region,
    const Vec3 &direction,
    double margin
);

bool closest_safe_direction_in_visible_region(
    const ProjectedTargetRegion &region,
    const Vec3 &direction,
    double margin,
    Vec3 &out
);

}  // namespace minecraft_miner::aim
