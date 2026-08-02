#include "minecraft_miner/aim/target_region.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace minecraft_miner::aim {

namespace {

constexpr double PROJECTION_EPSILON = 1.0e-12;

bool finite(const Vec3 &value) {
    return std::isfinite(value.x) &&
           std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool finite(Point2 value) {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

bool normalize(const Vec3 &value, Vec3 &out) {
    if (!finite(value)) {
        return false;
    }
    const double magnitude_squared = length_squared(value);
    if (!(magnitude_squared > PROJECTION_EPSILON * PROJECTION_EPSILON) ||
        !std::isfinite(magnitude_squared)) {
        return false;
    }
    out = value * (1.0 / std::sqrt(magnitude_squared));
    return true;
}

double cross2(Point2 lhs, Point2 rhs) {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}

double polygon_orientation(const std::vector<Point2> &vertices) {
    double area_twice = 0.0;
    double coordinate_scale = 1.0;
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        const Point2 current = vertices[index];
        const Point2 next = vertices[(index + 1) % vertices.size()];
        area_twice += cross2(current, next);
        coordinate_scale = std::max({
            coordinate_scale,
            std::abs(current.x),
            std::abs(current.y),
        });
    }
    const double guard = PROJECTION_EPSILON * coordinate_scale *
        coordinate_scale * static_cast<double>(vertices.size());
    if (std::abs(area_twice) <= guard) {
        return 0.0;
    }
    return area_twice > 0.0 ? 1.0 : -1.0;
}

bool point_in_component(
    const ProjectedTargetComponent &component,
    Point2 point
) {
    for (std::size_t index = 0;
         index < component.vertices.size();
         ++index) {
        const Point2 a = component.vertices[index];
        const Point2 b = component.vertices[
            (index + 1) % component.vertices.size()
        ];
        const Point2 edge{b.x - a.x, b.y - a.y};
        const Point2 relative{point.x - a.x, point.y - a.y};
        const double side = component.orientation * cross2(edge, relative);
        const double scale = std::max({
            1.0,
            std::abs(edge.x),
            std::abs(edge.y),
            std::abs(relative.x),
            std::abs(relative.y),
        });
        if (side < -PROJECTION_EPSILON * scale * scale) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool make_target_projection(
    const Vec3 &center_direction,
    TargetProjection &out
) {
    TargetProjection projection{};
    if (!normalize(center_direction, projection.forward)) {
        out = {};
        return false;
    }

    Vec3 right = cross({0.0, 1.0, 0.0}, projection.forward);
    if (!normalize(right, projection.right)) {
        right = cross({1.0, 0.0, 0.0}, projection.forward);
        if (!normalize(right, projection.right)) {
            out = {};
            return false;
        }
    }
    if (!normalize(
            cross(projection.forward, projection.right),
            projection.up
        )) {
        out = {};
        return false;
    }

    out = projection;
    return true;
}

bool project_target_direction(
    const TargetProjection &projection,
    const Vec3 &direction,
    Point2 &out
) {
    if (!finite(direction)) {
        out = {};
        return false;
    }
    const double depth = dot(direction, projection.forward);
    if (!(depth > PROJECTION_EPSILON) || !std::isfinite(depth)) {
        out = {};
        return false;
    }

    const Point2 projected{
        dot(direction, projection.right) / depth,
        dot(direction, projection.up) / depth,
    };
    if (!finite(projected)) {
        out = {};
        return false;
    }
    out = projected;
    return true;
}

bool project_visible_target_region(
    const Vec3 &center_direction,
    const VisibleDirectionComponents &components,
    ProjectedTargetRegion &out
) {
    ProjectedTargetRegion region{};
    if (components.empty() ||
        !make_target_projection(center_direction, region.projection)) {
        out = {};
        return false;
    }

    region.components.reserve(components.size());
    for (const VisibleDirectionComponent &directions : components) {
        if (directions.size() < 3) {
            out = {};
            return false;
        }

        ProjectedTargetComponent component{};
        component.vertices.reserve(directions.size());
        for (const Vec3 &direction : directions) {
            Point2 projected{};
            if (!project_target_direction(
                    region.projection,
                    direction,
                    projected
                )) {
                out = {};
                return false;
            }
            component.vertices.push_back(projected);
        }
        component.orientation = polygon_orientation(component.vertices);
        if (component.orientation == 0.0) {
            out = {};
            return false;
        }
        region.components.push_back(std::move(component));
    }

    out = std::move(region);
    return true;
}

bool point_in_visible_region(
    const ProjectedTargetRegion &region,
    const Vec3 &direction
) {
    Point2 projected{};
    if (region.components.empty() ||
        !project_target_direction(region.projection, direction, projected)) {
        return false;
    }
    return std::any_of(
        region.components.begin(),
        region.components.end(),
        [projected](const ProjectedTargetComponent &component) {
            return point_in_component(component, projected);
        }
    );
}

}  // namespace minecraft_miner::aim
