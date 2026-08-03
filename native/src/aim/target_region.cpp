#include "minecraft_miner/aim/target_region.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace minecraft_miner::aim {

namespace {

constexpr double PROJECTION_EPSILON = 1.0e-12;
constexpr double SAFE_INTERIOR_BLEND = 1.0e-3;

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

double polygon_area_twice(const std::vector<Point2> &vertices) {
    double area_twice = 0.0;
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        area_twice += cross2(
            vertices[index],
            vertices[(index + 1) % vertices.size()]
        );
    }
    return area_twice;
}

bool polygon_centroid(
    const std::vector<Point2> &vertices,
    Point2 &out
) {
    const double area_twice = polygon_area_twice(vertices);
    if (!std::isfinite(area_twice) ||
        std::abs(area_twice) <= PROJECTION_EPSILON) {
        out = {};
        return false;
    }

    Point2 weighted{};
    for (std::size_t index = 0; index < vertices.size(); ++index) {
        const Point2 current = vertices[index];
        const Point2 next = vertices[(index + 1) % vertices.size()];
        const double weight = cross2(current, next);
        weighted.x += (current.x + next.x) * weight;
        weighted.y += (current.y + next.y) * weight;
    }
    const double denominator = 3.0 * area_twice;
    out = {
        weighted.x / denominator,
        weighted.y / denominator,
    };
    return finite(out);
}

bool point_in_polygon(
    const std::vector<Point2> &vertices,
    double orientation,
    Point2 point
) {
    for (std::size_t index = 0;
         index < vertices.size();
         ++index) {
        const Point2 a = vertices[index];
        const Point2 b = vertices[(index + 1) % vertices.size()];
        const Point2 edge{b.x - a.x, b.y - a.y};
        const Point2 relative{point.x - a.x, point.y - a.y};
        const double side = orientation * cross2(edge, relative);
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

double inset_side(
    Point2 a,
    Point2 b,
    Point2 point,
    double orientation,
    double margin
) {
    const Point2 edge{b.x - a.x, b.y - a.y};
    const Point2 relative{point.x - a.x, point.y - a.y};
    return orientation * cross2(edge, relative) -
        margin * std::hypot(edge.x, edge.y);
}

std::vector<Point2> inset_component(
    const ProjectedTargetComponent &component,
    double margin
) {
    std::vector<Point2> polygon = component.vertices;
    for (std::size_t edge_index = 0;
         edge_index < component.vertices.size() && !polygon.empty();
         ++edge_index) {
        const Point2 a = component.vertices[edge_index];
        const Point2 b = component.vertices[
            (edge_index + 1) % component.vertices.size()
        ];
        std::vector<Point2> clipped;
        clipped.reserve(polygon.size() + 1);

        Point2 previous = polygon.back();
        double previous_side = inset_side(
            a,
            b,
            previous,
            component.orientation,
            margin
        );
        bool previous_inside = previous_side >= -PROJECTION_EPSILON;
        for (const Point2 current : polygon) {
            const double current_side = inset_side(
                a,
                b,
                current,
                component.orientation,
                margin
            );
            const bool current_inside =
                current_side >= -PROJECTION_EPSILON;
            if (current_inside != previous_inside) {
                const double denominator = previous_side - current_side;
                if (std::abs(denominator) > PROJECTION_EPSILON) {
                    const double t = previous_side / denominator;
                    clipped.push_back({
                        previous.x + (current.x - previous.x) * t,
                        previous.y + (current.y - previous.y) * t,
                    });
                }
            }
            if (current_inside) {
                clipped.push_back(current);
            }
            previous = current;
            previous_side = current_side;
            previous_inside = current_inside;
        }
        polygon = std::move(clipped);
    }
    return polygon;
}

bool valid_polygon(const std::vector<Point2> &vertices) {
    return vertices.size() >= 3 && polygon_orientation(vertices) != 0.0;
}

std::vector<Point2> safe_component_vertices(
    const ProjectedTargetComponent &component,
    double requested_margin
) {
    double margin = std::max(0.0, requested_margin);
    for (int attempt = 0; attempt < 12; ++attempt) {
        std::vector<Point2> vertices = inset_component(component, margin);
        if (valid_polygon(vertices)) {
            return vertices;
        }
        margin *= 0.5;
    }
    return component.vertices;
}

Point2 closest_point_on_segment(Point2 point, Point2 a, Point2 b) {
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double length_squared = dx * dx + dy * dy;
    if (!(length_squared > PROJECTION_EPSILON * PROJECTION_EPSILON)) {
        return a;
    }
    const double t = std::clamp(
        ((point.x - a.x) * dx + (point.y - a.y) * dy) /
            length_squared,
        0.0,
        1.0
    );
    return {a.x + dx * t, a.y + dy * t};
}

double distance_squared(Point2 lhs, Point2 rhs) {
    const double dx = lhs.x - rhs.x;
    const double dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
}

bool direction_from_projected_point(
    const TargetProjection &projection,
    Point2 point,
    Vec3 &out
) {
    return normalize({
        projection.forward.x + projection.right.x * point.x +
            projection.up.x * point.y,
        projection.forward.y + projection.right.y * point.x +
            projection.up.y * point.y,
        projection.forward.z + projection.right.z * point.x +
            projection.up.z * point.y,
    }, out);
}

struct SafeComponentCandidate {
    ProjectedTargetComponent component{};
    Point2 centroid{};
    double area = 0.0;
    double center_distance = 0.0;
};

bool better_anchor_candidate(
    const SafeComponentCandidate &candidate,
    const SafeComponentCandidate &best
) {
    if (candidate.center_distance != best.center_distance) {
        return candidate.center_distance < best.center_distance;
    }
    return candidate.area > best.area;
}

bool target_region_centroid(
    const ProjectedTargetRegion &region,
    Point2 &out
) {
    Point2 weighted{};
    double total_area = 0.0;
    for (const ProjectedTargetComponent &component : region.components) {
        Point2 centroid{};
        if (!polygon_centroid(component.vertices, centroid)) {
            continue;
        }
        const double area = std::abs(
            polygon_area_twice(component.vertices)
        ) * 0.5;
        weighted.x += centroid.x * area;
        weighted.y += centroid.y * area;
        total_area += area;
    }
    if (!(total_area > PROJECTION_EPSILON) || !std::isfinite(total_area)) {
        out = {};
        return false;
    }
    out = {weighted.x / total_area, weighted.y / total_area};
    return finite(out);
}

bool build_safe_components(
    const ProjectedTargetRegion &region,
    Point2 target_center,
    double margin,
    std::vector<SafeComponentCandidate> &out
) {
    out.clear();
    out.reserve(region.components.size());
    for (std::size_t index = 0; index < region.components.size(); ++index) {
        const ProjectedTargetComponent &source = region.components[index];
        std::vector<Point2> vertices = inset_component(source, margin);
        const double orientation = polygon_orientation(vertices);
        Point2 centroid{};
        if (vertices.size() < 3 || orientation == 0.0 ||
            !polygon_centroid(vertices, centroid) ||
            !point_in_polygon(vertices, orientation, centroid)) {
            continue;
        }

        SafeComponentCandidate candidate{};
        candidate.component.vertices = std::move(vertices);
        candidate.component.orientation = orientation;
        candidate.centroid = centroid;
        candidate.area = std::abs(
            polygon_area_twice(candidate.component.vertices)
        ) * 0.5;
        candidate.center_distance = distance_squared(
            target_center,
            centroid
        );
        out.push_back(std::move(candidate));
    }
    return !out.empty();
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

bool make_safe_target_region(
    const ProjectedTargetRegion &region,
    double requested_margin,
    SafeTargetRegion &out
) {
    out = {};
    if (region.components.empty() || !std::isfinite(requested_margin) ||
        requested_margin < 0.0) {
        return false;
    }

    Point2 target_center{};
    if (!target_region_centroid(region, target_center)) {
        return false;
    }

    std::vector<SafeComponentCandidate> candidates;
    double applied_margin = requested_margin;
    bool built = false;
    for (int attempt = 0; attempt < 12; ++attempt) {
        if (build_safe_components(
                region,
                target_center,
                applied_margin,
                candidates
            )) {
            built = true;
            break;
        }
        applied_margin *= 0.5;
    }
    if (!built && !build_safe_components(
            region,
            target_center,
            0.0,
            candidates
        )) {
        return false;
    }
    if (!built) {
        applied_margin = 0.0;
    }

    std::size_t best_index = 0;
    for (std::size_t index = 1; index < candidates.size(); ++index) {
        if (better_anchor_candidate(candidates[index], candidates[best_index])) {
            best_index = index;
        }
    }

    SafeTargetRegion result{};
    result.region.projection = region.projection;
    result.region.components.reserve(candidates.size());
    for (SafeComponentCandidate &candidate : candidates) {
        result.region.components.push_back(std::move(candidate.component));
    }
    result.anchor_component_index = best_index;
    result.applied_margin = applied_margin;
    if (!direction_from_projected_point(
            result.region.projection,
            candidates[best_index].centroid,
            result.anchor_direction
        )) {
        return false;
    }

    out = std::move(result);
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
            return point_in_polygon(
                component.vertices,
                component.orientation,
                projected
            );
        }
    );
}

bool point_in_visible_region_with_margin(
    const ProjectedTargetRegion &region,
    const Vec3 &direction,
    double margin
) {
    Point2 projected{};
    if (region.components.empty() || !std::isfinite(margin) || margin < 0.0 ||
        !project_target_direction(region.projection, direction, projected)) {
        return false;
    }
    return std::any_of(
        region.components.begin(),
        region.components.end(),
        [projected, margin](const ProjectedTargetComponent &component) {
            const std::vector<Point2> safe_vertices =
                safe_component_vertices(component, margin);
            return point_in_polygon(
                safe_vertices,
                polygon_orientation(safe_vertices),
                projected
            );
        }
    );
}

bool closest_safe_direction_in_visible_region(
    const ProjectedTargetRegion &region,
    const Vec3 &direction,
    double margin,
    Vec3 &out
) {
    if (region.components.empty() || !std::isfinite(margin) || margin < 0.0) {
        out = {};
        return false;
    }

    Point2 query{};
    project_target_direction(region.projection, direction, query);
    Point2 best{};
    double best_distance = std::numeric_limits<double>::infinity();
    for (const ProjectedTargetComponent &component : region.components) {
        const std::vector<Point2> vertices =
            safe_component_vertices(component, margin);
        const double orientation = polygon_orientation(vertices);
        if (orientation == 0.0) {
            continue;
        }

        if (point_in_polygon(vertices, orientation, query)) {
            best = query;
            best_distance = 0.0;
            break;
        }
        Point2 centroid{};
        for (const Point2 vertex : vertices) {
            centroid.x += vertex.x;
            centroid.y += vertex.y;
        }
        centroid.x /= static_cast<double>(vertices.size());
        centroid.y /= static_cast<double>(vertices.size());
        for (std::size_t index = 0; index < vertices.size(); ++index) {
            Point2 candidate = closest_point_on_segment(
                query,
                vertices[index],
                vertices[(index + 1) % vertices.size()]
            );
            candidate.x += (centroid.x - candidate.x) * SAFE_INTERIOR_BLEND;
            candidate.y += (centroid.y - candidate.y) * SAFE_INTERIOR_BLEND;
            const double candidate_distance =
                distance_squared(query, candidate);
            if (candidate_distance < best_distance) {
                best = candidate;
                best_distance = candidate_distance;
            }
        }
    }
    if (!std::isfinite(best_distance) ||
        !direction_from_projected_point(region.projection, best, out)) {
        out = {};
        return false;
    }
    return true;
}

}  // namespace minecraft_miner::aim
