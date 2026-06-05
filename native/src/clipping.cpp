#include "minescript_miner/clipping.hpp"

#include <algorithm>

namespace minescript_miner {

static_assert(orient2d({0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}) > 0.0);
static_assert(orient2d({0.0, 0.0}, {1.0, 0.0}, {0.0, -1.0}) < 0.0);
static_assert(orient2d({0.0, 0.0}, {1.0, 0.0}, {0.5, 0.0}) == 0.0);
static_assert(half_plane_value(
                  edge_inside_half_plane({0.0, 0.0}, {1.0, 0.0}),
                  {0.0, 1.0}
              ) > 0.0);

namespace {

bool append_point(Polygon2 &polygon, Point2 point) {
    if (polygon.count >= polygon.points.size()) {
        return false;
    }
    polygon.points[polygon.count] = point;
    ++polygon.count;
    return true;
}

Point2 half_plane_intersection(
    Point2 from,
    Point2 to,
    double from_value,
    double to_value
) {
    const double t = from_value / (from_value - to_value);
    return {
        from.x + t * (to.x - from.x),
        from.y + t * (to.y - from.y),
    };
}

}  // namespace

double signed_polygon_area2(const Polygon2 &polygon) {
    double area = 0.0;
    for (std::uint8_t i = 0; i < polygon.count; ++i) {
        const Point2 current = polygon.points[i];
        const Point2 next = polygon.points[(i + 1) % polygon.count];
        area += current.x * next.y - current.y * next.x;
    }
    return area;
}

void ensure_counter_clockwise(Polygon2 &polygon) {
    if (signed_polygon_area2(polygon) >= 0.0) {
        return;
    }
    std::reverse(polygon.points.begin(), polygon.points.begin() + polygon.count);
}

bool clip_half_plane(
    const Polygon2 &polygon,
    const LinearHalfPlane2 &plane,
    Polygon2 &out
) {
    const Polygon2 input = polygon;
    out = {};
    if (input.count == 0) {
        return true;
    }

    Point2 previous = input.points[input.count - 1];
    double previous_value = half_plane_value(plane, previous);
    bool previous_inside = previous_value >= 0.0;

    for (std::uint8_t i = 0; i < input.count; ++i) {
        const Point2 current = input.points[i];
        const double current_value = half_plane_value(plane, current);
        const bool current_inside = current_value >= 0.0;

        if (previous_inside != current_inside) {
            if (!append_point(
                    out,
                    half_plane_intersection(previous, current, previous_value, current_value)
                )) {
                return false;
            }
        }
        if (current_inside && !append_point(out, current)) {
            return false;
        }

        previous = current;
        previous_value = current_value;
        previous_inside = current_inside;
    }
    return true;
}

bool clip_inside_edge(
    const Polygon2 &polygon,
    Point2 edge_a,
    Point2 edge_b,
    Polygon2 &out
) {
    return clip_half_plane(polygon, edge_inside_half_plane(edge_a, edge_b), out);
}

bool clip_outside_edge(
    const Polygon2 &polygon,
    Point2 edge_a,
    Point2 edge_b,
    Polygon2 &out
) {
    return clip_half_plane(
        polygon,
        negate_half_plane(edge_inside_half_plane(edge_a, edge_b)),
        out
    );
}

std::vector<Tri2> triangulate_convex_polygon(const Polygon2 &polygon) {
    std::vector<Tri2> triangles;
    if (polygon.count < 3) {
        return triangles;
    }

    triangles.reserve(polygon.count - 2);
    for (std::uint8_t i = 1; i + 1 < polygon.count; ++i) {
        const Tri2 triangle{
            polygon.points[0],
            polygon.points[i],
            polygon.points[i + 1],
        };
        if (signed_area2(triangle) != 0.0) {
            triangles.push_back(triangle);
        }
    }
    return triangles;
}

std::vector<Tri2> subtract_convex_polygon(
    const Tri2 &target,
    Polygon2 occluder
) {
    Polygon2 remaining = polygon_from_triangle(target);
    ensure_counter_clockwise(remaining);
    if (occluder.count < 3) {
        return triangulate_convex_polygon(remaining);
    }
    ensure_counter_clockwise(occluder);

    std::vector<Tri2> visible_triangles;
    for (std::uint8_t edge_index = 0; edge_index < occluder.count; ++edge_index) {
        const Point2 edge_a = occluder.points[edge_index];
        const Point2 edge_b = occluder.points[(edge_index + 1) % occluder.count];

        Polygon2 outside{};
        if (!clip_outside_edge(remaining, edge_a, edge_b, outside)) {
            return {};
        }
        std::vector<Tri2> outside_triangles = triangulate_convex_polygon(outside);
        visible_triangles.insert(
            visible_triangles.end(),
            outside_triangles.begin(),
            outside_triangles.end()
        );

        Polygon2 inside{};
        if (!clip_inside_edge(remaining, edge_a, edge_b, inside)) {
            return {};
        }
        remaining = inside;
        if (remaining.count == 0) {
            break;
        }
    }

    return visible_triangles;
}

}  // namespace minescript_miner
