#pragma once

#include "minescript_miner/tri2.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace minescript_miner {

inline constexpr std::size_t MAX_CLIP_VERTICES = 8;

struct Polygon2 {
    std::array<Point2, MAX_CLIP_VERTICES> points{};
    std::uint8_t count = 0;
};

struct LinearHalfPlane2 {
    double x = 0.0;
    double y = 0.0;
    double constant = 0.0;
};

enum class Orientation : std::int8_t {
    Clockwise = -1,
    Collinear = 0,
    CounterClockwise = 1,
};

// Approximate determinant for metric calculations. Do not use its sign for
// topological decisions; use orient2d() instead.
constexpr double orient2d_determinant(Point2 a, Point2 b, Point2 point) {
    return (b.x - a.x) * (point.y - a.y) -
           (b.y - a.y) * (point.x - a.x);
}

Orientation orient2d(Point2 a, Point2 b, Point2 point);

constexpr LinearHalfPlane2 edge_inside_half_plane(Point2 a, Point2 b) {
    return {
        a.y - b.y,
        b.x - a.x,
        (b.y - a.y) * a.x - (b.x - a.x) * a.y,
    };
}

constexpr LinearHalfPlane2 negate_half_plane(const LinearHalfPlane2 &plane) {
    return {-plane.x, -plane.y, -plane.constant};
}

constexpr double half_plane_value(const LinearHalfPlane2 &plane, Point2 point) {
    return plane.x * point.x + plane.y * point.y + plane.constant;
}

constexpr Polygon2 polygon_from_triangle(const Tri2 &triangle) {
    Polygon2 polygon{};
    polygon.points[0] = triangle.a;
    polygon.points[1] = triangle.b;
    polygon.points[2] = triangle.c;
    polygon.count = 3;
    return polygon;
}

constexpr Polygon2 polygon_from_quad(const std::array<Point2, 4> &points) {
    Polygon2 polygon{};
    polygon.points[0] = points[0];
    polygon.points[1] = points[1];
    polygon.points[2] = points[2];
    polygon.points[3] = points[3];
    polygon.count = 4;
    return polygon;
}

double signed_polygon_area2(const Polygon2 &polygon);
void ensure_counter_clockwise(Polygon2 &polygon);

bool clip_half_plane(
    const Polygon2 &polygon,
    const LinearHalfPlane2 &plane,
    Polygon2 &out
);

bool clip_inside_edge(
    const Polygon2 &polygon,
    Point2 edge_a,
    Point2 edge_b,
    Polygon2 &out
);

bool clip_outside_edge(
    const Polygon2 &polygon,
    Point2 edge_a,
    Point2 edge_b,
    Polygon2 &out
);

std::vector<Tri2> triangulate_convex_polygon(const Polygon2 &polygon);

std::vector<Tri2> subtract_convex_polygon(
    const Tri2 &target,
    Polygon2 occluder
);

}  // namespace minescript_miner
