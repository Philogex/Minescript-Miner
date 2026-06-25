#pragma once

#include <array>

namespace minescript_miner {

struct Point2 {
    double x = 0.0;
    double y = 0.0;
};

struct Tri2 {
    Point2 a{};
    Point2 b{};
    Point2 c{};
};

constexpr Point2 point2(double x, double y) {
    return {x, y};
}

constexpr Tri2 tri2(Point2 a, Point2 b, Point2 c) {
    return {a, b, c};
}

constexpr double cross(Point2 origin, Point2 lhs, Point2 rhs) {
    return (lhs.x - origin.x) * (rhs.y - origin.y) -
           (lhs.y - origin.y) * (rhs.x - origin.x);
}

constexpr double signed_area2(const Tri2 &tri) {
    return cross(tri.a, tri.b, tri.c);
}

constexpr std::array<Tri2, 2> rect_to_tris(double min_x, double min_y, double max_x, double max_y) {
    const Point2 p0 = point2(min_x, min_y);
    const Point2 p1 = point2(max_x, min_y);
    const Point2 p2 = point2(max_x, max_y);
    const Point2 p3 = point2(min_x, max_y);
    return {tri2(p0, p1, p2), tri2(p0, p2, p3)};
}

}  // namespace minescript_miner
