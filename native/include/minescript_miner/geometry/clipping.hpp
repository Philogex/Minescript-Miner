#pragma once

#include "minescript_miner/geometry/tri2.hpp"

#include <cstdint>

namespace minescript_miner {

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

}  // namespace minescript_miner
