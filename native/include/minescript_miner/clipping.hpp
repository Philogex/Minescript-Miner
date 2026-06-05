#pragma once

#include "minescript_miner/tri2.hpp"

namespace minescript_miner {

constexpr double orient2d(Point2 a, Point2 b, Point2 point) {
    return (b.x - a.x) * (point.y - a.y) -
           (b.y - a.y) * (point.x - a.x);
}

}  // namespace minescript_miner
