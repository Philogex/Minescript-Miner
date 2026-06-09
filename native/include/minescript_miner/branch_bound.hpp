#pragma once

#include "minescript_miner/scan_region.hpp"
#include "minescript_miner/target_solver.hpp"

#include <limits>

namespace minescript_miner {

BranchBoundResult solve_visible_target(
    const ScanRegionGeometry &geometry,
    const Vec3 &eye,
    const Vec3 &look_direction,
    double reach = std::numeric_limits<double>::infinity(),
    BranchBoundOptions options = {}
);

}  // namespace minescript_miner
