#pragma once

#include "minescript_miner/branch_bound.hpp"

#include <cstdint>

namespace minescript_miner {

// Isolated exact-topology milestone for one target face. Reach clipping and
// the outer target loop remain responsibilities of the later parallel solver.
BranchBoundResult solve_visible_target_face_exact(
    const ScanRegionGeometry &geometry,
    std::uint32_t target_world_face_index,
    const Vec3 &eye,
    const Vec3 &look_direction,
    BranchBoundOptions options = {}
);

}  // namespace minescript_miner
