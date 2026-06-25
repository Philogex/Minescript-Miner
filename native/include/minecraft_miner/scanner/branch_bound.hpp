#pragma once

#include "minecraft_miner/scanner/scan_region.hpp"
#include "minecraft_miner/scanner/target_solver.hpp"

#include <cstdint>

namespace minecraft_miner {

BranchBoundResult solve_visible_target_face(
    const ScanRegionGeometry &geometry,
    std::uint32_t target_world_face_index,
    const Vec3 &eye,
    const Vec3 &look_direction,
    double reach = std::numeric_limits<double>::infinity(),
    double angle_limit = std::numeric_limits<double>::infinity(),
    BranchBoundOptions options = {}
);

BranchBoundResult solve_visible_target(
    const ScanRegionGeometry &geometry,
    const Vec3 &eye,
    const Vec3 &look_direction,
    double reach = std::numeric_limits<double>::infinity(),
    BranchBoundOptions options = {}
);

}  // namespace minecraft_miner
