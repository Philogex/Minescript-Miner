#pragma once

#include "minecraft_miner/scanner/scan_region.hpp"
#include "minecraft_miner/scanner/target_solver.hpp"

#include <cstdint>
#include <vector>

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

struct VisibleRegionComponent {
    std::uint32_t target_world_face_index = 0;
    std::vector<Vec3> boundary_directions{};
};

struct VisibleTargetRegionResult {
    BranchBoundResult target{};
    BlockPos target_block{};
    std::vector<VisibleRegionComponent> components{};
    BranchBoundStats region_stats{};
};

// Finds the same best target as solve_visible_target(), then enumerates the
// complete visible region of every candidate face belonging to that block.
// Components remain separate because their union can be non-convex and can
// span faces with different projective bases.
VisibleTargetRegionResult solve_full_visible_target(
    const ScanRegionGeometry &geometry,
    const Vec3 &eye,
    const Vec3 &look_direction,
    double reach = std::numeric_limits<double>::infinity(),
    BranchBoundOptions options = {}
);

}  // namespace minecraft_miner
