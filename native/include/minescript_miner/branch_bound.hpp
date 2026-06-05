#pragma once

#include "minescript_miner/scan_region.hpp"
#include "minescript_miner/tri2.hpp"
#include "minescript_miner/vec.hpp"

#include <cstdint>
#include <limits>

namespace minescript_miner {

struct TriangleAngleResult {
    Point2 point{};
    double angle = std::numeric_limits<double>::infinity();
};

struct BranchBoundOptions {
    std::uint16_t occluder_probe_limit = 4;
};

struct BranchBoundStats {
    std::uint32_t target_faces_considered = 0;
    std::uint32_t target_faces_pruned = 0;
    std::uint32_t occluders_prepared = 0;
    std::uint32_t effective_occluders = 0;
    std::uint64_t branches_visited = 0;
    std::uint64_t clips_performed = 0;
};

struct BranchBoundResult {
    bool found = false;
    std::uint32_t target_world_face_index = 0;
    Point2 projected_point{};
    Vec3 direction{};
    double angle = std::numeric_limits<double>::infinity();
    BranchBoundStats stats{};
};

TriangleAngleResult minimum_angle_to_triangle(
    const Tri2 &triangle,
    const Vec3 &look_direction_in_view
);

BranchBoundResult solve_visible_target(
    const ScanRegionGeometry &geometry,
    const Vec3 &eye,
    const Vec3 &look_direction,
    BranchBoundOptions options = {}
);

}  // namespace minescript_miner
