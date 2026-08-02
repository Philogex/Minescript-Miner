#pragma once

#include "minecraft_miner/aim/sigmadrift.hpp"
#include "minecraft_miner/aim/target_region.hpp"

#include <cstdint>

namespace minecraft_miner::aim {

AimPath generate_geometry_feedback_sigmadrift_path(
    const Orientation &start,
    const TargetMetrics &target,
    const VisibleDirectionComponents &visible_components,
    double angular_step_deg,
    const SigmaDriftConfig &config,
    std::uint64_t seed
);

}  // namespace minecraft_miner::aim
