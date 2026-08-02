#pragma once

#include "minecraft_miner/aim/sigmadrift.hpp"
#include "minecraft_miner/aim/target_region.hpp"

#include <cstdint>

namespace minecraft_miner::aim {

struct GeometryFeedbackSigmaDriftConfig {
    double feedback_latency_ms = 100.0;
    double safe_margin_steps = 1.0;
    int max_corrections = 3;
};

AimPath generate_geometry_feedback_sigmadrift_path(
    const Orientation &start,
    const TargetMetrics &target,
    const VisibleDirectionComponents &visible_components,
    double angular_step_deg,
    const SigmaDriftConfig &motion_config,
    const GeometryFeedbackSigmaDriftConfig &feedback_config,
    std::uint64_t seed
);

}  // namespace minecraft_miner::aim
