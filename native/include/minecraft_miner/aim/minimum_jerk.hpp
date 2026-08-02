#pragma once

#include "minecraft_miner/aim/path.hpp"

namespace minecraft_miner::aim {

struct AimPathConfig {
    double angular_step_deg = 0.0;
    double fitts_a_ms = 0.0;
    double fitts_b_ms = 0.0;
    double min_duration_ms = 0.0;
    double max_duration_ms = 0.0;
    int sample_hz = 0;
};

AimPath generate_minimum_jerk_path(
    const Orientation &start,
    const TargetMetrics &target,
    const AimPathConfig &config
);

}  // namespace minecraft_miner::aim
