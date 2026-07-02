#pragma once

#include <vector>

namespace minecraft_miner::aim {

struct Orientation {
    double yaw = 0.0;
    double pitch = 0.0;
};

struct TargetMetrics {
    double yaw = 0.0;
    double pitch = 0.0;
    double width_yaw = 0.0;
    double width_pitch = 0.0;
    double distance = 0.0;
};

struct AimPathConfig {
    double angular_step_deg = 0.0;
    double fitts_a_ms = 0.0;
    double fitts_b_ms = 0.0;
    double min_duration_ms = 0.0;
    double max_duration_ms = 0.0;
    int sample_hz = 0;
};

struct AimSample {
    double yaw = 0.0;
    double pitch = 0.0;
    double t_ms = 0.0;
};

using AimPath = std::vector<AimSample>;

AimPath generate_minimum_jerk_path(
    const Orientation &start,
    const TargetMetrics &target,
    const AimPathConfig &config
);

}  // namespace minecraft_miner::aim
