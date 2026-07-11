#pragma once

#include <cstdint>
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

struct SigmaDriftConfig {
    double fitts_a = 50.0;
    double fitts_b = 150.0;
    double target_width = 20.0;
    double undershoot_min = 0.92;
    double undershoot_max = 0.97;
    double peak_time_ratio = 0.35;
    double primary_sigma_min = 0.18;
    double primary_sigma_max = 0.28;
    double overshoot_prob = 0.15;
    double overshoot_min = 1.02;
    double overshoot_max = 1.08;
    double correction_sigma_min = 0.12;
    double correction_sigma_max = 0.20;
    double second_correction_prob = 0.25;
    double curvature_scale = 0.025;
    double ou_theta = 3.5;
    double ou_sigma = 1.2;
    double tremor_freq_min = 8.0;
    double tremor_freq_max = 12.0;
    double tremor_amp_min = 0.15;
    double tremor_amp_max = 0.55;
    double sdn_k = 0.04;
    double sample_dt_mean = 7.8;
    double gamma_shape = 3.5;
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

AimPath generate_sigmadrift_path(
    const Orientation &start,
    const TargetMetrics &target,
    double angular_step_deg,
    const SigmaDriftConfig &config,
    std::uint64_t seed
);

}  // namespace minecraft_miner::aim
