#include "minecraft_miner/aim/path.hpp"

#include <algorithm>
#include <cmath>

namespace minecraft_miner::aim {

namespace {

double signed_angle_delta_degrees(double value, double origin) {
    double delta = value - origin;
    while (delta <= -180.0) {
        delta += 360.0;
    }
    while (delta > 180.0) {
        delta -= 360.0;
    }
    return delta;
}

double clamp_double(double value, double minimum, double maximum) {
    return std::max(minimum, std::min(maximum, value));
}

}  // namespace

AimPath generate_minimum_jerk_path(
    const Orientation &start,
    const TargetMetrics &target,
    const AimPathConfig &config
) {
    const double width_yaw = std::max(0.0, target.width_yaw);
    const double width_pitch = std::max(0.0, target.width_pitch);
    const double yaw_delta = signed_angle_delta_degrees(target.yaw, start.yaw);
    const double pitch_delta = target.pitch - start.pitch;
    const double amplitude = std::hypot(yaw_delta, pitch_delta);
    const double target_width =
        std::max(config.angular_step_deg, std::min(width_yaw, width_pitch));
    const double index_of_difficulty =
        std::log2(amplitude / target_width + 1.0);
    const double duration_ms = clamp_double(
        config.fitts_a_ms + config.fitts_b_ms * index_of_difficulty,
        config.min_duration_ms,
        config.max_duration_ms
    );

    return {
        AimSample{start.yaw, start.pitch, 0.0},
        AimSample{target.yaw, target.pitch, duration_ms},
    };
}

}  // namespace minecraft_miner::aim
