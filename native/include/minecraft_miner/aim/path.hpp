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
    double effective_width = 0.0;
};

struct AimSample {
    double yaw = 0.0;
    double pitch = 0.0;
    double t_ms = 0.0;
};

using AimPath = std::vector<AimSample>;

inline double shortest_yaw_delta_degrees(double value, double origin) {
    double delta = value - origin;
    while (delta <= -180.0) {
        delta += 360.0;
    }
    while (delta > 180.0) {
        delta -= 360.0;
    }
    return delta;
}

inline double continuous_yaw_near(double value, double reference) {
    return reference + shortest_yaw_delta_degrees(value, reference);
}

}  // namespace minecraft_miner::aim
