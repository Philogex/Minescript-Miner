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

}  // namespace minecraft_miner::aim
