#include "minecraft_miner/aim/geometry_feedback_sigmadrift.hpp"

#include "minecraft_miner/aim/angle.hpp"

namespace minecraft_miner::aim {

AimPath generate_geometry_feedback_sigmadrift_path(
    const Orientation &start,
    const TargetMetrics &target,
    const VisibleDirectionComponents &visible_components,
    double angular_step_deg,
    const SigmaDriftConfig &config,
    std::uint64_t seed
) {
    (void) angular_step_deg;
    (void) config;
    (void) seed;

    const Vec3 target_direction = look_direction_from_yaw_pitch(
        target.yaw,
        target.pitch
    );
    ProjectedTargetRegion projected_region{};
    if (!project_visible_target_region(
            target_direction,
            visible_components,
            projected_region
        ) ||
        !point_in_visible_region(projected_region, target_direction)) {
        return {};
    }

    // Placeholder path until the stateful feedback model is introduced.
    return {
        AimSample{start.yaw, start.pitch, 0.0},
        AimSample{target.yaw, target.pitch, 100.0},
    };
}

}  // namespace minecraft_miner::aim
