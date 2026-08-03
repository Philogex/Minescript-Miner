#pragma once

#include "minecraft_miner/aim/sigmadrift.hpp"
#include "minecraft_miner/aim/target_region.hpp"

#include <cstddef>
#include <cstdint>

namespace minecraft_miner::aim {

struct GeometryFeedbackSigmaDriftConfig {
    double feedback_latency_mean_ms = 100.0;
    double feedback_latency_stddev_ms = 15.0;
    double feedback_latency_min_ms = 60.0;
    double feedback_latency_max_ms = 160.0;
    double undershoot_width_min = 0.05;
    double undershoot_width_max = 0.25;
    double overshoot_width_min = 0.05;
    double overshoot_width_max = 0.25;
    double safe_margin_steps = 1.0;
    int max_corrections = 3;
};

struct GeometryFeedbackSigmaDriftDiagnostics {
    double motor_target_yaw = 0.0;
    double motor_target_pitch = 0.0;
    double applied_margin_steps = 0.0;
    std::size_t anchor_component_index = 0;
    double directional_width_steps = 0.0;
    double s_enter_steps = 0.0;
    double s_anchor_steps = 0.0;
    double s_exit_steps = 0.0;
    double primary_endpoint_steps = 0.0;
    double first_feedback_observation_ms = -1.0;
    double first_feedback_latency_ms = -1.0;
    int feedback_check_count = 0;
    int correction_count = 0;
    double first_visible_entry_ms = -1.0;
    double first_safe_entry_ms = -1.0;
    int visible_entry_count = 0;
    int visible_exit_count = 0;
    int safe_entry_count = 0;
    int safe_exit_count = 0;
    bool final_visible = false;
    bool final_safe = false;
};

AimPath generate_geometry_feedback_sigmadrift_path(
    const Orientation &start,
    const TargetMetrics &target,
    const VisibleDirectionComponents &visible_components,
    double angular_step_deg,
    const SigmaDriftConfig &motion_config,
    const GeometryFeedbackSigmaDriftConfig &feedback_config,
    std::uint64_t seed,
    GeometryFeedbackSigmaDriftDiagnostics *diagnostics = nullptr
);

}  // namespace minecraft_miner::aim
