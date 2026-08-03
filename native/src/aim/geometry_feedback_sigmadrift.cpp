#include "minecraft_miner/aim/geometry_feedback_sigmadrift.hpp"

#include "minecraft_miner/aim/angle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace minecraft_miner::aim {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double SQRT_2 = 1.414213562373095048801688724209698079;
constexpr std::size_t MAX_PATH_SAMPLES = 100000;

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

double wrap_yaw_degrees(double yaw) {
    while (yaw <= -180.0) {
        yaw += 360.0;
    }
    while (yaw > 180.0) {
        yaw -= 360.0;
    }
    return yaw;
}

double normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / SQRT_2));
}

double lognormal_cdf(double t, double t0, double mu, double sigma) {
    if (t <= t0) {
        return 0.0;
    }
    return normal_cdf((std::log(t - t0) - mu) / sigma);
}

double lognormal_pdf(double t, double t0, double mu, double sigma) {
    if (t <= t0) {
        return 0.0;
    }
    const double dt = t - t0;
    const double z = (std::log(dt) - mu) / sigma;
    return std::exp(-0.5 * z * z) /
        (sigma * std::sqrt(2.0 * PI) * dt);
}

double curvature_profile(double progress) {
    if (progress <= 0.0 || progress >= 1.0) {
        return 0.0;
    }
    const double value = progress * progress * (1.0 - progress) *
        (1.0 - progress) * (1.0 - progress);
    constexpr double normalization = 0.4 * 0.4 * 0.6 * 0.6 * 0.6;
    return value / normalization;
}

double direction_factor(double angle) {
    const double sine = std::abs(std::sin(angle));
    const double cosine = std::abs(std::cos(angle));
    return 0.5 + 0.8 * sine - 0.15 * cosine;
}

double quantized_target_width(
    const TargetMetrics &target,
    double angular_step_deg,
    const SigmaDriftConfig &config
) {
    if (target.effective_width > 0.0) {
        return std::max(1.0, target.effective_width / angular_step_deg);
    }
    const double width_yaw = std::max(0.0, target.width_yaw);
    const double width_pitch = std::max(0.0, target.width_pitch);
    if (width_yaw > 0.0 && width_pitch > 0.0) {
        return std::max(
            1.0,
            std::min(width_yaw, width_pitch) / angular_step_deg
        );
    }
    return std::max(1.0, config.target_width);
}

struct Submovement {
    double x = 0.0;
    double y = 0.0;
    double t0 = 0.0;
    double mu = 0.0;
    double sigma = 0.0;
    double peak_time = 0.0;
    double tail_time = 0.0;
};

struct EvaluatedMotion {
    double x = 0.0;
    double y = 0.0;
    double speed = 0.0;
};

struct DirectionalTargetInterval {
    double enter = 0.0;
    double anchor = 0.0;
    double exit = 0.0;

    double width() const {
        return exit - enter;
    }
};

EvaluatedMotion evaluate_submovements(
    const std::vector<Submovement> &submovements,
    double t
) {
    EvaluatedMotion result{};
    for (const Submovement &movement : submovements) {
        const double progress = lognormal_cdf(
            t,
            movement.t0,
            movement.mu,
            movement.sigma
        );
        result.x += movement.x * progress;
        result.y += movement.y * progress;
        result.speed += std::hypot(movement.x, movement.y) * lognormal_pdf(
            t,
            movement.t0,
            movement.mu,
            movement.sigma
        );
    }
    return result;
}

Orientation orientation_from_position(
    const Orientation &start,
    double x,
    double y,
    double angular_step_deg
) {
    return {
        wrap_yaw_degrees(start.yaw + x * angular_step_deg),
        clamp_double(
            start.pitch + y * angular_step_deg,
            -90.0,
            90.0
        ),
    };
}

bool directional_target_interval(
    const Orientation &start,
    const ProjectedTargetRegion &visible_region,
    const SafeTargetRegion &safe_target,
    double target_x,
    double target_y,
    double angular_step_deg,
    DirectionalTargetInterval &out
) {
    out = {};
    const double anchor_distance = std::hypot(target_x, target_y);
    if (!(anchor_distance > 0.0) || !std::isfinite(anchor_distance)) {
        return false;
    }
    const double tangent_x = target_x / anchor_distance;
    const double tangent_y = target_y / anchor_distance;

    Point2 projected_anchor{};
    if (!project_target_direction(
            safe_target.region.projection,
            safe_target.anchor_direction,
            projected_anchor
        )) {
        return false;
    }

    // A one-input-step backward probe gives the local projected approach axis
    // without requiring the potentially distant start direction to remain in
    // the target-centered gnomonic hemisphere.
    const Orientation probe_orientation = orientation_from_position(
        start,
        target_x - tangent_x,
        target_y - tangent_y,
        angular_step_deg
    );
    const Vec3 probe_direction = look_direction_from_yaw_pitch(
        probe_orientation.yaw,
        probe_orientation.pitch
    );
    Point2 projected_probe{};
    if (!project_target_direction(
            safe_target.region.projection,
            probe_direction,
            projected_probe
        )) {
        return false;
    }

    Point2 projected_axis{
        projected_anchor.x - projected_probe.x,
        projected_anchor.y - projected_probe.y,
    };
    const double projected_axis_length = std::hypot(
        projected_axis.x,
        projected_axis.y
    );
    if (!(projected_axis_length > 1.0e-12) ||
        !std::isfinite(projected_axis_length)) {
        return false;
    }
    projected_axis.x /= projected_axis_length;
    projected_axis.y /= projected_axis_length;

    auto scalar_at_projected_distance = [&](double projected_distance) {
        const Point2 point{
            projected_anchor.x + projected_axis.x * projected_distance,
            projected_anchor.y + projected_axis.y * projected_distance,
        };
        Vec3 direction{};
        if (!direction_from_projected_target_point(
                safe_target.region.projection,
                point,
                direction
            )) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const YawPitch orientation = yaw_pitch_from_direction(direction);
        const double x = signed_angle_delta_degrees(
            orientation.yaw,
            start.yaw
        ) / angular_step_deg;
        const double y = (orientation.pitch - start.pitch) /
            angular_step_deg;
        return x * tangent_x + y * tangent_y;
    };

    std::vector<DirectionalTargetInterval> intervals;
    intervals.reserve(visible_region.components.size());
    for (std::size_t component_index = 0;
         component_index < visible_region.components.size();
         ++component_index) {
        ProjectedLineInterval projected_interval{};
        if (!projected_component_line_interval(
                visible_region,
                component_index,
                projected_anchor,
                projected_axis,
                projected_interval
            )) {
            continue;
        }
        double enter = scalar_at_projected_distance(projected_interval.enter);
        double exit = scalar_at_projected_distance(projected_interval.exit);
        if (!std::isfinite(enter) || !std::isfinite(exit)) {
            continue;
        }
        if (enter > exit) {
            std::swap(enter, exit);
        }
        if (exit - enter > 1.0e-9) {
            intervals.push_back({enter, anchor_distance, exit});
        }
    }
    if (intervals.empty()) {
        return false;
    }
    std::sort(
        intervals.begin(),
        intervals.end(),
        [](const DirectionalTargetInterval &lhs,
           const DirectionalTargetInterval &rhs) {
            return lhs.enter < rhs.enter;
        }
    );

    double merged_enter = intervals.front().enter;
    double merged_exit = intervals.front().exit;
    for (std::size_t index = 1; index <= intervals.size(); ++index) {
        if (index < intervals.size() &&
            intervals[index].enter <= merged_exit + 1.0e-9) {
            merged_exit = std::max(merged_exit, intervals[index].exit);
            continue;
        }
        if (merged_enter <= anchor_distance + 1.0e-6 &&
            merged_exit >= anchor_distance - 1.0e-6) {
            out = {merged_enter, anchor_distance, merged_exit};
            return out.width() > 1.0e-9;
        }
        if (index < intervals.size()) {
            merged_enter = intervals[index].enter;
            merged_exit = intervals[index].exit;
        }
    }
    return false;
}

void asymptotic_position(
    const std::vector<Submovement> &submovements,
    double &x,
    double &y
) {
    x = 0.0;
    y = 0.0;
    for (const Submovement &movement : submovements) {
        x += movement.x;
        y += movement.y;
    }
}

void summarize_region_trace(
    const AimPath &path,
    const ProjectedTargetRegion &visible_region,
    const ProjectedTargetRegion &safe_region,
    GeometryFeedbackSigmaDriftDiagnostics &diagnostics
) {
    bool previous_visible = false;
    bool previous_safe = false;
    bool have_previous = false;
    for (const AimSample &sample : path) {
        const Vec3 direction = look_direction_from_yaw_pitch(
            sample.yaw,
            sample.pitch
        );
        const bool visible = point_in_visible_region(
            visible_region,
            direction
        );
        const bool safe = point_in_visible_region_with_margin(
            safe_region,
            direction,
            0.0
        );
        if (visible && (!have_previous || !previous_visible)) {
            ++diagnostics.visible_entry_count;
            if (diagnostics.first_visible_entry_ms < 0.0) {
                diagnostics.first_visible_entry_ms = sample.t_ms;
            }
        } else if (!visible && have_previous && previous_visible) {
            ++diagnostics.visible_exit_count;
        }
        if (safe && (!have_previous || !previous_safe)) {
            ++diagnostics.safe_entry_count;
            if (diagnostics.first_safe_entry_ms < 0.0) {
                diagnostics.first_safe_entry_ms = sample.t_ms;
            }
        } else if (!safe && have_previous && previous_safe) {
            ++diagnostics.safe_exit_count;
        }
        previous_visible = visible;
        previous_safe = safe;
        have_previous = true;
    }
    diagnostics.final_visible = have_previous && previous_visible;
    diagnostics.final_safe = have_previous && previous_safe;
}

}  // namespace

AimPath generate_geometry_feedback_sigmadrift_path(
    const Orientation &start,
    const TargetMetrics &target,
    const VisibleDirectionComponents &visible_components,
    double angular_step_deg,
    const SigmaDriftConfig &motion_config,
    const GeometryFeedbackSigmaDriftConfig &feedback_config,
    std::uint64_t seed,
    GeometryFeedbackSigmaDriftDiagnostics *diagnostics
) {
    if (diagnostics != nullptr) {
        *diagnostics = {};
    }
    if (!(angular_step_deg > 0.0) || !std::isfinite(angular_step_deg) ||
        feedback_config.feedback_latency_mean_ms < 0.0 ||
        !std::isfinite(feedback_config.feedback_latency_mean_ms) ||
        feedback_config.feedback_latency_stddev_ms < 0.0 ||
        !std::isfinite(feedback_config.feedback_latency_stddev_ms) ||
        feedback_config.feedback_latency_min_ms < 0.0 ||
        !std::isfinite(feedback_config.feedback_latency_min_ms) ||
        feedback_config.feedback_latency_max_ms <
            feedback_config.feedback_latency_min_ms ||
        !std::isfinite(feedback_config.feedback_latency_max_ms) ||
        feedback_config.undershoot_width_min < 0.0 ||
        feedback_config.undershoot_width_max <
            feedback_config.undershoot_width_min ||
        feedback_config.overshoot_width_min < 0.0 ||
        feedback_config.overshoot_width_max <
            feedback_config.overshoot_width_min ||
        feedback_config.safe_margin_steps < 0.0 ||
        !std::isfinite(feedback_config.safe_margin_steps) ||
        feedback_config.max_corrections < 0) {
        return {};
    }

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

    const double step = std::max(1.0e-9, angular_step_deg);
    const double safe_margin_angle = std::min(
        feedback_config.safe_margin_steps * step * PI / 180.0,
        PI / 4.0
    );
    SafeTargetRegion safe_target{};
    if (!make_safe_target_region(
            projected_region,
            std::tan(safe_margin_angle),
            safe_target
        )) {
        return {};
    }
    const YawPitch motor_target = yaw_pitch_from_direction(
        safe_target.anchor_direction
    );
    if (diagnostics != nullptr) {
        diagnostics->motor_target_yaw = motor_target.yaw;
        diagnostics->motor_target_pitch = motor_target.pitch;
        diagnostics->applied_margin_steps =
            std::atan(safe_target.applied_margin) * 180.0 / PI / step;
        diagnostics->anchor_component_index =
            safe_target.anchor_component_index;
    }
    const double target_x =
        signed_angle_delta_degrees(motor_target.yaw, start.yaw) / step;
    const double target_y = (motor_target.pitch - start.pitch) / step;
    const double distance = std::hypot(target_x, target_y);
    if (distance < 1.0) {
        AimPath short_path{
            AimSample{start.yaw, start.pitch, 0.0},
            AimSample{motor_target.yaw, motor_target.pitch, 50.0},
        };
        if (diagnostics != nullptr) {
            summarize_region_trace(
                short_path,
                projected_region,
                safe_target.region,
                *diagnostics
            );
        }
        return short_path;
    }

    std::mt19937_64 rng(seed);
    auto uniform = [&](double lower, double upper) {
        return std::uniform_real_distribution<double>(lower, upper)(rng);
    };
    auto normal = [&](double mean, double sigma) {
        return std::normal_distribution<double>(mean, sigma)(rng);
    };
    auto gamma = [&](double shape, double scale) {
        return std::gamma_distribution<double>(shape, scale)(rng);
    };

    const double direction = std::atan2(target_y, target_x);
    const double tangent_x = target_x / distance;
    const double tangent_y = target_y / distance;
    const double normal_x = -tangent_y;
    const double normal_y = tangent_x;
    DirectionalTargetInterval target_interval{};
    const bool have_directional_interval = directional_target_interval(
        start,
        projected_region,
        safe_target,
        target_x,
        target_y,
        step,
        target_interval
    );
    const double target_width = have_directional_interval
        ? target_interval.width()
        : quantized_target_width(target, step, motion_config);
    if (!have_directional_interval) {
        target_interval = {
            distance - target_width * 0.5,
            distance,
            distance + target_width * 0.5,
        };
    }
    if (diagnostics != nullptr) {
        diagnostics->directional_width_steps = target_interval.width();
        diagnostics->s_enter_steps = target_interval.enter;
        diagnostics->s_anchor_steps = target_interval.anchor;
        diagnostics->s_exit_steps = target_interval.exit;
    }
    const double index_of_difficulty =
        std::log2(distance / target_width + 1.0);
    double movement_time =
        (motion_config.fitts_a + motion_config.fitts_b * index_of_difficulty) *
        std::exp(normal(0.0, 0.08));
    movement_time = std::max(movement_time, 80.0);

    const bool overshoot =
        uniform(0.0, 1.0) < motion_config.overshoot_prob;
    const double endpoint_error_widths = overshoot
        ? uniform(
            feedback_config.overshoot_width_min,
            feedback_config.overshoot_width_max
        )
        : -uniform(
            feedback_config.undershoot_width_min,
            feedback_config.undershoot_width_max
        );
    const double primary_distance = std::max(
        0.0,
        distance + endpoint_error_widths * target_width
    );
    if (diagnostics != nullptr) {
        diagnostics->primary_endpoint_steps = primary_distance;
    }
    const double primary_sigma = uniform(
        motion_config.primary_sigma_min,
        motion_config.primary_sigma_max
    );
    const double primary_peak = movement_time * uniform(
        motion_config.peak_time_ratio - 0.03,
        motion_config.peak_time_ratio + 0.03
    );

    std::vector<Submovement> submovements{
        Submovement{
            tangent_x * primary_distance,
            tangent_y * primary_distance,
            0.0,
            std::log(primary_peak) + primary_sigma * primary_sigma,
            primary_sigma,
            primary_peak,
            movement_time * 1.15,
        },
    };

    const double curvature_amplitude =
        distance * motion_config.curvature_scale * direction_factor(direction) *
        normal(0.0, 1.0);
    const double tremor_frequency = uniform(
        motion_config.tremor_freq_min,
        motion_config.tremor_freq_max
    );
    const double tremor_amplitude = uniform(
        motion_config.tremor_amp_min,
        motion_config.tremor_amp_max
    );
    const double tremor_phase_x = uniform(0.0, 2.0 * PI);
    const double tremor_phase_y = uniform(0.0, 2.0 * PI);
    const double gamma_scale =
        motion_config.sample_dt_mean / motion_config.gamma_shape;

    AimPath result;
    result.reserve(static_cast<std::size_t>(
        movement_time / motion_config.sample_dt_mean
    ) + 16);

    double ou_x = 0.0;
    double ou_y = 0.0;
    double t = 0.0;
    double previous_t = 0.0;
    double latest_tail_time = submovements.front().tail_time;
    auto sample_feedback_latency = [&]() {
        if (feedback_config.feedback_latency_stddev_ms == 0.0) {
            return clamp_double(
                feedback_config.feedback_latency_mean_ms,
                feedback_config.feedback_latency_min_ms,
                feedback_config.feedback_latency_max_ms
            );
        }
        return clamp_double(
            normal(
                feedback_config.feedback_latency_mean_ms,
                feedback_config.feedback_latency_stddev_ms
            ),
            feedback_config.feedback_latency_min_ms,
            feedback_config.feedback_latency_max_ms
        );
    };
    const double first_feedback_observation =
        submovements.front().peak_time;
    const double first_feedback_latency = sample_feedback_latency();
    double next_feedback_time = first_feedback_observation +
        first_feedback_latency;
    if (diagnostics != nullptr) {
        diagnostics->first_feedback_observation_ms =
            first_feedback_observation;
        diagnostics->first_feedback_latency_ms = first_feedback_latency;
    }
    bool feedback_pending = true;
    int correction_count = 0;

    while (result.size() < MAX_PATH_SAMPLES) {
        EvaluatedMotion motion = evaluate_submovements(submovements, t);
        const double primary_progress = lognormal_cdf(
            t,
            submovements.front().t0,
            submovements.front().mu,
            submovements.front().sigma
        );
        const double curve = curvature_profile(primary_progress);
        motion.x += normal_x * curvature_amplitude * curve;
        motion.y += normal_y * curvature_amplitude * curve;

        const double dt_ms = result.empty()
            ? motion_config.sample_dt_mean
            : t - previous_t;
        const double dt_s = dt_ms / 1000.0;
        ou_x += -motion_config.ou_theta * ou_x * dt_s +
            motion_config.ou_sigma * std::sqrt(dt_s) * normal(0.0, 1.0);
        ou_y += -motion_config.ou_theta * ou_y * dt_s +
            motion_config.ou_sigma * std::sqrt(dt_s) * normal(0.0, 1.0);

        const double t_s = t / 1000.0;
        const double tremor_modulation = 1.0 / (1.0 + motion.speed * 0.3);
        const double tremor_x = tremor_amplitude * tremor_modulation *
            std::sin(2.0 * PI * tremor_frequency * t_s + tremor_phase_x);
        const double tremor_y = tremor_amplitude * tremor_modulation *
            std::sin(2.0 * PI * tremor_frequency * t_s + tremor_phase_y);
        const double signal_noise_x =
            motion_config.sdn_k * motion.speed * normal(0.0, 1.0);
        const double signal_noise_y =
            motion_config.sdn_k * motion.speed * normal(0.0, 1.0);

        const double sample_x =
            motion.x + ou_x + tremor_x + signal_noise_x;
        const double sample_y =
            motion.y + ou_y + tremor_y + signal_noise_y;
        const Orientation sample_orientation = orientation_from_position(
            start,
            sample_x,
            sample_y,
            step
        );
        result.push_back({
            sample_orientation.yaw,
            sample_orientation.pitch,
            t,
        });

        if (feedback_pending && t >= next_feedback_time) {
            if (diagnostics != nullptr) {
                ++diagnostics->feedback_check_count;
            }
            const Vec3 current_direction = look_direction_from_yaw_pitch(
                sample_orientation.yaw,
                sample_orientation.pitch
            );
            const bool current_safe = point_in_visible_region_with_margin(
                safe_target.region,
                current_direction,
                0.0
            );
            double endpoint_x = 0.0;
            double endpoint_y = 0.0;
            asymptotic_position(submovements, endpoint_x, endpoint_y);

            if (current_safe ||
                correction_count >= feedback_config.max_corrections) {
                feedback_pending = false;
            } else {
                const double correction_reach = uniform(0.88, 1.02);
                const double correction_x =
                    (target_x - endpoint_x) * correction_reach;
                const double correction_y =
                    (target_y - endpoint_y) * correction_reach;
                const double correction_distance = std::hypot(
                    correction_x,
                    correction_y
                );
                if (correction_distance < 1.0e-9) {
                    feedback_pending = false;
                } else {
                    const double correction_id = std::log2(
                        correction_distance / target_width + 1.0
                    );
                    const double correction_duration = std::max(
                        60.0,
                        motion_config.fitts_a +
                            motion_config.fitts_b * correction_id
                    );
                    const double correction_sigma = uniform(
                        motion_config.correction_sigma_min,
                        motion_config.correction_sigma_max
                    );
                    const double correction_peak_delay =
                        correction_duration * uniform(0.30, 0.40);
                    const Submovement correction{
                        correction_x,
                        correction_y,
                        t,
                        std::log(correction_peak_delay) +
                            correction_sigma * correction_sigma,
                        correction_sigma,
                        t + correction_peak_delay,
                        t + correction_duration * 1.15,
                    };
                    submovements.push_back(correction);
                    ++correction_count;
                    if (diagnostics != nullptr) {
                        diagnostics->correction_count = correction_count;
                    }
                    const double feedback_observation = correction.peak_time;
                    next_feedback_time = feedback_observation +
                        sample_feedback_latency();
                    latest_tail_time = std::max(
                        latest_tail_time,
                        correction.tail_time
                    );
                }
            }
        }

        const double required_end_time = feedback_pending
            ? std::max(latest_tail_time, next_feedback_time)
            : latest_tail_time;
        if (t >= required_end_time) {
            break;
        }
        previous_t = t;
        t += clamp_double(
            gamma(motion_config.gamma_shape, gamma_scale),
            2.0,
            25.0
        );
        if (t > required_end_time) {
            t = required_end_time;
        }
    }

    if (result.empty()) {
        return {};
    }
    result.front() = AimSample{start.yaw, start.pitch, 0.0};
    if (diagnostics != nullptr) {
        summarize_region_trace(
            result,
            projected_region,
            safe_target.region,
            *diagnostics
        );
    }
    return result;
}

}  // namespace minecraft_miner::aim
