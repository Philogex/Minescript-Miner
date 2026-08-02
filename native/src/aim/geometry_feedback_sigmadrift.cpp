#include "minecraft_miner/aim/geometry_feedback_sigmadrift.hpp"

#include "minecraft_miner/aim/angle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

Vec3 direction_from_position(
    const Orientation &start,
    double x,
    double y,
    double angular_step_deg
) {
    const Orientation orientation = orientation_from_position(
        start,
        x,
        y,
        angular_step_deg
    );
    return look_direction_from_yaw_pitch(
        orientation.yaw,
        orientation.pitch
    );
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

}  // namespace

AimPath generate_geometry_feedback_sigmadrift_path(
    const Orientation &start,
    const TargetMetrics &target,
    const VisibleDirectionComponents &visible_components,
    double angular_step_deg,
    const SigmaDriftConfig &motion_config,
    const GeometryFeedbackSigmaDriftConfig &feedback_config,
    std::uint64_t seed
) {
    if (!(angular_step_deg > 0.0) || !std::isfinite(angular_step_deg) ||
        feedback_config.feedback_latency_ms < 0.0 ||
        !std::isfinite(feedback_config.feedback_latency_ms) ||
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
    const double target_x =
        signed_angle_delta_degrees(target.yaw, start.yaw) / step;
    const double target_y = (target.pitch - start.pitch) / step;
    const double distance = std::hypot(target_x, target_y);
    if (distance < 1.0) {
        return {
            AimSample{start.yaw, start.pitch, 0.0},
            AimSample{target.yaw, target.pitch, 50.0},
        };
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
    const double target_width = quantized_target_width(
        target,
        step,
        motion_config
    );
    const double index_of_difficulty =
        std::log2(distance / target_width + 1.0);
    double movement_time =
        (motion_config.fitts_a + motion_config.fitts_b * index_of_difficulty) *
        std::exp(normal(0.0, 0.08));
    movement_time = std::max(movement_time, 80.0);

    const bool overshoot =
        uniform(0.0, 1.0) < motion_config.overshoot_prob;
    const double reach = overshoot
        ? uniform(motion_config.overshoot_min, motion_config.overshoot_max)
        : uniform(motion_config.undershoot_min, motion_config.undershoot_max);
    const double primary_distance = distance * reach;
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
    const double safe_margin_angle = std::min(
        feedback_config.safe_margin_steps * step * PI / 180.0,
        PI / 4.0
    );
    const double safe_margin = std::tan(safe_margin_angle);

    AimPath result;
    result.reserve(static_cast<std::size_t>(
        movement_time / motion_config.sample_dt_mean
    ) + 16);

    double ou_x = 0.0;
    double ou_y = 0.0;
    double t = 0.0;
    double previous_t = 0.0;
    double latest_tail_time = submovements.front().tail_time;
    double next_feedback_time = submovements.front().peak_time +
        feedback_config.feedback_latency_ms;
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
            const Vec3 current_direction = look_direction_from_yaw_pitch(
                sample_orientation.yaw,
                sample_orientation.pitch
            );
            double endpoint_x = 0.0;
            double endpoint_y = 0.0;
            asymptotic_position(submovements, endpoint_x, endpoint_y);
            const Vec3 endpoint_direction = direction_from_position(
                start,
                endpoint_x,
                endpoint_y,
                step
            );
            // A safe current sample can still leave the region while the
            // remaining submovement tails decay, so both states must be safe.
            const bool current_safe = point_in_visible_region_with_margin(
                projected_region,
                current_direction,
                safe_margin
            );
            const bool endpoint_safe = point_in_visible_region_with_margin(
                projected_region,
                endpoint_direction,
                safe_margin
            );

            if ((current_safe && endpoint_safe) ||
                correction_count >= feedback_config.max_corrections) {
                feedback_pending = false;
            } else {
                Vec3 safe_direction{};
                if (!closest_safe_direction_in_visible_region(
                        projected_region,
                        current_direction,
                        safe_margin,
                        safe_direction
                    )) {
                    feedback_pending = false;
                } else {
                    const YawPitch safe_orientation =
                        yaw_pitch_from_direction(safe_direction);
                    const double safe_x = signed_angle_delta_degrees(
                        safe_orientation.yaw,
                        start.yaw
                    ) / step;
                    const double safe_y =
                        (safe_orientation.pitch - start.pitch) / step;
                    const double correction_x = safe_x - endpoint_x;
                    const double correction_y = safe_y - endpoint_y;
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
                        next_feedback_time = correction.peak_time +
                            feedback_config.feedback_latency_ms;
                        latest_tail_time = std::max(
                            latest_tail_time,
                            correction.tail_time
                        );
                    }
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
    return result;
}

}  // namespace minecraft_miner::aim
