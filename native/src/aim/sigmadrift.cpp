#include "minecraft_miner/aim/sigmadrift.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace minecraft_miner::aim {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double SQRT_2 = 1.414213562373095048801688724209698079;

double clamp_double(double value, double minimum, double maximum) {
    return std::max(minimum, std::min(maximum, value));
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
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2.0 * PI) * dt);
}

double curvature_profile(double s) {
    if (s <= 0.0 || s >= 1.0) {
        return 0.0;
    }
    const double v = s * s * (1.0 - s) * (1.0 - s) * (1.0 - s);
    constexpr double norm = 0.4 * 0.4 * 0.6 * 0.6 * 0.6;
    return v / norm;
}

double direction_factor(double angle) {
    const double sa = std::abs(std::sin(angle));
    const double ca = std::abs(std::cos(angle));
    return 0.5 + 0.8 * sa - 0.15 * ca;
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
        return std::max(1.0, std::min(width_yaw, width_pitch) / angular_step_deg);
    }
    return std::max(1.0, config.target_width);
}

struct Correction {
    double distance = 0.0;
    double t0 = 0.0;
    double mu = 0.0;
    double sigma = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
};

}  // namespace

AimPath generate_sigmadrift_path(
    const Orientation &start,
    const TargetMetrics &target,
    double angular_step_deg,
    const SigmaDriftConfig &config,
    std::uint64_t seed
) {
    const double step = std::max(1.0e-9, angular_step_deg);
    const double target_yaw = continuous_yaw_near(target.yaw, start.yaw);
    const double dx = (target_yaw - start.yaw) / step;
    const double dy = (target.pitch - start.pitch) / step;
    const double distance = std::hypot(dx, dy);

    if (distance < 1.0) {
        return {
            AimSample{start.yaw, start.pitch, 0.0},
            AimSample{target_yaw, target.pitch, 50.0},
        };
    }

    std::mt19937_64 rng(seed);
    auto uniform = [&](double lo, double hi) {
        return std::uniform_real_distribution<double>(lo, hi)(rng);
    };
    auto normal = [&](double mean, double sigma) {
        return std::normal_distribution<double>(mean, sigma)(rng);
    };
    auto gamma = [&](double shape, double scale) {
        return std::gamma_distribution<double>(shape, scale)(rng);
    };

    const double direction = std::atan2(dy, dx);
    const double tx = dx / distance;
    const double ty = dy / distance;
    const double nx = -ty;
    const double ny = tx;

    const double target_width = quantized_target_width(target, step, config);
    const double id = std::log2(distance / target_width + 1.0);
    double movement_time =
        (config.fitts_a + config.fitts_b * id) * std::exp(normal(0.0, 0.08));
    movement_time = std::max(movement_time, 80.0);

    const bool overshoot = uniform(0.0, 1.0) < config.overshoot_prob;
    const double reach = overshoot
        ? uniform(config.overshoot_min, config.overshoot_max)
        : uniform(config.undershoot_min, config.undershoot_max);

    const double primary_distance = distance * reach;
    const double primary_sigma =
        uniform(config.primary_sigma_min, config.primary_sigma_max);
    const double peak_t = movement_time * uniform(
        config.peak_time_ratio - 0.03,
        config.peak_time_ratio + 0.03
    );
    const double primary_mu = std::log(peak_t) + primary_sigma * primary_sigma;

    std::vector<Correction> corrections;
    const double remaining = distance - primary_distance;
    if (std::abs(remaining) > 0.5) {
        const double dir = remaining > 0.0 ? 1.0 : -1.0;
        const double correction_distance = std::abs(remaining) * uniform(0.88, 1.02);
        const double correction_sigma = uniform(
            config.correction_sigma_min,
            config.correction_sigma_max
        );
        const double correction_peak = movement_time * uniform(0.12, 0.18);
        corrections.push_back(Correction{
            correction_distance,
            movement_time * uniform(0.55, 0.68),
            std::log(correction_peak) + correction_sigma * correction_sigma,
            correction_sigma,
            tx * dir,
            ty * dir,
        });

        const double left = remaining - correction_distance * dir;
        if (std::abs(left) > 0.3 &&
            uniform(0.0, 1.0) < config.second_correction_prob) {
            const double second_dir = left > 0.0 ? 1.0 : -1.0;
            const double second_distance = std::abs(left) * uniform(0.85, 1.05);
            const double second_sigma = uniform(0.10, 0.16);
            const double second_peak = movement_time * uniform(0.08, 0.12);
            corrections.push_back(Correction{
                second_distance,
                movement_time * uniform(0.78, 0.88),
                std::log(second_peak) + second_sigma * second_sigma,
                second_sigma,
                tx * second_dir,
                ty * second_dir,
            });
        }
    }

    const double curvature_amplitude =
        distance * config.curvature_scale * direction_factor(direction) *
        normal(0.0, 1.0);
    const double tremor_frequency =
        uniform(config.tremor_freq_min, config.tremor_freq_max);
    const double tremor_amplitude =
        uniform(config.tremor_amp_min, config.tremor_amp_max);
    const double tremor_phase_x = uniform(0.0, 2.0 * PI);
    const double tremor_phase_y = uniform(0.0, 2.0 * PI);

    const double total_t = movement_time * 1.15;
    const double gamma_scale = config.sample_dt_mean / config.gamma_shape;
    std::vector<double> times{0.0};
    for (double t = 0.0; t < total_t;) {
        const double dt = clamp_double(gamma(config.gamma_shape, gamma_scale), 2.0, 25.0);
        t += dt;
        if (t <= total_t + 15.0) {
            times.push_back(t);
        }
    }

    AimPath result;
    result.reserve(times.size() + 1);

    double ou_x = 0.0;
    double ou_y = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double t = times[i];
        const double dt_ms = (i > 0) ? (t - times[i - 1]) : config.sample_dt_mean;
        const double dt_s = dt_ms / 1000.0;
        const double s = lognormal_cdf(t, 0.0, primary_mu, primary_sigma);

        double x = tx * primary_distance * s;
        double y = ty * primary_distance * s;

        const double curve = curvature_profile(s);
        x += nx * curvature_amplitude * curve;
        y += ny * curvature_amplitude * curve;

        for (const Correction &correction : corrections) {
            const double correction_s = lognormal_cdf(
                t,
                correction.t0,
                correction.mu,
                correction.sigma
            );
            x += correction.dir_x * correction.distance * correction_s;
            y += correction.dir_y * correction.distance * correction_s;
        }

        double speed =
            primary_distance * lognormal_pdf(t, 0.0, primary_mu, primary_sigma);
        for (const Correction &correction : corrections) {
            speed += correction.distance * lognormal_pdf(
                t,
                correction.t0,
                correction.mu,
                correction.sigma
            );
        }

        ou_x += -config.ou_theta * ou_x * dt_s +
            config.ou_sigma * std::sqrt(dt_s) * normal(0.0, 1.0);
        ou_y += -config.ou_theta * ou_y * dt_s +
            config.ou_sigma * std::sqrt(dt_s) * normal(0.0, 1.0);

        const double t_s = t / 1000.0;
        const double tremor_mod = 1.0 / (1.0 + speed * 0.3);
        const double tremor_x =
            tremor_amplitude * tremor_mod *
            std::sin(2.0 * PI * tremor_frequency * t_s + tremor_phase_x);
        const double tremor_y =
            tremor_amplitude * tremor_mod *
            std::sin(2.0 * PI * tremor_frequency * t_s + tremor_phase_y);

        const double sdn_x = config.sdn_k * speed * normal(0.0, 1.0);
        const double sdn_y = config.sdn_k * speed * normal(0.0, 1.0);

        result.push_back(AimSample{
            start.yaw + (x + ou_x + tremor_x + sdn_x) * step,
            clamp_double(start.pitch + (y + ou_y + tremor_y + sdn_y) * step, -90.0, 90.0),
            t,
        });
    }

    if (result.empty()) {
        result.push_back(AimSample{start.yaw, start.pitch, 0.0});
    } else {
        result.front() = AimSample{start.yaw, start.pitch, 0.0};
    }
    if (result.back().t_ms <= 0.0 ||
        std::abs(result.back().yaw - target_yaw) > 1.0e-9 ||
        std::abs(result.back().pitch - target.pitch) > 1.0e-9) {
        result.push_back(AimSample{target_yaw, target.pitch, total_t});
    } else {
        result.back() = AimSample{target_yaw, target.pitch, result.back().t_ms};
    }
    return result;
}

}  // namespace minecraft_miner::aim
