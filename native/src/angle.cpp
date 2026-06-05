#include "minescript_miner/angle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace minescript_miner {

namespace {

constexpr double PI = 3.14159265358979323846;

}  // namespace

double angle_to_point(const Vec3 &look_dir, const Vec3 &point_from_eye) {
    const double look_length_squared = length_squared(look_dir);
    const double point_length_squared = length_squared(point_from_eye);
    if (look_length_squared <= 0.0 || point_length_squared <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double denominator = std::sqrt(look_length_squared * point_length_squared);
    const double cosine = std::clamp(dot(look_dir, point_from_eye) / denominator, -1.0, 1.0);
    return std::acos(cosine);
}

Vec3 look_direction_from_yaw_pitch(double yaw_degrees, double pitch_degrees) {
    const double yaw_rad = yaw_degrees * PI / 180.0;
    const double pitch_rad = pitch_degrees * PI / 180.0;
    return {
        -std::sin(yaw_rad) * std::cos(pitch_rad),
        -std::sin(pitch_rad),
        std::cos(yaw_rad) * std::cos(pitch_rad),
    };
}

YawPitch yaw_pitch_from_direction(const Vec3 &direction) {
    if (length_squared(direction) <= 0.0) {
        return {};
    }

    const double horizontal = std::hypot(direction.x, direction.z);
    double yaw = std::atan2(-direction.x, direction.z) * 180.0 / PI;
    double pitch = std::atan2(-direction.y, horizontal) * 180.0 / PI;
    if (yaw == 0.0) {
        yaw = 0.0;
    }
    if (pitch == 0.0) {
        pitch = 0.0;
    }
    return {yaw, pitch};
}

double angle_to_tri_corners(
    const Vec3 &eye,
    const Vec3 &look_dir,
    const Vec3 &a,
    const Vec3 &b,
    const Vec3 &c
) {
    return std::min({
        angle_to_point(look_dir, a - eye),
        angle_to_point(look_dir, b - eye),
        angle_to_point(look_dir, c - eye),
    });
}

}  // namespace minescript_miner
