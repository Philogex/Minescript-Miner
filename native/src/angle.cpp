#include "minescript_miner/angle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace minescript_miner {

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
