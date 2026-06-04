#pragma once

#include "minescript_miner/vec.hpp"

namespace minescript_miner {

double angle_to_point(const Vec3 &look_dir, const Vec3 &point_from_eye);

Vec3 look_direction_from_yaw_pitch(double yaw_degrees, double pitch_degrees);

double angle_to_tri_corners(
    const Vec3 &eye,
    const Vec3 &look_dir,
    const Vec3 &a,
    const Vec3 &b,
    const Vec3 &c
);

}
