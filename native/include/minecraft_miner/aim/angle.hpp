#pragma once

#include "minecraft_miner/geometry/vec.hpp"

namespace minecraft_miner {

struct YawPitch {
    double yaw = 0.0;
    double pitch = 0.0;
};

double angle_to_point(const Vec3 &look_dir, const Vec3 &point_from_eye);

Vec3 look_direction_from_yaw_pitch(double yaw_degrees, double pitch_degrees);
YawPitch yaw_pitch_from_direction(const Vec3 &direction);

double angle_to_tri_corners(
    const Vec3 &eye,
    const Vec3 &look_dir,
    const Vec3 &a,
    const Vec3 &b,
    const Vec3 &c
);

}
