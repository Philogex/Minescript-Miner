#pragma once

namespace minescript_miner {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

constexpr Vec3 vec3(double x, double y, double z) {
    return {x, y, z};
}

constexpr Vec3 operator-(const Vec3 &lhs, const Vec3 &rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

constexpr double dot(const Vec3 &lhs, const Vec3 &rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

constexpr double length_squared(const Vec3 &value) {
    return dot(value, value);
}

}
