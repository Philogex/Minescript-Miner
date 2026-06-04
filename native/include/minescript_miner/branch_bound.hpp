#pragma once

#include "minescript_miner/tri2.hpp"

#include <cstdint>

namespace minescript_miner {

struct TargetBranch {
    Tri2 triangle{};
    std::uint16_t target_index = 0;
    std::uint16_t next_occluder = 0;
    double lower_angle_bound = 0.0;
};

constexpr TargetBranch target_branch(
    Tri2 triangle,
    std::uint16_t target_index,
    std::uint16_t next_occluder,
    double lower_angle_bound
) {
    return {triangle, target_index, next_occluder, lower_angle_bound};
}

}  // namespace minescript_miner
