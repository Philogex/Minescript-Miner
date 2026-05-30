#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace minescript_miner {

inline constexpr int SHAPE_CATALOG_VERSION = 1;
inline constexpr std::int32_t SHAPE_EMPTY = 0;
inline constexpr std::int32_t SHAPE_FULL_CUBE = 1;
inline constexpr std::int32_t SHAPE_OAK_SLAB_BOTTOM = 2;
inline constexpr std::int32_t SHAPE_OAK_SLAB_TOP = 3;

std::string shape_id_name(std::int32_t shape_id);
std::int32_t shape_count();
const std::vector<std::string> &shape_names();

inline bool is_empty_shape(std::int32_t shape_id) {
    return shape_id == SHAPE_EMPTY;
}

}
