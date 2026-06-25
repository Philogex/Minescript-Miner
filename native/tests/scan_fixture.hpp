#pragma once

#include "minescript_miner/geometry/vec.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace minescript_miner::test {

struct SparseBlock {
    std::uint16_t index = 0;
    std::uint16_t shape_id = 0;
    bool target = false;
};

struct ScanFixture {
    int fixture_version = 0;
    int shape_catalog_version = 0;
    std::int32_t side = 0;
    Vec3 position{};
    double yaw = 0.0;
    double pitch = 0.0;
    double reach = std::numeric_limits<double>::infinity();
    std::uint16_t default_shape = 0;
    std::vector<SparseBlock> blocks{};
    bool has_expect_found = false;
    bool expect_found = false;
    bool has_expect_angle_range = false;
    double expect_angle_min = 0.0;
    double expect_angle_max = 0.0;
    bool has_expect_world_faces = false;
    std::size_t expect_world_faces = 0;
    bool has_expect_target_faces = false;
    std::size_t expect_target_faces = 0;
    bool has_expect_min_clips = false;
    std::uint64_t expect_min_clips = 0;
};

ScanFixture load_scan_fixture(const std::string &path);

}  // namespace minescript_miner::test
