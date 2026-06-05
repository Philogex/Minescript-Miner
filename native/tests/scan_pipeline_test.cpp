#include "minescript_miner/branch_bound.hpp"
#include "minescript_miner/angle.hpp"
#include "minescript_miner/geometry_catalog.hpp"
#include "minescript_miner/scan_region.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SparseBlock {
    std::uint16_t index = 0;
    std::uint16_t shape_id = 0;
    bool target = false;
};

struct ScanFixture {
    int fixture_version = 0;
    int shape_catalog_version = 0;
    std::int32_t side = 0;
    minescript_miner::Vec3 position{};
    double yaw = 0.0;
    double pitch = 0.0;
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

std::string next_value(std::istringstream &line, const char *key) {
    std::string value;
    if (!(line >> value)) {
        throw std::runtime_error(std::string("missing value for ") + key);
    }
    return value;
}

ScanFixture load_fixture(const char *path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("cannot open fixture: ") + path);
    }

    ScanFixture fixture{};
    std::string raw_line;
    std::size_t line_number = 0;
    while (std::getline(input, raw_line)) {
        ++line_number;
        const std::size_t comment = raw_line.find('#');
        if (comment != std::string::npos) {
            raw_line.erase(comment);
        }

        std::istringstream line(raw_line);
        std::string key;
        if (!(line >> key)) {
            continue;
        }

        if (key == "fixture_version") {
            fixture.fixture_version = std::stoi(next_value(line, key.c_str()));
        } else if (key == "shape_catalog_version") {
            fixture.shape_catalog_version = std::stoi(next_value(line, key.c_str()));
        } else if (key == "side") {
            fixture.side = std::stoi(next_value(line, key.c_str()));
        } else if (key == "position") {
            line >> fixture.position.x >> fixture.position.y >> fixture.position.z;
        } else if (key == "orientation_yaw_pitch") {
            line >> fixture.yaw >> fixture.pitch;
        } else if (key == "default_shape") {
            fixture.default_shape =
                static_cast<std::uint16_t>(std::stoul(next_value(line, key.c_str())));
        } else if (key == "block" || key == "target") {
            SparseBlock block{};
            unsigned index = 0;
            unsigned shape_id = 0;
            line >> index >> shape_id;
            block.index = static_cast<std::uint16_t>(index);
            block.shape_id = static_cast<std::uint16_t>(shape_id);
            block.target = key == "target";
            fixture.blocks.push_back(block);
        } else if (key == "expect_found") {
            fixture.has_expect_found = true;
            fixture.expect_found = std::stoi(next_value(line, key.c_str())) != 0;
        } else if (key == "expect_angle_min") {
            fixture.has_expect_angle_range = true;
            fixture.expect_angle_min = std::stod(next_value(line, key.c_str()));
        } else if (key == "expect_angle_max") {
            fixture.has_expect_angle_range = true;
            fixture.expect_angle_max = std::stod(next_value(line, key.c_str()));
        } else if (key == "expect_world_faces") {
            fixture.has_expect_world_faces = true;
            fixture.expect_world_faces = std::stoull(next_value(line, key.c_str()));
        } else if (key == "expect_target_faces") {
            fixture.has_expect_target_faces = true;
            fixture.expect_target_faces = std::stoull(next_value(line, key.c_str()));
        } else if (key == "expect_min_clips") {
            fixture.has_expect_min_clips = true;
            fixture.expect_min_clips = std::stoull(next_value(line, key.c_str()));
        } else {
            throw std::runtime_error(
                "unknown fixture key at line " + std::to_string(line_number) + ": " + key
            );
        }

        if (line.fail()) {
            throw std::runtime_error(
                "invalid fixture value at line " + std::to_string(line_number) + ": " + key
            );
        }
    }
    return fixture;
}

}  // namespace

int main(int argc, char **argv) {
    using namespace minescript_miner;

    assert(argc == 2);
    const Vec3 round_trip_direction =
        look_direction_from_yaw_pitch(-98.1, -13.2);
    const YawPitch round_trip_orientation =
        yaw_pitch_from_direction(round_trip_direction);
    assert(std::abs(round_trip_orientation.yaw - -98.1) < 1e-12);
    assert(std::abs(round_trip_orientation.pitch - -13.2) < 1e-12);

    const ScanFixture fixture = load_fixture(argv[1]);
    assert(fixture.fixture_version == 1);
    assert(fixture.shape_catalog_version == GEOMETRY_SHAPE_CATALOG_VERSION);
    assert(fixture.side > 0);
    assert(fixture.side <= 39);
    assert(fixture.default_shape < GEOMETRY_SHAPE_COUNT);

    const std::size_t block_count =
        static_cast<std::size_t>(fixture.side) *
        static_cast<std::size_t>(fixture.side) *
        static_cast<std::size_t>(fixture.side);
    assert(block_count <= 65535);

    std::vector<std::uint16_t> shape_ids(block_count, fixture.default_shape);
    std::vector<std::uint16_t> target_indices{};
    for (const SparseBlock &block : fixture.blocks) {
        assert(block.index < block_count);
        assert(block.shape_id < GEOMETRY_SHAPE_COUNT);
        shape_ids[block.index] = block.shape_id;
        if (block.target) {
            target_indices.push_back(block.index);
        }
    }
    const Vec3 look_direction =
        look_direction_from_yaw_pitch(fixture.yaw, fixture.pitch);
    const ScanRegionGeometry geometry = build_scan_region_geometry(
        shape_ids,
        target_indices,
        fixture.position,
        look_direction,
        fixture.side
    );
    if (fixture.has_expect_world_faces) {
        assert(geometry.world_faces.size() == fixture.expect_world_faces);
    }
    if (fixture.has_expect_target_faces) {
        assert(geometry.target_faces.size() == fixture.expect_target_faces);
    }

    const BranchBoundResult result =
        solve_visible_target(geometry, fixture.position, look_direction);
    if (fixture.has_expect_found) {
        assert(result.found == fixture.expect_found);
    }
    if (fixture.has_expect_min_clips) {
        assert(result.stats.clips_performed >= fixture.expect_min_clips);
    }

    if (result.found) {
        assert(std::isfinite(result.angle));
        if (fixture.has_expect_angle_range) {
            assert(result.angle >= fixture.expect_angle_min);
            assert(result.angle <= fixture.expect_angle_max);
        }
        assert(std::abs(length_squared(result.direction) - 1.0) < 1e-12);

        const bool result_is_target_face = std::any_of(
            geometry.target_faces.begin(),
            geometry.target_faces.end(),
            [&result](const TargetFaceCandidate &target) {
                return target.world_face_index == result.target_world_face_index;
            }
        );
        assert(result_is_target_face);
    }

    std::cout
        << "world_faces=" << geometry.world_faces.size()
        << " target_faces=" << geometry.target_faces.size()
        << " found=" << result.found
        << " angle=" << result.angle
        << " targets_considered=" << result.stats.target_faces_considered
        << " targets_pruned=" << result.stats.target_faces_pruned
        << " occluders_prepared=" << result.stats.occluders_prepared
        << " effective_occluders=" << result.stats.effective_occluders
        << " branches=" << result.stats.branches_visited
        << " clips=" << result.stats.clips_performed
        << '\n';
}
