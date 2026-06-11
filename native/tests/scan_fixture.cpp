#include "scan_fixture.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace minescript_miner::test {
namespace {

std::string next_value(std::istringstream &line, const char *key) {
    std::string value;
    if (!(line >> value)) {
        throw std::runtime_error(std::string("missing value for ") + key);
    }
    return value;
}

}  // namespace

ScanFixture load_scan_fixture(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open fixture: " + path);
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
        } else if (key == "reach") {
            fixture.reach = std::stod(next_value(line, key.c_str()));
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

}  // namespace minescript_miner::test
