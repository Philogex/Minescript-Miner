#include "minescript_miner/angle.hpp"
#include "minescript_miner/branch_bound.hpp"
#include "minescript_miner/geometry_catalog.hpp"
#include "minescript_miner/scan_region.hpp"
#include "scan_fixture.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ScanInput {
    std::vector<std::uint16_t> shape_ids{};
    std::vector<std::uint16_t> target_indices{};
};

ScanInput make_scan_input(const minescript_miner::test::ScanFixture &fixture) {
    using namespace minescript_miner;

    if (fixture.fixture_version != 1) {
        throw std::runtime_error("unsupported fixture version");
    }
    if (fixture.shape_catalog_version != GEOMETRY_SHAPE_CATALOG_VERSION) {
        throw std::runtime_error("fixture shape catalog version mismatch");
    }
    if (fixture.side <= 0 || fixture.side > 39) {
        throw std::runtime_error("fixture side must be in [1, 39]");
    }
    if (fixture.default_shape >= GEOMETRY_SHAPE_COUNT) {
        throw std::runtime_error("fixture default shape is outside the catalog");
    }

    const std::size_t side = static_cast<std::size_t>(fixture.side);
    const std::size_t block_count = side * side * side;
    if (block_count > 65535) {
        throw std::runtime_error("fixture exceeds uint16 block indices");
    }

    ScanInput input{};
    input.shape_ids.assign(block_count, fixture.default_shape);
    for (const test::SparseBlock &block : fixture.blocks) {
        if (block.index >= block_count) {
            throw std::runtime_error("fixture block index is outside the scan cube");
        }
        if (block.shape_id >= GEOMETRY_SHAPE_COUNT) {
            throw std::runtime_error("fixture shape id is outside the catalog");
        }
        input.shape_ids[block.index] = block.shape_id;
        if (block.target) {
            input.target_indices.push_back(block.index);
        }
    }
    return input;
}

std::size_t parse_iterations(const char *value) {
    const std::string text(value);
    std::size_t parsed = 0;
    const unsigned long long iterations = std::stoull(text, &parsed);
    if (parsed != text.size() || iterations == 0) {
        throw std::runtime_error("iterations must be a positive integer");
    }
    return static_cast<std::size_t>(iterations);
}

void validate_result(
    const minescript_miner::test::ScanFixture &fixture,
    const minescript_miner::ScanRegionGeometry &geometry,
    const minescript_miner::BranchBoundResult &result
) {
    if (fixture.has_expect_world_faces &&
        geometry.world_faces.size() != fixture.expect_world_faces) {
        throw std::runtime_error("unexpected world face count");
    }
    if (fixture.has_expect_target_faces &&
        geometry.target_faces.size() != fixture.expect_target_faces) {
        throw std::runtime_error("unexpected target face count");
    }
    if (fixture.has_expect_found && result.found != fixture.expect_found) {
        throw std::runtime_error("unexpected target visibility result");
    }
    if (fixture.has_expect_min_clips &&
        result.stats.clips_performed < fixture.expect_min_clips) {
        throw std::runtime_error("solver performed fewer clips than expected");
    }
    if (result.found && fixture.has_expect_angle_range &&
        (result.angle < fixture.expect_angle_min ||
         result.angle > fixture.expect_angle_max)) {
        throw std::runtime_error("result angle is outside the expected range");
    }
}

}  // namespace

int main(int argc, char **argv) {
    using namespace minescript_miner;

    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0] << " FIXTURE [ITERATIONS]\n";
        return 2;
    }

    try {
        const test::ScanFixture fixture = test::load_scan_fixture(argv[1]);
        const ScanInput input = make_scan_input(fixture);
        const std::size_t iterations = argc == 3 ? parse_iterations(argv[2]) : 1;
        const Vec3 look_direction =
            look_direction_from_yaw_pitch(fixture.yaw, fixture.pitch);

        BranchBoundResult last_result{};
        std::size_t last_world_faces = 0;
        std::size_t last_target_faces = 0;
        double checksum = 0.0;

        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            const ScanRegionGeometry geometry = build_scan_region_geometry(
                input.shape_ids,
                input.target_indices,
                fixture.position,
                look_direction,
                fixture.side,
                fixture.reach
            );
            last_result = solve_visible_target(
                geometry,
                fixture.position,
                look_direction,
                fixture.reach
            );
            validate_result(fixture, geometry, last_result);

            last_world_faces = geometry.world_faces.size();
            last_target_faces = geometry.target_faces.size();
            checksum += last_result.found
                ? last_result.angle + last_result.direction.x +
                    last_result.direction.y + last_result.direction.z
                : -1.0;
        }

        if (!std::isfinite(checksum)) {
            throw std::runtime_error("non-finite profiling checksum");
        }

        std::cout
            << "iterations=" << iterations
            << " world_faces=" << last_world_faces
            << " target_faces=" << last_target_faces
            << " found=" << last_result.found
            << " angle=" << last_result.angle
            << " branches=" << last_result.stats.branches_visited
            << " clips=" << last_result.stats.clips_performed
            << " checksum=" << checksum
            << '\n';
    } catch (const std::exception &error) {
        std::cerr << "scan_pipeline_profile: " << error.what() << '\n';
        return 1;
    }
}
