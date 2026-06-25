#include "minescript_miner/aim/angle.hpp"
#include "minescript_miner/scanner/branch_bound.hpp"
#include "minescript_miner/catalog/geometry_catalog.hpp"
#include "minescript_miner/scanner/scan_region.hpp"
#include "scan_fixture.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MINESCRIPT_MINER_CALLGRIND
#include <valgrind/callgrind.h>
#endif

namespace {

struct ScanInput {
    std::vector<std::uint16_t> shape_ids{};
    std::vector<std::uint16_t> target_indices{};
};

struct ProfileOptions {
    std::string fixture_path{};
    std::size_t iterations = 1;
    bool random_look = false;
    std::uint64_t random_seed = 0x4d595df4d0f33173ULL;
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

std::uint64_t parse_seed(const std::string &text) {
    std::size_t parsed = 0;
    const unsigned long long seed = std::stoull(text, &parsed);
    if (parsed != text.size()) {
        throw std::runtime_error("random-look seed must be an unsigned integer");
    }
    return static_cast<std::uint64_t>(seed);
}

ProfileOptions parse_options(int argc, char **argv) {
    if (argc < 2) {
        throw std::runtime_error("missing fixture path");
    }

    ProfileOptions options{};
    options.fixture_path = argv[1];

    int option_start = 2;
    if (option_start < argc && std::string(argv[option_start]).rfind("--", 0) != 0) {
        options.iterations = parse_iterations(argv[option_start]);
        ++option_start;
    }

    for (int index = option_start; index < argc; ++index) {
        const std::string option(argv[index]);
        constexpr const char *random_look_prefix = "--random-look=";
        if (option == "--random-look") {
            options.random_look = true;
        } else if (option.rfind(random_look_prefix, 0) == 0) {
            options.random_look = true;
            options.random_seed = parse_seed(option.substr(std::string(random_look_prefix).size()));
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }

    return options;
}

minescript_miner::Vec3 random_unit_direction(std::mt19937_64 &rng) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double z = 2.0 * unit(rng) - 1.0;
    const double angle = 2.0 * pi * unit(rng);
    const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
    return {radius * std::cos(angle), z, radius * std::sin(angle)};
}

void validate_result(
    const minescript_miner::test::ScanFixture &fixture,
    const minescript_miner::ScanRegionGeometry &geometry,
    const minescript_miner::BranchBoundResult &result,
    bool random_look
) {
    if (fixture.has_expect_world_faces &&
        geometry.world_faces.size() > fixture.expect_world_faces) {
        throw std::runtime_error("unexpected world face count");
    }
    if (fixture.has_expect_target_faces &&
        geometry.target_faces.size() != fixture.expect_target_faces) {
        throw std::runtime_error("unexpected target face count");
    }
    if (fixture.has_expect_found && result.found != fixture.expect_found) {
        throw std::runtime_error("unexpected target visibility result");
    }
    if (!random_look &&
        fixture.has_expect_min_clips &&
        result.stats.clips_performed < fixture.expect_min_clips) {
        throw std::runtime_error("solver performed fewer clips than expected");
    }
    if (!random_look &&
        result.found && fixture.has_expect_angle_range &&
        (result.angle < fixture.expect_angle_min ||
         result.angle > fixture.expect_angle_max)) {
        throw std::runtime_error("result angle is outside the expected range");
    }
}

}  // namespace

int main(int argc, char **argv) {
    using namespace minescript_miner;

    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " FIXTURE [ITERATIONS] [--random-look[=SEED]]\n";
        return 2;
    }

    try {
        const ProfileOptions options = parse_options(argc, argv);
        const test::ScanFixture fixture = test::load_scan_fixture(options.fixture_path);
        const ScanInput input = make_scan_input(fixture);
        const Vec3 look_direction =
            look_direction_from_yaw_pitch(fixture.yaw, fixture.pitch);
        std::mt19937_64 rng(options.random_seed);

        BranchBoundResult last_result{};
        std::size_t last_world_faces = 0;
        std::size_t last_target_faces = 0;
        double checksum = 0.0;

#ifdef MINESCRIPT_MINER_CALLGRIND
        CALLGRIND_START_INSTRUMENTATION;
        CALLGRIND_ZERO_STATS;
#endif
        for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
            const Vec3 iteration_look_direction = options.random_look
                ? random_unit_direction(rng)
                : look_direction;
            const ScanRegionGeometry geometry = build_scan_region_geometry(
                input.shape_ids,
                input.target_indices,
                fixture.position,
                iteration_look_direction,
                fixture.side,
                fixture.reach
            );
            last_result = solve_visible_target(
                geometry,
                fixture.position,
                iteration_look_direction,
                fixture.reach
            );
            validate_result(fixture, geometry, last_result, options.random_look);

            last_world_faces = geometry.world_faces.size();
            last_target_faces = geometry.target_faces.size();
            checksum += last_result.found
                ? last_result.angle + last_result.direction.x +
                    last_result.direction.y + last_result.direction.z
                : -1.0;
        }
#ifdef MINESCRIPT_MINER_CALLGRIND
        CALLGRIND_STOP_INSTRUMENTATION;
#endif

        if (!std::isfinite(checksum)) {
            throw std::runtime_error("non-finite profiling checksum");
        }

        std::cout
            << "iterations=" << options.iterations
            << " random_look=" << options.random_look
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
