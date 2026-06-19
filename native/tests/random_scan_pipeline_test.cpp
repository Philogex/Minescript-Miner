#include "minescript_miner/angle.hpp"
#include "minescript_miner/branch_bound.hpp"
#include "minescript_miner/catalog_contract.hpp"
#include "minescript_miner/geometry_catalog.hpp"
#include "minescript_miner/scan_region.hpp"
#include "minescript_miner/target_solver.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using minescript_miner::BranchBoundResult;
using minescript_miner::ScanRegionGeometry;
using minescript_miner::TargetFaceCandidate;
using minescript_miner::Vec3;

constexpr std::int32_t kSide = minescript_miner::MAX_CUBE_SIDE;
constexpr std::uint16_t kCenterIndex =
    static_cast<std::uint16_t>((kSide / 2) + (kSide / 2) * kSide +
                               (kSide / 2) * kSide * kSide);

struct RandomConfig {
    std::uint64_t seed = 0x4d595df4d0f33173ULL;
    std::uint32_t cases = 5;
    double density = 0.25;
    std::uint16_t target_count = 5;
};

struct CaseData {
    std::vector<std::uint16_t> shape_ids{};
    std::vector<std::uint16_t> target_indices{};
    Vec3 eye{0.5, 0.5, 0.5};
    Vec3 look_direction{0.0, 0.0, 1.0};
};

struct SolveStats {
    std::uint64_t found = 0;
    std::uint64_t missing = 0;
    std::uint64_t world_faces = 0;
    std::uint64_t target_faces = 0;
    std::uint64_t branches = 0;
    std::uint64_t clips = 0;
    std::uint64_t reference_us = 0;
    std::uint64_t candidate_us = 0;
    std::uint64_t checksum = 1469598103934665603ULL;
};

class SplitMix64 {
  public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }

    double unit() {
        return static_cast<double>(next() >> 11U) *
               (1.0 / static_cast<double>(std::uint64_t{1} << 53U));
    }

    std::uint64_t bounded(std::uint64_t upper_exclusive) {
        if (upper_exclusive == 0) {
            throw std::invalid_argument("empty random range");
        }
        return next() % upper_exclusive;
    }

  private:
    std::uint64_t state_;
};

std::uint64_t parse_u64(const char *text) {
    return std::stoull(text, nullptr, 0);
}

double parse_density(const char *text) {
    const double density = std::stod(text);
    if (density < 0.0 || density > 1.0) {
        throw std::invalid_argument("density must be between 0 and 1");
    }
    return density;
}

RandomConfig parse_config(int argc, char **argv) {
    RandomConfig config{};
    if (argc > 1) {
        config.seed = parse_u64(argv[1]);
    }
    if (argc > 2) {
        config.cases = static_cast<std::uint32_t>(parse_u64(argv[2]));
    }
    if (argc > 3) {
        config.density = parse_density(argv[3]);
    }
    if (argc > 4) {
        config.target_count = static_cast<std::uint16_t>(parse_u64(argv[4]));
    }
    if (argc > 5) {
        throw std::invalid_argument(
            "usage: random_scan_pipeline_test [seed] [cases] [density] [targets]"
        );
    }
    if (config.cases == 0) {
        throw std::invalid_argument("cases must be positive");
    }
    if (config.target_count == 0) {
        throw std::invalid_argument("targets must be positive");
    }
    constexpr std::size_t block_count =
        static_cast<std::size_t>(kSide) * kSide * kSide;
    if (config.target_count >= block_count) {
        throw std::invalid_argument("targets must be smaller than block count");
    }
    return config;
}

Vec3 random_look_direction(SplitMix64 &rng) {
    const double yaw = rng.unit() * 360.0 - 180.0;
    const double pitch = rng.unit() * 160.0 - 80.0;
    return minescript_miner::look_direction_from_yaw_pitch(yaw, pitch);
}

void shuffle_prefix(
    std::vector<std::uint16_t> &values,
    std::size_t prefix_size,
    SplitMix64 &rng
) {
    prefix_size = std::min(prefix_size, values.size());
    for (std::size_t index = 0; index < prefix_size; ++index) {
        const std::size_t selected =
            index + static_cast<std::size_t>(rng.bounded(values.size() - index));
        std::swap(values[index], values[selected]);
    }
}

CaseData make_case(const RandomConfig &config, std::uint64_t case_index) {
    SplitMix64 rng(config.seed + case_index * 0x9e3779b97f4a7c15ULL);
    constexpr std::size_t block_count =
        static_cast<std::size_t>(kSide) * kSide * kSide;

    CaseData data{};
    data.shape_ids.assign(block_count, minescript_miner::SHAPE_EMPTY);
    std::vector<std::uint16_t> full_indices{};
    full_indices.reserve(static_cast<std::size_t>(
        static_cast<double>(block_count) * config.density + config.target_count
    ));

    for (std::size_t index = 0; index < block_count; ++index) {
        if (index == kCenterIndex || rng.unit() >= config.density) {
            continue;
        }
        data.shape_ids[index] = minescript_miner::SHAPE_FULL_CUBE;
        full_indices.push_back(static_cast<std::uint16_t>(index));
    }

    while (full_indices.size() < config.target_count) {
        const auto index = static_cast<std::uint16_t>(
            rng.bounded(block_count)
        );
        if (index == kCenterIndex ||
            data.shape_ids[index] == minescript_miner::SHAPE_FULL_CUBE) {
            continue;
        }
        data.shape_ids[index] = minescript_miner::SHAPE_FULL_CUBE;
        full_indices.push_back(index);
    }

    shuffle_prefix(full_indices, config.target_count, rng);
    data.target_indices.assign(
        full_indices.begin(),
        full_indices.begin() + config.target_count
    );
    data.look_direction = random_look_direction(rng);
    return data;
}

BranchBoundResult solve_reference(
    const ScanRegionGeometry &geometry,
    const Vec3 &eye,
    const Vec3 &look_direction
) {
    return minescript_miner::solve_visible_target(
        geometry,
        eye,
        look_direction,
        std::numeric_limits<double>::infinity()
    );
}

BranchBoundResult solve_candidate(
    const ScanRegionGeometry &geometry,
    const Vec3 &eye,
    const Vec3 &look_direction
) {
    return minescript_miner::solve_visible_target(
        geometry,
        eye,
        look_direction,
        std::numeric_limits<double>::infinity()
    );
}

std::uint64_t mix_checksum(std::uint64_t checksum, std::uint64_t value) {
    checksum ^= value;
    checksum *= 1099511628211ULL;
    return checksum;
}

std::uint64_t double_bits(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void update_checksum(SolveStats &stats, const BranchBoundResult &result) {
    stats.checksum = mix_checksum(stats.checksum, result.found ? 1 : 0);
    stats.checksum = mix_checksum(stats.checksum, result.target_world_face_index);
    stats.checksum = mix_checksum(stats.checksum, double_bits(result.angle));
    stats.checksum = mix_checksum(stats.checksum, double_bits(result.direction.x));
    stats.checksum = mix_checksum(stats.checksum, double_bits(result.direction.y));
    stats.checksum = mix_checksum(stats.checksum, double_bits(result.direction.z));
}

bool same_result(
    const BranchBoundResult &reference,
    const BranchBoundResult &candidate
) {
    if (reference.found != candidate.found) {
        return false;
    }
    if (!reference.found) {
        return true;
    }
    constexpr double epsilon = 1e-12;
    return reference.target_world_face_index == candidate.target_world_face_index &&
           std::abs(reference.angle - candidate.angle) <= epsilon &&
           std::abs(reference.direction.x - candidate.direction.x) <= epsilon &&
           std::abs(reference.direction.y - candidate.direction.y) <= epsilon &&
           std::abs(reference.direction.z - candidate.direction.z) <= epsilon;
}

void assert_result_invariants(
    const ScanRegionGeometry &geometry,
    const BranchBoundResult &result
) {
    if (!result.found) {
        return;
    }
    assert(std::isfinite(result.angle));
    assert(std::isfinite(result.distance));
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

template <typename Function>
BranchBoundResult time_solve(Function &&function, std::uint64_t &elapsed_us) {
    const auto start = std::chrono::steady_clock::now();
    BranchBoundResult result = function();
    const auto end = std::chrono::steady_clock::now();
    elapsed_us += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    );
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    using namespace minescript_miner;
    const RandomConfig config = parse_config(argc, argv);
    static_assert(kSide == 39);
    static_assert(static_cast<std::size_t>(kSide) * kSide * kSide <= 65535);

    SolveStats stats{};
    for (std::uint32_t case_index = 0; case_index < config.cases; ++case_index) {
        const CaseData data = make_case(config, case_index);
        const ScanRegionGeometry geometry = build_scan_region_geometry(
            data.shape_ids,
            data.target_indices,
            data.eye,
            data.look_direction,
            kSide,
            std::numeric_limits<double>::infinity()
        );

        std::uint64_t reference_us = 0;
        std::uint64_t candidate_us = 0;
        const BranchBoundResult reference = time_solve(
            [&]() {
                return solve_reference(geometry, data.eye, data.look_direction);
            },
            reference_us
        );
        const BranchBoundResult candidate = time_solve(
            [&]() {
                return solve_candidate(geometry, data.eye, data.look_direction);
            },
            candidate_us
        );

        assert_result_invariants(geometry, reference);
        assert_result_invariants(geometry, candidate);
        if (!same_result(reference, candidate)) {
            std::cerr
                << "random scan mismatch: seed=" << config.seed
                << " case=" << case_index
                << " density=" << config.density
                << " targets=" << config.target_count
                << " reference_found=" << reference.found
                << " candidate_found=" << candidate.found
                << '\n';
            return 1;
        }

        if (reference.found) {
            ++stats.found;
        } else {
            ++stats.missing;
        }
        stats.world_faces += geometry.world_faces.size();
        stats.target_faces += geometry.target_faces.size();
        stats.branches += reference.stats.branches_visited;
        stats.clips += reference.stats.clips_performed;
        stats.reference_us += reference_us;
        stats.candidate_us += candidate_us;
        update_checksum(stats, reference);
    }

    std::cout
        << "random_scan"
        << " seed=" << config.seed
        << " cases=" << config.cases
        << " side=" << kSide
        << " density=" << std::setprecision(4) << config.density
        << " targets=" << config.target_count
        << " found=" << stats.found
        << " missing=" << stats.missing
        << " world_faces=" << stats.world_faces
        << " target_faces=" << stats.target_faces
        << " branches=" << stats.branches
        << " clips=" << stats.clips
        << " reference_us=" << stats.reference_us
        << " candidate_us=" << stats.candidate_us
        << " checksum=" << stats.checksum
        << '\n';
}
