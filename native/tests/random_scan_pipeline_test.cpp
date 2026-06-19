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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using minescript_miner::BranchBoundResult;
using minescript_miner::BlockPos;
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
    double density = 0.14;
    std::uint16_t target_count = 5;
    std::string dump_dir{};
};

struct CaseData {
    std::vector<std::uint16_t> shape_ids{};
    std::vector<std::uint16_t> target_indices{};
    Vec3 eye{0.5, 0.5, 0.5};
    Vec3 look_direction{0.0, 0.0, 1.0};
};

struct SolveStats {
    std::uint64_t both_found = 0;
    std::uint64_t both_missing = 0;
    std::uint64_t reference_only = 0;
    std::uint64_t candidate_only = 0;
    std::uint64_t world_faces = 0;
    std::uint64_t target_faces = 0;
    std::uint64_t branches = 0;
    std::uint64_t clips = 0;
    std::uint64_t reference_us = 0;
    std::uint64_t candidate_us = 0;
    std::uint64_t checksum = 1469598103934665603ULL;
};

struct RayHit {
    bool hit = false;
    std::uint16_t block_index = 0;
    double distance = std::numeric_limits<double>::infinity();
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
        config.dump_dir = argv[5];
    }
    if (argc > 6) {
        throw std::invalid_argument(
            "usage: random_scan_pipeline_test [seed] [cases] [density] [targets] [dump_dir]"
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

std::string join_path(const std::string &directory, const std::string &filename) {
    if (directory.empty()) {
        return filename;
    }
    const char last = directory[directory.size() - 1];
    if (last == '/' || last == '\\') {
        return directory + filename;
    }
    return directory + "/" + filename;
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

double axis_interval_entry(
    double origin,
    double direction,
    double minimum,
    double maximum,
    double &exit
) {
    if (direction == 0.0) {
        if (origin < minimum || origin > maximum) {
            exit = -std::numeric_limits<double>::infinity();
            return std::numeric_limits<double>::infinity();
        }
        exit = std::numeric_limits<double>::infinity();
        return -std::numeric_limits<double>::infinity();
    }
    double first = (minimum - origin) / direction;
    double second = (maximum - origin) / direction;
    if (first > second) {
        std::swap(first, second);
    }
    exit = second;
    return first;
}

bool ray_intersects_unit_block(
    const Vec3 &origin,
    const Vec3 &direction,
    BlockPos block_pos,
    double &distance
) {
    double x_exit = 0.0;
    double y_exit = 0.0;
    double z_exit = 0.0;
    const double x_entry = axis_interval_entry(
        origin.x,
        direction.x,
        static_cast<double>(block_pos.x),
        static_cast<double>(block_pos.x + 1),
        x_exit
    );
    const double y_entry = axis_interval_entry(
        origin.y,
        direction.y,
        static_cast<double>(block_pos.y),
        static_cast<double>(block_pos.y + 1),
        y_exit
    );
    const double z_entry = axis_interval_entry(
        origin.z,
        direction.z,
        static_cast<double>(block_pos.z),
        static_cast<double>(block_pos.z + 1),
        z_exit
    );
    const double entry = std::max({x_entry, y_entry, z_entry, 0.0});
    const double exit = std::min({x_exit, y_exit, z_exit});
    if (entry > exit || exit < 0.0) {
        return false;
    }
    distance = entry;
    return true;
}

RayHit first_full_cube_hit(
    const std::vector<std::uint16_t> &shape_ids,
    const Vec3 &eye,
    const Vec3 &direction
) {
    constexpr BlockPos center{0, 0, 0};
    RayHit best{};
    for (std::size_t index = 0; index < shape_ids.size(); ++index) {
        if (shape_ids[index] != minescript_miner::SHAPE_FULL_CUBE) {
            continue;
        }
        const auto block_index = static_cast<std::uint16_t>(index);
        const BlockPos block_pos =
            minescript_miner::index_to_block_pos(block_index, kSide, center);
        double distance = 0.0;
        if (!ray_intersects_unit_block(eye, direction, block_pos, distance)) {
            continue;
        }
        constexpr double epsilon = 1e-10;
        if (!best.hit || distance + epsilon < best.distance) {
            best = {true, block_index, distance};
        }
    }
    return best;
}

Vec3 target_sample_point(std::uint16_t target_index, std::uint8_t sample_index) {
    constexpr BlockPos center{0, 0, 0};
    const BlockPos block_pos =
        minescript_miner::index_to_block_pos(target_index, kSide, center);
    switch (sample_index) {
        case 0:
            return {
                static_cast<double>(block_pos.x) + 0.5,
                static_cast<double>(block_pos.y) + 0.5,
                static_cast<double>(block_pos.z) + 0.5,
            };
        case 1:
            return {
                static_cast<double>(block_pos.x) + 0.5,
                static_cast<double>(block_pos.y),
                static_cast<double>(block_pos.z) + 0.5,
            };
        case 2:
            return {
                static_cast<double>(block_pos.x) + 0.5,
                static_cast<double>(block_pos.y) + 1.0,
                static_cast<double>(block_pos.z) + 0.5,
            };
        case 3:
            return {
                static_cast<double>(block_pos.x) + 0.5,
                static_cast<double>(block_pos.y) + 0.5,
                static_cast<double>(block_pos.z),
            };
        case 4:
            return {
                static_cast<double>(block_pos.x) + 0.5,
                static_cast<double>(block_pos.y) + 0.5,
                static_cast<double>(block_pos.z) + 1.0,
            };
        case 5:
            return {
                static_cast<double>(block_pos.x),
                static_cast<double>(block_pos.y) + 0.5,
                static_cast<double>(block_pos.z) + 0.5,
            };
        default:
            return {
                static_cast<double>(block_pos.x) + 1.0,
                static_cast<double>(block_pos.y) + 0.5,
                static_cast<double>(block_pos.z) + 0.5,
            };
    }
}

BranchBoundResult solve_candidate(
    const CaseData &data,
    const Vec3 &eye,
    const Vec3 &look_direction
) {
    BranchBoundResult best{};
    for (const std::uint16_t target_index : data.target_indices) {
        for (std::uint8_t sample_index = 0; sample_index < 7; ++sample_index) {
            const Vec3 sample = target_sample_point(target_index, sample_index);
            Vec3 direction = sample - eye;
            const double distance = std::sqrt(length_squared(direction));
            if (distance <= 0.0) {
                continue;
            }
            direction = direction * (1.0 / distance);
            const RayHit hit = first_full_cube_hit(
                data.shape_ids,
                eye,
                direction
            );
            if (!hit.hit || hit.block_index != target_index) {
                continue;
            }
            const double angle = minescript_miner::angle_to_point(
                look_direction,
                direction
            );
            if (!best.found || angle < best.angle) {
                best.found = true;
                best.direction = direction;
                best.angle = angle;
                best.distance = hit.distance;
                best.target_world_face_index =
                    std::numeric_limits<std::uint32_t>::max();
            }
        }
    }
    return best;
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

bool is_target_index(const CaseData &data, std::uint16_t index) {
    return std::find(
        data.target_indices.begin(),
        data.target_indices.end(),
        index
    ) != data.target_indices.end();
}

std::string candidate_only_basename(
    const RandomConfig &config,
    std::uint32_t case_index
) {
    return "candidate_only_seed" + std::to_string(config.seed) +
           "_case" + std::to_string(case_index) +
           "_density" + std::to_string(static_cast<int>(config.density * 10000.0)) +
           "_targets" + std::to_string(config.target_count);
}

void dump_candidate_only_scan(
    const RandomConfig &config,
    std::uint32_t case_index,
    const CaseData &data,
    const BranchBoundResult &candidate
) {
    if (config.dump_dir.empty()) {
        return;
    }

    const std::string basename = candidate_only_basename(config, case_index);
    const std::string scan_path = join_path(config.dump_dir, basename + ".scan");
    std::ofstream scan(scan_path);
    if (!scan) {
        throw std::runtime_error("cannot write candidate-only scan: " + scan_path);
    }

    const auto orientation =
        minescript_miner::yaw_pitch_from_direction(data.look_direction);
    scan
        << "# Uniform random candidate-only case.\n"
        << "# Reproduce with seed=" << config.seed
        << " case=" << case_index
        << " density=" << config.density
        << " targets=" << config.target_count << "\n"
        << "# Candidate angle=" << candidate.angle
        << " distance=" << candidate.distance << "\n"
        << "# Cube center is the command/source origin. Index order is x fastest, then z, then y.\n"
        << "fixture_version 1\n"
        << "shape_catalog_version " << minescript_miner::GEOMETRY_SHAPE_CATALOG_VERSION << "\n"
        << "side " << kSide << "\n"
        << "position " << data.eye.x << ' ' << data.eye.y << ' ' << data.eye.z << "\n"
        << "orientation_yaw_pitch " << orientation.yaw << ' ' << orientation.pitch << "\n"
        << "default_shape " << minescript_miner::SHAPE_EMPTY << "\n\n"
        << "expect_found 0\n\n";

    for (std::size_t index = 0; index < data.shape_ids.size(); ++index) {
        if (data.shape_ids[index] != minescript_miner::SHAPE_FULL_CUBE) {
            continue;
        }
        const auto block_index = static_cast<std::uint16_t>(index);
        scan
            << (is_target_index(data, block_index) ? "target " : "block ")
            << block_index << ' ' << minescript_miner::SHAPE_FULL_CUBE << '\n';
    }

    const std::string mcfunction_path =
        join_path(config.dump_dir, basename + ".mcfunction");
    std::ofstream function(mcfunction_path);
    if (!function) {
        throw std::runtime_error(
            "cannot write candidate-only mcfunction: " + mcfunction_path
        );
    }
    function
        << "# Execute at the intended cube center. Full cubes are stone; targets are gold_block.\n"
        << "fill ~-19 ~-19 ~-19 ~19 ~19 ~19 minecraft:air\n";

    constexpr BlockPos center{0, 0, 0};
    for (std::size_t index = 0; index < data.shape_ids.size(); ++index) {
        if (data.shape_ids[index] != minescript_miner::SHAPE_FULL_CUBE) {
            continue;
        }
        const auto block_index = static_cast<std::uint16_t>(index);
        const BlockPos block_pos =
            minescript_miner::index_to_block_pos(block_index, kSide, center);
        function
            << "setblock ~" << block_pos.x
            << " ~" << block_pos.y
            << " ~" << block_pos.z
            << (is_target_index(data, block_index)
                    ? " minecraft:gold_block\n"
                    : " minecraft:stone\n");
    }

    std::cerr
        << "dumped candidate_only case: " << scan_path
        << " and " << mcfunction_path << '\n';
}

void assert_result_invariants(
    const ScanRegionGeometry &geometry,
    const BranchBoundResult &result,
    bool require_target_face
) {
    if (!result.found) {
        return;
    }
    assert(std::isfinite(result.angle));
    assert(std::isfinite(result.distance));
    assert(std::abs(length_squared(result.direction) - 1.0) < 1e-12);
    if (require_target_face) {
        const bool result_is_target_face = std::any_of(
            geometry.target_faces.begin(),
            geometry.target_faces.end(),
            [&result](const TargetFaceCandidate &target) {
                return target.world_face_index == result.target_world_face_index;
            }
        );
        assert(result_is_target_face);
    }
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
                return solve_candidate(data, data.eye, data.look_direction);
            },
            candidate_us
        );

        assert_result_invariants(geometry, reference, true);
        assert_result_invariants(geometry, candidate, false);

        if (reference.found && candidate.found) {
            ++stats.both_found;
        } else if (!reference.found && !candidate.found) {
            ++stats.both_missing;
        } else if (reference.found) {
            ++stats.reference_only;
        } else {
            ++stats.candidate_only;
            dump_candidate_only_scan(config, case_index, data, candidate);
        }

        stats.world_faces += geometry.world_faces.size();
        stats.target_faces += geometry.target_faces.size();
        stats.branches += reference.stats.branches_visited;
        stats.clips += reference.stats.clips_performed;
        stats.reference_us += reference_us;
        stats.candidate_us += candidate_us;
        update_checksum(stats, reference);
        update_checksum(stats, candidate);
    }

    std::cout
        << "random_scan"
        << " seed=" << config.seed
        << " cases=" << config.cases
        << " side=" << kSide
        << " density=" << std::setprecision(4) << config.density
        << " targets=" << config.target_count
        << " both_found=" << stats.both_found
        << " both_missing=" << stats.both_missing
        << " reference_only=" << stats.reference_only
        << " candidate_only=" << stats.candidate_only
        << " world_faces=" << stats.world_faces
        << " target_faces=" << stats.target_faces
        << " branches=" << stats.branches
        << " clips=" << stats.clips
        << " reference_us=" << stats.reference_us
        << " candidate_us=" << stats.candidate_us
        << " checksum=" << stats.checksum
        << '\n';
}
