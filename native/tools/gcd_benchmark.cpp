#include "gcd_benchmark_support.hpp"

#include "minecraft_miner/aim/angle.hpp"
#include "minecraft_miner/scanner/branch_bound.hpp"
#include "minecraft_miner/catalog/geometry_catalog.hpp"
#include "minecraft_miner/scanner/scan_region.hpp"
#include "scan_fixture.hpp"

#include <boost/integer/common_factor_rt.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using GcdFunction = minecraft_miner::ExactInt (*)(
    minecraft_miner::ExactInt,
    minecraft_miner::ExactInt
);

struct ScanInput {
    std::vector<std::uint16_t> shape_ids{};
    std::vector<std::uint16_t> target_indices{};
};

struct TimingResult {
    double median_seconds = 0.0;
    std::uint64_t checksum = 0;
};

ScanInput make_scan_input(const minecraft_miner::test::ScanFixture &fixture) {
    using namespace minecraft_miner;

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
            throw std::runtime_error(
                "fixture block index is outside the scan cube"
            );
        }
        if (block.shape_id >= GEOMETRY_SHAPE_COUNT) {
            throw std::runtime_error(
                "fixture shape id is outside the catalog"
            );
        }
        input.shape_ids[block.index] = block.shape_id;
        if (block.target) {
            input.target_indices.push_back(block.index);
        }
    }
    return input;
}

std::size_t parse_positive_size(const char *value, const char *name) {
    const std::string text(value);
    std::size_t parsed = 0;
    const unsigned long long result = std::stoull(text, &parsed);
    if (parsed != text.size() || result == 0) {
        throw std::runtime_error(
            std::string(name) + " must be a positive integer"
        );
    }
    return static_cast<std::size_t>(result);
}

minecraft_miner::ExactInt boost_integer_gcd(
    minecraft_miner::ExactInt lhs,
    minecraft_miner::ExactInt rhs
) {
    return boost::integer::gcd(std::move(lhs), std::move(rhs));
}

std::uint64_t result_fragment(const minecraft_miner::ExactInt &value) {
    static const minecraft_miner::ExactInt mask =
        (minecraft_miner::ExactInt{1} << 64U) - 1;
    return (value & mask).convert_to<std::uint64_t>();
}

std::uint64_t run_once(
    const std::vector<minecraft_miner::gcd_benchmark::OperandPair> &samples,
    GcdFunction function
) {
    std::uint64_t checksum = 0;
    for (const auto &sample : samples) {
        const minecraft_miner::ExactInt result =
            function(sample.lhs, sample.rhs);
        checksum ^= result_fragment(result) +
                    0x9e3779b97f4a7c15ULL +
                    (checksum << 6U) +
                    (checksum >> 2U);
    }
    return checksum;
}

TimingResult benchmark(
    const std::vector<minecraft_miner::gcd_benchmark::OperandPair> &samples,
    GcdFunction function,
    std::size_t rounds
) {
    std::vector<double> durations;
    durations.reserve(rounds);
    std::uint64_t checksum = 0;
    for (std::size_t round = 0; round < rounds; ++round) {
        const auto start = Clock::now();
        checksum ^= run_once(samples, function);
        const auto end = Clock::now();
        durations.push_back(
            std::chrono::duration<double>(end - start).count()
        );
    }
    std::sort(durations.begin(), durations.end());
    return {durations[durations.size() / 2], checksum};
}

std::size_t bit_length(const minecraft_miner::ExactInt &value) {
    if (value == 0) {
        return 0;
    }
    const minecraft_miner::ExactInt magnitude =
        value < 0 ? -value : value;
    return boost::multiprecision::msb(magnitude) + 1;
}

void print_operand_statistics(
    const std::vector<minecraft_miner::gcd_benchmark::OperandPair> &samples
) {
    std::size_t minimum = std::numeric_limits<std::size_t>::max();
    std::size_t maximum = 0;
    long double total = 0.0;
    std::uint64_t zero_operands = 0;
    for (const auto &sample : samples) {
        for (const minecraft_miner::ExactInt *value :
             {&sample.lhs, &sample.rhs}) {
            const std::size_t bits = bit_length(*value);
            minimum = std::min(minimum, bits);
            maximum = std::max(maximum, bits);
            total += static_cast<long double>(bits);
            zero_operands += bits == 0;
        }
    }
    const std::size_t operand_count = samples.size() * 2;
    const double average = operand_count == 0
        ? 0.0
        : static_cast<double>(total / operand_count);
    if (minimum == std::numeric_limits<std::size_t>::max()) {
        minimum = 0;
    }
    std::cout
        << "operand_bits_min=" << minimum
        << " operand_bits_average=" << average
        << " operand_bits_max=" << maximum
        << " zero_operands=" << zero_operands
        << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    using namespace minecraft_miner;

    if (argc < 2 || argc > 4) {
        std::cerr
            << "usage: " << argv[0]
            << " FIXTURE [SAMPLES] [ROUNDS]\n";
        return 2;
    }

    try {
        const std::size_t sample_limit =
            argc >= 3 ? parse_positive_size(argv[2], "samples") : 100000;
        const std::size_t rounds =
            argc >= 4 ? parse_positive_size(argv[3], "rounds") : 5;
        const test::ScanFixture fixture = test::load_scan_fixture(argv[1]);
        const ScanInput input = make_scan_input(fixture);
        const Vec3 look_direction =
            look_direction_from_yaw_pitch(fixture.yaw, fixture.pitch);

        gcd_benchmark::start_collection(sample_limit);
        const ScanRegionGeometry geometry = build_scan_region_geometry(
            input.shape_ids,
            input.target_indices,
            fixture.position,
            look_direction,
            fixture.side,
            fixture.reach
        );
        const BranchBoundResult result = solve_visible_target(
            geometry,
            fixture.position,
            look_direction,
            fixture.reach
        );
        gcd_benchmark::stop_collection();

        if (fixture.has_expect_found && result.found != fixture.expect_found) {
            throw std::runtime_error("unexpected target visibility result");
        }
        const auto &samples = gcd_benchmark::recorded_operands;
        if (samples.empty()) {
            throw std::runtime_error("pipeline recorded no GCD operands");
        }

        for (const auto &sample : samples) {
            if (
                gcd_benchmark::legacy_integer_gcd(
                    sample.lhs,
                    sample.rhs
                ) != boost_integer_gcd(sample.lhs, sample.rhs)
            ) {
                throw std::runtime_error("GCD implementations disagree");
            }
        }

        (void) run_once(samples, gcd_benchmark::legacy_integer_gcd);
        (void) run_once(samples, boost_integer_gcd);

        std::vector<double> legacy_durations;
        std::vector<double> boost_durations;
        legacy_durations.reserve(rounds);
        boost_durations.reserve(rounds);
        std::uint64_t legacy_checksum = 0;
        std::uint64_t boost_checksum = 0;
        for (std::size_t round = 0; round < rounds; ++round) {
            const bool boost_first = (round & 1U) != 0;
            const TimingResult first = benchmark(
                samples,
                boost_first
                    ? boost_integer_gcd
                    : gcd_benchmark::legacy_integer_gcd,
                1
            );
            const TimingResult second = benchmark(
                samples,
                boost_first
                    ? gcd_benchmark::legacy_integer_gcd
                    : boost_integer_gcd,
                1
            );
            if (boost_first) {
                boost_durations.push_back(first.median_seconds);
                legacy_durations.push_back(second.median_seconds);
                boost_checksum ^= first.checksum;
                legacy_checksum ^= second.checksum;
            } else {
                legacy_durations.push_back(first.median_seconds);
                boost_durations.push_back(second.median_seconds);
                legacy_checksum ^= first.checksum;
                boost_checksum ^= second.checksum;
            }
        }
        std::sort(legacy_durations.begin(), legacy_durations.end());
        std::sort(boost_durations.begin(), boost_durations.end());
        const double legacy_seconds =
            legacy_durations[legacy_durations.size() / 2];
        const double boost_seconds =
            boost_durations[boost_durations.size() / 2];

        std::cout << std::fixed << std::setprecision(6)
                  << "observed_calls=" << gcd_benchmark::observed_operands
                  << " sampled_calls=" << samples.size()
                  << " rounds=" << rounds
                  << " world_faces=" << geometry.world_faces.size()
                  << " branches=" << result.stats.branches_visited
                  << " clips=" << result.stats.clips_performed
                  << '\n';
        print_operand_statistics(samples);
        std::cout
            << "legacy_seconds=" << legacy_seconds
            << " legacy_ns_per_gcd="
            << legacy_seconds * 1e9 / samples.size()
            << " legacy_checksum=" << legacy_checksum
            << '\n'
            << "boost_seconds=" << boost_seconds
            << " boost_ns_per_gcd="
            << boost_seconds * 1e9 / samples.size()
            << " boost_checksum=" << boost_checksum
            << '\n'
            << "boost_speedup=" << legacy_seconds / boost_seconds
            << '\n';
    } catch (const std::exception &error) {
        gcd_benchmark::stop_collection();
        std::cerr << "gcd_benchmark: " << error.what() << '\n';
        return 1;
    }
}
