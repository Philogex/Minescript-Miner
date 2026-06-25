#pragma once

#include "minecraft_miner/geometry/geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minecraft_miner::gcd_benchmark {

struct OperandPair {
    ExactInt lhs{};
    ExactInt rhs{};
};

inline std::vector<OperandPair> recorded_operands{};
inline std::size_t sample_limit = 0;
inline std::uint64_t observed_operands = 0;
inline std::uint64_t random_state = 0x9e3779b97f4a7c15ULL;
inline bool collection_enabled = false;

inline std::uint64_t next_random() {
    random_state ^= random_state >> 12U;
    random_state ^= random_state << 25U;
    random_state ^= random_state >> 27U;
    return random_state * 0x2545f4914f6cdd1dULL;
}

inline void start_collection(std::size_t limit) {
    if (limit == 0) {
        throw std::invalid_argument("GCD sample limit must be positive");
    }
    recorded_operands.clear();
    recorded_operands.reserve(limit);
    sample_limit = limit;
    observed_operands = 0;
    random_state = 0x9e3779b97f4a7c15ULL;
    collection_enabled = true;
}

inline void stop_collection() {
    collection_enabled = false;
}

inline void record_operands(const ExactInt &lhs, const ExactInt &rhs) {
    if (!collection_enabled) {
        return;
    }

    ++observed_operands;
    if (recorded_operands.size() < sample_limit) {
        recorded_operands.push_back({lhs, rhs});
        return;
    }

    const std::uint64_t selected = next_random() % observed_operands;
    if (selected < sample_limit) {
        recorded_operands[static_cast<std::size_t>(selected)] = {lhs, rhs};
    }
}

ExactInt legacy_integer_gcd(ExactInt lhs, ExactInt rhs);

}  // namespace minecraft_miner::gcd_benchmark
