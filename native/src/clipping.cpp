#include "minescript_miner/clipping.hpp"

namespace minescript_miner {

static_assert(orient2d({0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}) > 0.0);
static_assert(orient2d({0.0, 0.0}, {1.0, 0.0}, {0.0, -1.0}) < 0.0);
static_assert(orient2d({0.0, 0.0}, {1.0, 0.0}, {0.5, 0.0}) == 0.0);

}  // namespace minescript_miner
