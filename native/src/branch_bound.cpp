#include "minescript_miner/branch_bound.hpp"

namespace minescript_miner {

static_assert(target_branch(
                  tri2(point2(0.0, 0.0), point2(1.0, 0.0), point2(0.0, 1.0)),
                  7,
                  3,
                  0.25
              )
                  .target_index == 7);

}  // namespace minescript_miner
