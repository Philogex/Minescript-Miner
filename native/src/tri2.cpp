#include "minescript_miner/tri2.hpp"

namespace minescript_miner {

static_assert(signed_area2(tri2(point2(0.0, 0.0), point2(1.0, 0.0), point2(0.0, 1.0))) > 0.0);
static_assert(rect_to_tris(0.0, 0.0, 1.0, 1.0)[0].a.x == 0.0);

}  // namespace minescript_miner
