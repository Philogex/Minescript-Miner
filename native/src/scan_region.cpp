#include "minescript_miner/scan_region.hpp"

namespace minescript_miner {

static_assert(index_to_offset(0, 3).x == 0);
static_assert(index_to_offset(0, 3).y == 0);
static_assert(index_to_offset(0, 3).z == 0);

static_assert(index_to_offset(1, 3).x == 1);
static_assert(index_to_offset(3, 3).z == 1);
static_assert(index_to_offset(9, 3).y == 1);

static_assert(offset_to_index({2, 1, 2}, 3) == 17);
static_assert(index_to_block_pos(0, 3, {10, 64, -4}).x == 9);
static_assert(index_to_block_pos(0, 3, {10, 64, -4}).y == 63);
static_assert(index_to_block_pos(0, 3, {10, 64, -4}).z == -5);
static_assert(index_to_block_pos(26, 3, {10, 64, -4}).x == 11);
static_assert(index_to_block_pos(26, 3, {10, 64, -4}).y == 65);
static_assert(index_to_block_pos(26, 3, {10, 64, -4}).z == -3);

constexpr RectFace16 X_FACE{PlaneAxis::X, 8, 2, 6, 3, 7, 1};
constexpr WorldRectFace16 X_WORLD = face_to_world(X_FACE, {10, 64, -4});
static_assert(X_WORLD.p0.x == 168);
static_assert(X_WORLD.p0.y == 1026);
static_assert(X_WORLD.p0.z == -61);
static_assert(X_WORLD.p2.x == 168);
static_assert(X_WORLD.p2.y == 1030);
static_assert(X_WORLD.p2.z == -57);

constexpr RectFace16 Y_FACE{PlaneAxis::Y, 8, 2, 6, 3, 7, 1};
constexpr WorldRectFace16 Y_WORLD = face_to_world(Y_FACE, {10, 64, -4});
static_assert(Y_WORLD.p0.x == 162);
static_assert(Y_WORLD.p0.y == 1032);
static_assert(Y_WORLD.p0.z == -61);
static_assert(Y_WORLD.p2.x == 166);
static_assert(Y_WORLD.p2.y == 1032);
static_assert(Y_WORLD.p2.z == -57);

constexpr RectFace16 Z_FACE{PlaneAxis::Z, 8, 2, 6, 3, 7, 1};
constexpr WorldRectFace16 Z_WORLD = face_to_world(Z_FACE, {10, 64, -4});
static_assert(Z_WORLD.p0.x == 162);
static_assert(Z_WORLD.p0.y == 1027);
static_assert(Z_WORLD.p0.z == -56);
static_assert(Z_WORLD.p2.x == 166);
static_assert(Z_WORLD.p2.y == 1031);
static_assert(Z_WORLD.p2.z == -56);

}  // namespace minescript_miner
