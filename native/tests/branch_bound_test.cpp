#include "minescript_miner/branch_bound.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace {

minescript_miner::WorldFace z_face(
    std::int32_t min_x,
    std::int32_t min_y,
    std::int32_t max_x,
    std::int32_t max_y,
    std::int32_t z
) {
    const minescript_miner::WorldRectFace16 face{
        minescript_miner::PlaneAxis::Z,
        -1,
        {min_x, min_y, z},
        {max_x, min_y, z},
        {max_x, max_y, z},
        {min_x, max_y, z},
    };
    return {face, minescript_miner::face_center(face)};
}

minescript_miner::ScanRegionGeometry target_with_occluder(bool full_occluder) {
    minescript_miner::ScanRegionGeometry geometry{};
    geometry.world_faces.push_back(z_face(-16, -16, 16, 16, 64));
    if (full_occluder) {
        geometry.world_faces.push_back(z_face(-16, -16, 16, 16, 32));
    } else {
        geometry.world_faces.push_back(z_face(-4, -4, 4, 4, 32));
    }
    geometry.target_faces.push_back({0, 0.0});
    return geometry;
}

}  // namespace

int main() {
    using namespace minescript_miner;

    const TriangleAngleResult edge_bound = minimum_angle_to_triangle(
        {{1.0, -1.0}, {1.0, 1.0}, {2.0, 0.0}},
        {0.0, 0.0, 1.0}
    );
    assert(std::abs(edge_bound.point.x - 1.0) < 1e-12);
    assert(std::abs(edge_bound.point.y) < 1e-12);
    assert(std::abs(edge_bound.angle - std::atan(1.0)) < 1e-12);

    ScanRegionGeometry free_geometry{};
    free_geometry.world_faces.push_back(z_face(-16, -16, 16, 16, 64));
    free_geometry.target_faces.push_back({0, 0.0});
    const BranchBoundResult free_result =
        solve_visible_target(free_geometry, {}, {0.0, 0.0, 1.0});
    assert(free_result.found);
    assert(free_result.angle == 0.0);
    assert(std::abs(free_result.distance - 4.0) < 1e-12);
    assert(free_result.stats.clips_performed == 0);

    const BranchBoundResult hidden_result =
        solve_visible_target(target_with_occluder(true), {}, {0.0, 0.0, 1.0});
    assert(!hidden_result.found);
    assert(hidden_result.stats.clips_performed == 2);

    const BranchBoundResult partial_result =
        solve_visible_target(target_with_occluder(false), {}, {0.0, 0.0, 1.0});
    assert(partial_result.found);
    assert(partial_result.angle > 0.1);
    assert(partial_result.angle < 0.2);
    assert(std::max(
        std::abs(partial_result.projected_point.x),
        std::abs(partial_result.projected_point.y)
    ) > 0.125);
    assert(partial_result.stats.clips_performed >= 2);

    const BranchBoundResult stable_partial_result =
        solve_visible_target(
            target_with_occluder(false),
            {},
            partial_result.direction
        );
    assert(stable_partial_result.found);
    assert(stable_partial_result.angle < 1e-12);
    assert(length_squared(
        stable_partial_result.direction - partial_result.direction
    ) < 1e-24);

    ScanRegionGeometry two_targets{};
    two_targets.world_faces.push_back(z_face(-16, -16, 16, 16, 64));
    two_targets.world_faces.push_back(z_face(48, -16, 80, 16, 64));
    two_targets.target_faces.push_back({0, 0.0});
    two_targets.target_faces.push_back({1, 0.75});
    const BranchBoundResult early_result =
        solve_visible_target(two_targets, {}, {0.0, 0.0, 1.0});
    assert(early_result.found);
    assert(early_result.angle == 0.0);
    assert(early_result.target_world_face_index == 0);
    assert(early_result.stats.target_faces_considered == 1);
}
