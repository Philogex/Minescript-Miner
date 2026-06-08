#include "minescript_miner/exact_branch_bound.hpp"

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
    using namespace minescript_miner;
    const WorldRectFace16 face{
        PlaneAxis::Z,
        -1,
        {min_x, min_y, z},
        {max_x, min_y, z},
        {max_x, max_y, z},
        {min_x, max_y, z},
    };
    return {face, face_center(face)};
}

minescript_miner::ScanRegionGeometry target_with_occluder(
    bool full_occluder
) {
    minescript_miner::ScanRegionGeometry geometry{};
    geometry.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    geometry.world_faces.push_back(
        full_occluder
            ? z_face(-16, -16, 16, 16, 32)
            : z_face(-4, -4, 4, 4, 32)
    );
    return geometry;
}

}  // namespace

int main() {
    using namespace minescript_miner;

    ScanRegionGeometry free_geometry{};
    free_geometry.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    const BranchBoundResult free_result =
        solve_visible_target_face_exact(
            free_geometry,
            0,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(free_result.found);
    assert(free_result.target_world_face_index == 0);
    assert(free_result.angle == 0.0);
    assert(std::abs(free_result.distance - 4.0) < 1e-12);
    assert(free_result.stats.target_faces_considered == 1);
    assert(free_result.stats.occluders_prepared == 0);
    assert(free_result.stats.clips_performed == 0);

    const BranchBoundResult hidden_result =
        solve_visible_target_face_exact(
            target_with_occluder(true),
            0,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(!hidden_result.found);
    assert(hidden_result.stats.occluders_prepared == 1);
    assert(hidden_result.stats.effective_occluders == 1);
    assert(hidden_result.stats.clips_performed == 1);

    const BranchBoundResult partial_result =
        solve_visible_target_face_exact(
            target_with_occluder(false),
            0,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(partial_result.found);
    assert(partial_result.angle > 0.1);
    assert(partial_result.angle < 0.25);
    assert(
        std::max(
            std::abs(partial_result.projected_point.x),
            std::abs(partial_result.projected_point.y)
        ) > 0.125
    );
    assert(partial_result.stats.occluders_prepared == 1);
    assert(partial_result.stats.clips_performed == 1);

    const BranchBoundResult stable_partial =
        solve_visible_target_face_exact(
            target_with_occluder(false),
            0,
            {},
            partial_result.direction
        );
    assert(stable_partial.found);
    assert(stable_partial.angle < 1e-12);

    ScanRegionGeometry far_occluder{};
    far_occluder.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    far_occluder.world_faces.push_back(
        z_face(-32, -32, 32, 32, 96)
    );
    const BranchBoundResult far_result =
        solve_visible_target_face_exact(
            far_occluder,
            0,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(far_result.found);
    assert(far_result.angle == 0.0);
    assert(far_result.stats.occluders_prepared == 1);
    assert(far_result.stats.clips_performed == 0);

    ScanRegionGeometry touching_edge{};
    touching_edge.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    touching_edge.world_faces.push_back(
        z_face(8, -16, 24, 16, 32)
    );
    const BranchBoundResult touching_result =
        solve_visible_target_face_exact(
            touching_edge,
            0,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(touching_result.found);
    assert(touching_result.angle == 0.0);
    assert(touching_result.stats.clips_performed == 0);

    ScanRegionGeometry thin_sliver{};
    thin_sliver.world_faces.push_back(
        z_face(-16, -16, 16, 16, 1600)
    );
    thin_sliver.world_faces.push_back(
        z_face(-16, -16, 15, 16, 1584)
    );
    const BranchBoundResult thin_result =
        solve_visible_target_face_exact(
            thin_sliver,
            0,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(thin_result.found);
    assert(thin_result.projected_point.x > 15.0 / 1584.0);
    assert(thin_result.projected_point.x < 16.0 / 1600.0);
    assert(thin_result.stats.clips_performed == 1);

    const BranchBoundResult invalid_target =
        solve_visible_target_face_exact(
            free_geometry,
            12,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(!invalid_target.found);
}
