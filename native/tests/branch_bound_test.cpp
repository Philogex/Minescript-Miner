#include "minecraft_miner/scanner/branch_bound.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace {

constexpr std::int32_t from_sixteenths(std::int32_t value) {
    // TODO: Rename this legacy test helper. The production geometry uses a
    // 32-unit grid; these regression literals are still written in historical
    // sixteenth-style coordinates and are scaled here to keep the cases stable.
    static_assert(
        minecraft_miner::GEOMETRY_UNITS_PER_BLOCK % 16 == 0
    );
    return value * (minecraft_miner::GEOMETRY_UNITS_PER_BLOCK / 16);
}

minecraft_miner::WorldRectFace z_face(
    std::int32_t min_x,
    std::int32_t min_y,
    std::int32_t max_x,
    std::int32_t max_y,
    std::int32_t z
) {
    using namespace minecraft_miner;
    const WorldRectFace face{
        PlaneAxis::Z,
        -1,
        from_sixteenths(z),
        from_sixteenths(min_x),
        from_sixteenths(max_x),
        from_sixteenths(min_y),
        from_sixteenths(max_y),
    };
    return face;
}

minecraft_miner::ScanRegionGeometry target_with_occluder(
    bool full_occluder
) {
    minecraft_miner::ScanRegionGeometry geometry{};
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
    using namespace minecraft_miner;

    const BlockPos negative_block{-2, -3, -4};
    const LocalRectFace local_faces[]{
        {PlaneAxis::X, 0, 0, 32, 0, 32, -1},
        {PlaneAxis::X, 32, 0, 32, 0, 32, 1},
        {PlaneAxis::Y, 0, 0, 32, 0, 32, -1},
        {PlaneAxis::Y, 32, 0, 32, 0, 32, 1},
        {PlaneAxis::Z, 0, 0, 32, 0, 32, -1},
        {PlaneAxis::Z, 32, 0, 32, 0, 32, 1},
    };
    for (const LocalRectFace &local_face : local_faces) {
        const BlockPos owner = owning_block(
            face_to_world(local_face, negative_block)
        );
        assert(owner.x == negative_block.x);
        assert(owner.y == negative_block.y);
        assert(owner.z == negative_block.z);
    }

    ScanRegionGeometry free_geometry{};
    free_geometry.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    const BranchBoundResult free_result =
        solve_visible_target_face(
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

    free_geometry.target_faces.push_back({0, 0.0});
    const VisibleTargetRegionResult full_free_result =
        solve_full_visible_target(
            free_geometry,
            {},
            {0.0, 0.0, 1.0}
        );
    const double full_free_width = effective_target_width_degrees(
        full_free_result,
        {0.0, 0.0, 1.0}
    );
    const double expected_full_free_width =
        2.0 * std::atan(0.25) * 180.0 /
        3.141592653589793238462643383279502884;
    assert(
        std::abs(full_free_width - expected_full_free_width) < 1.0e-12
    );

    const BranchBoundResult hidden_result =
        solve_visible_target_face(
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
        solve_visible_target_face(
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

    ScanRegionGeometry full_partial = target_with_occluder(false);
    full_partial.target_faces.push_back({0, 0.0});
    const VisibleTargetRegionResult full_partial_result =
        solve_full_visible_target(
            full_partial,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(full_partial_result.target.found);
    assert(full_partial_result.components.size() == 4);
    const double full_partial_width = effective_target_width_degrees(
        full_partial_result,
        {0.0, 0.0, 1.0}
    );
    assert(full_partial_width > 0.0);
    assert(full_partial_width < full_free_width);
    for (const VisibleRegionComponent &component :
         full_partial_result.components) {
        assert(component.target_world_face_index == 0);
        assert(component.boundary_directions.size() >= 3);
        for (const Vec3 direction : component.boundary_directions) {
            assert(std::abs(length_squared(direction) - 1.0) < 1e-12);
        }
    }

    ScanRegionGeometry multi_face_target{};
    const BlockPos origin_block{};
    multi_face_target.world_faces = {
        face_to_world(local_faces[0], origin_block),
        face_to_world(local_faces[3], origin_block),
        face_to_world(local_faces[4], origin_block),
    };
    multi_face_target.target_faces = {
        {0, 0.0},
        {1, 0.0},
        {2, 0.0},
    };
    const Vec3 corner_eye{-2.0, 2.0, -2.0};
    Vec3 corner_look = Vec3{0.5, 0.5, 0.5} - corner_eye;
    corner_look = corner_look * (
        1.0 / std::sqrt(length_squared(corner_look))
    );
    const VisibleTargetRegionResult multi_face_result =
        solve_full_visible_target(
            multi_face_target,
            corner_eye,
            corner_look
        );
    assert(multi_face_result.target.found);
    assert(multi_face_result.target_block.x == 0);
    assert(multi_face_result.target_block.y == 0);
    assert(multi_face_result.target_block.z == 0);
    assert(multi_face_result.components.size() == 3);

    const BranchBoundResult stable_partial =
        solve_visible_target_face(
            target_with_occluder(false),
            0,
            {},
            partial_result.direction
        );
    assert(stable_partial.found);
    assert(stable_partial.angle < 1e-12);

    Vec3 visible_side_look{0.2, 0.0, 1.0};
    visible_side_look = visible_side_look * (
        1.0 / std::sqrt(length_squared(visible_side_look))
    );
    const BranchBoundResult pruned_partial =
        solve_visible_target_face(
            target_with_occluder(false),
            0,
            {},
            visible_side_look
        );
    assert(pruned_partial.found);
    assert(pruned_partial.angle < 1e-12);
    assert(pruned_partial.stats.branches_pruned > 0);

    ScanRegionGeometry far_occluder{};
    far_occluder.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    far_occluder.world_faces.push_back(
        z_face(-32, -32, 32, 32, 96)
    );
    const BranchBoundResult far_result =
        solve_visible_target_face(
            far_occluder,
            0,
            {},
            {0.0, 0.0, 1.0}
    );
    assert(far_result.found);
    assert(far_result.angle == 0.0);
    assert(far_result.stats.occluders_prepared == 0);
    assert(far_result.stats.clips_performed == 0);

    ScanRegionGeometry touching_edge{};
    touching_edge.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    touching_edge.world_faces.push_back(
        z_face(8, -16, 24, 16, 32)
    );
    const BranchBoundResult touching_result =
        solve_visible_target_face(
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
        solve_visible_target_face(
            thin_sliver,
            0,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(thin_result.found);
    assert(thin_result.projected_point.x > 15.0 / 1584.0);
    assert(thin_result.projected_point.x < 16.0 / 1600.0);
    assert(thin_result.stats.clips_performed == 1);

    ScanRegionGeometry repeated_occluders{};
    repeated_occluders.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    for (int i = 0; i < 8; ++i) {
        repeated_occluders.world_faces.push_back(
            z_face(-4, -4, 4, 4, 32)
        );
    }
    const BranchBoundResult repeated_result =
        solve_visible_target_face(
            repeated_occluders,
            0,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(repeated_result.found);
    assert(repeated_result.stats.clips_performed == 1);
    assert(repeated_result.stats.branches_visited <= 5);

    ScanRegionGeometry reachable_edge{};
    reachable_edge.world_faces.push_back(
        z_face(0, -8, 16, 8, 76)
    );
    reachable_edge.target_faces.push_back({0, 0.0});
    Vec3 reach_look{1.0, 0.0, 4.75};
    reach_look = reach_look * (
        1.0 / std::sqrt(length_squared(reach_look))
    );
    const BranchBoundResult reach_result =
        solve_visible_target(
            reachable_edge,
            {},
            reach_look,
            4.8
        );
    assert(reach_result.found);
    assert(reach_result.distance <= 4.8);

    ScanRegionGeometry target_loop{};
    target_loop.world_faces.push_back(
        z_face(-16, -16, 16, 16, 64)
    );
    target_loop.world_faces.push_back(
        z_face(-16, -16, 16, 16, 32)
    );
    target_loop.world_faces.push_back(
        z_face(48, -16, 80, 16, 64)
    );
    target_loop.target_faces.push_back({0, 0.0});
    target_loop.target_faces.push_back({2, 0.75});
    const BranchBoundResult loop_result =
        solve_visible_target(
            target_loop,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(loop_result.found);
    assert(loop_result.target_world_face_index == 2);
    assert(loop_result.stats.target_faces_considered == 2);

    const BranchBoundResult invalid_target =
        solve_visible_target_face(
            free_geometry,
            12,
            {},
            {0.0, 0.0, 1.0}
        );
    assert(!invalid_target.found);
}
