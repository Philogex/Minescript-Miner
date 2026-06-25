#include "minescript_miner/aim/angle.hpp"
#include "minescript_miner/scanner/branch_bound.hpp"
#include "minescript_miner/catalog/geometry_catalog.hpp"
#include "minescript_miner/scanner/scan_region.hpp"
#include "minescript_miner/scanner/target_solver.hpp"
#include "scan_fixture.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

int main(int argc, char **argv) {
    using namespace minescript_miner;
    using minescript_miner::test::ScanFixture;
    using minescript_miner::test::SparseBlock;

    assert(argc == 2);
    const Vec3 round_trip_direction =
        look_direction_from_yaw_pitch(-98.1, -13.2);
    const YawPitch round_trip_orientation =
        yaw_pitch_from_direction(round_trip_direction);
    assert(std::abs(round_trip_orientation.yaw - -98.1) < 1e-12);
    assert(std::abs(round_trip_orientation.pitch - -13.2) < 1e-12);

    std::vector<std::uint16_t> reach_shapes(5 * 5 * 5, SHAPE_EMPTY);
    const std::uint16_t reach_target_index = offset_to_index({2, 2, 4}, 5);
    reach_shapes[reach_target_index] = SHAPE_FULL_CUBE;
    const std::vector<std::uint16_t> reach_targets{reach_target_index};
    const Vec3 reach_eye{0.5, 0.5, 0.5};
    const Vec3 reach_look{0.0, 0.0, 1.0};
    const ScanRegionGeometry outside_reach = build_scan_region_geometry(
        reach_shapes,
        reach_targets,
        reach_eye,
        reach_look,
        5,
        1.49
    );
    assert(outside_reach.target_faces.empty());
    const ScanRegionGeometry touching_reach = build_scan_region_geometry(
        reach_shapes,
        reach_targets,
        reach_eye,
        reach_look,
        5,
        1.5
    );
    assert(touching_reach.target_faces.size() == 1);

    const ScanFixture fixture = test::load_scan_fixture(argv[1]);
    assert(fixture.fixture_version == 1);
    assert(fixture.shape_catalog_version == GEOMETRY_SHAPE_CATALOG_VERSION);
    assert(fixture.side > 0);
    assert(fixture.side <= 39);
    assert(fixture.default_shape < GEOMETRY_SHAPE_COUNT);

    const std::size_t block_count =
        static_cast<std::size_t>(fixture.side) *
        static_cast<std::size_t>(fixture.side) *
        static_cast<std::size_t>(fixture.side);
    assert(block_count <= 65535);

    std::vector<std::uint16_t> shape_ids(block_count, fixture.default_shape);
    std::vector<std::uint16_t> target_indices{};
    for (const SparseBlock &block : fixture.blocks) {
        assert(block.index < block_count);
        assert(block.shape_id < GEOMETRY_SHAPE_COUNT);
        shape_ids[block.index] = block.shape_id;
        if (block.target) {
            target_indices.push_back(block.index);
        }
    }
    const Vec3 look_direction =
        look_direction_from_yaw_pitch(fixture.yaw, fixture.pitch);
    const ScanRegionGeometry geometry = build_scan_region_geometry(
        shape_ids,
        target_indices,
        fixture.position,
        look_direction,
        fixture.side,
        fixture.reach
    );
    if (fixture.has_expect_target_faces) {
        assert(geometry.target_faces.size() == fixture.expect_target_faces);
    }

    const BranchBoundResult result =
        solve_visible_target(
            geometry,
            fixture.position,
            look_direction,
            fixture.reach
        );
    if (fixture.has_expect_world_faces) {
        assert(geometry.world_faces.size() <= fixture.expect_world_faces);
    }
    if (fixture.has_expect_found) {
        assert(result.found == fixture.expect_found);
    }
    if (fixture.has_expect_min_clips) {
        assert(result.stats.clips_performed >= fixture.expect_min_clips);
    }

    if (result.found) {
        assert(std::isfinite(result.angle));
        if (fixture.has_expect_angle_range) {
            assert(result.angle >= fixture.expect_angle_min);
            assert(result.angle <= fixture.expect_angle_max);
        }
        assert(std::abs(length_squared(result.direction) - 1.0) < 1e-12);

        const bool result_is_target_face = std::any_of(
            geometry.target_faces.begin(),
            geometry.target_faces.end(),
            [&result](const TargetFaceCandidate &target) {
                return target.world_face_index == result.target_world_face_index;
            }
        );
        assert(result_is_target_face);
    }

    std::cout
        << "world_faces=" << geometry.world_faces.size()
        << " target_faces=" << geometry.target_faces.size()
        << " found=" << result.found
        << " angle=" << result.angle
        << " targets_considered=" << result.stats.target_faces_considered
        << " targets_pruned=" << result.stats.target_faces_pruned
        << " occluders_prepared=" << result.stats.occluders_prepared
        << " effective_occluders=" << result.stats.effective_occluders
        << " branches=" << result.stats.branches_visited
        << " branches_pruned=" << result.stats.branches_pruned
        << " branches_memoized=" << result.stats.branches_memoized
        << " clips=" << result.stats.clips_performed
        << '\n';
}
