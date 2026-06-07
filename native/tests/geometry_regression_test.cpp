#include "minescript_miner/branch_bound.hpp"
#include "minescript_miner/clipping.hpp"
#include "minescript_miner/visibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using minescript_miner::BranchBoundResult;
using minescript_miner::PlaneAxis;
using minescript_miner::Point2;
using minescript_miner::ScanRegionGeometry;
using minescript_miner::Vec3;
using minescript_miner::ViewBasis;
using minescript_miner::WorldFace;
using minescript_miner::WorldRectFace16;

WorldFace z_face(
    std::int32_t min_x,
    std::int32_t min_y,
    std::int32_t max_x,
    std::int32_t max_y,
    std::int32_t z
) {
    const WorldRectFace16 face{
        PlaneAxis::Z,
        -1,
        {min_x, min_y, z},
        {max_x, min_y, z},
        {max_x, max_y, z},
        {min_x, max_y, z},
    };
    return {face, minescript_miner::face_center(face)};
}

bool robust_orientation_regression() {
    const Point2 a{-0.1825141931108063, 0.30808915383215085};
    const Point2 b{-0.557948852352874, -0.6988892259339703};
    const Point2 c{-0.44282367504482284, -0.39010429715305583};

    // The exact determinant of these represented doubles is negative.
    const minescript_miner::Orientation orientation =
        minescript_miner::orient2d(a, b, c);
    if (orientation != minescript_miner::Orientation::Clockwise) {
        std::cerr
            << "orient2d orientation="
            << static_cast<int>(orientation)
            << '\n';
        return false;
    }
    return true;
}

bool near_plane_regression() {
    const Vec3 eye{};
    const ViewBasis basis{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    const WorldRectFace16 crossing_face{
        PlaneAxis::X,
        -1,
        {8, -8, -16},
        {8, -8, 48},
        {8, 8, 48},
        {8, 8, -16},
    };

    minescript_miner::ProjectedFace projection{};
    const bool projected = minescript_miner::project_world_face(
        crossing_face,
        eye,
        basis,
        projection
    );
    if (!projected) {
        std::cerr << "crossing face was discarded\n";
        return false;
    }
    if (projection.count != 4) {
        std::cerr
            << "unexpected clipped point count="
            << static_cast<int>(projection.count)
            << '\n';
        return false;
    }
    for (std::uint8_t i = 0; i < projection.count; ++i) {
        if (projection.points[i].depth <
                minescript_miner::PROJECTION_NEAR_DEPTH ||
            !std::isfinite(projection.points[i].point.x) ||
            !std::isfinite(projection.points[i].point.y)) {
            std::cerr << "invalid projected near-plane point\n";
            return false;
        }
    }

    constexpr double inverse_sqrt_two = 0.7071067811865475244;
    const ViewBasis diagonal_basis{
        {inverse_sqrt_two, 0.0, -inverse_sqrt_two},
        {0.0, 1.0, 0.0},
        {inverse_sqrt_two, 0.0, inverse_sqrt_two},
    };
    const WorldRectFace16 one_corner_behind{
        PlaneAxis::Y,
        1,
        {-16, 0, -16},
        {16, 0, -16},
        {16, 0, 16},
        {-16, 0, 16},
    };
    const Vec3 diagonal_eye{-0.5, 0.5, -0.5};
    if (!minescript_miner::project_world_face(
            one_corner_behind,
            diagonal_eye,
            diagonal_basis,
            projection
        ) ||
        projection.count != 5) {
        std::cerr
            << "one-corner near clip point count="
            << static_cast<int>(projection.count)
            << '\n';
        return false;
    }
    return true;
}

bool reach_regression() {
    constexpr double reach = 4.8;
    ScanRegionGeometry geometry{};
    geometry.world_faces.push_back(z_face(0, -8, 16, 8, 76));
    geometry.target_faces.push_back({0, 0.0});

    Vec3 look_direction{1.0, 0.0, 4.75};
    look_direction = look_direction * (
        1.0 / std::sqrt(minescript_miner::length_squared(look_direction))
    );
    const BranchBoundResult result =
        minescript_miner::solve_visible_target(geometry, {}, look_direction);

    if (!result.found || result.distance > reach) {
        std::cerr
            << "found=" << result.found
            << " distance=" << result.distance
            << " reach=" << reach
            << '\n';
        return false;
    }
    return true;
}

bool thin_sliver_regression() {
    ScanRegionGeometry geometry{};
    geometry.world_faces.push_back(z_face(-16, -16, 16, 16, 1600));
    geometry.world_faces.push_back(z_face(-16, -16, 15, 16, 1584));
    geometry.target_faces.push_back({0, 0.0});

    const BranchBoundResult result =
        minescript_miner::solve_visible_target(
            geometry,
            {},
            {0.0, 0.0, 1.0}
        );
    if (!result.found) {
        return false;
    }

    const double occluder_right = 15.0 / 1584.0;
    const double target_right = 16.0 / 1600.0;
    const double visible_width = target_right - occluder_right;
    const double right_clearance = target_right - result.projected_point.x;
    const double left_clearance = result.projected_point.x - occluder_right;
    const double minimum_clearance = std::min(left_clearance, right_clearance);

    const double required_clearance = visible_width * 0.25;
    if (minimum_clearance < required_clearance) {
        std::cerr
            << "projected_x=" << result.projected_point.x
            << " minimum_clearance=" << minimum_clearance
            << " required_clearance=" << required_clearance
            << " visible_width=" << visible_width
            << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "expected one regression case name\n";
        return 2;
    }

    const std::string case_name = argv[1];
    bool passed = false;
    if (case_name == "robust_orientation") {
        passed = robust_orientation_regression();
    } else if (case_name == "near_plane") {
        passed = near_plane_regression();
    } else if (case_name == "reach") {
        passed = reach_regression();
    } else if (case_name == "thin_sliver") {
        passed = thin_sliver_regression();
    } else {
        std::cerr << "unknown regression case: " << case_name << '\n';
        return 2;
    }

    if (!passed) {
        std::cerr << "regression still present: " << case_name << '\n';
        return 1;
    }
    return 0;
}
