#include "minecraft_miner/aim/angle.hpp"
#include "minecraft_miner/aim/geometry_feedback_sigmadrift.hpp"
#include "minecraft_miner/aim/target_region.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace {

using minecraft_miner::Point2;
using minecraft_miner::Vec3;
using minecraft_miner::aim::TargetProjection;
using minecraft_miner::aim::VisibleDirectionComponent;

Vec3 normalized(Vec3 value) {
    const double inverse_length =
        1.0 / std::sqrt(minecraft_miner::length_squared(value));
    return value * inverse_length;
}

Vec3 z_direction(double x, double y) {
    return normalized({x, y, 1.0});
}

Vec3 local_direction(
    const TargetProjection &projection,
    double x,
    double y
) {
    return normalized({
        projection.forward.x + projection.right.x * x + projection.up.x * y,
        projection.forward.y + projection.right.y * x + projection.up.y * y,
        projection.forward.z + projection.right.z * x + projection.up.z * y,
    });
}

VisibleDirectionComponent rectangle(
    double min_x,
    double min_y,
    double max_x,
    double max_y
) {
    return {
        z_direction(min_x, min_y),
        z_direction(max_x, min_y),
        z_direction(max_x, max_y),
        z_direction(min_x, max_y),
    };
}

}  // namespace

int main() {
    using namespace minecraft_miner::aim;

    TargetProjection identity{};
    assert(make_target_projection({0.0, 0.0, 1.0}, identity));
    assert(std::abs(identity.right.x - 1.0) < 1.0e-12);
    assert(std::abs(identity.up.y - 1.0) < 1.0e-12);
    assert(std::abs(identity.forward.z - 1.0) < 1.0e-12);

    Point2 projected{};
    assert(project_target_direction(identity, z_direction(0.25, -0.5), projected));
    assert(std::abs(projected.x - 0.25) < 1.0e-12);
    assert(std::abs(projected.y + 0.5) < 1.0e-12);
    assert(!project_target_direction(identity, {0.0, 0.0, -1.0}, projected));

    ProjectedTargetRegion square{};
    assert(project_visible_target_region(
        {0.0, 0.0, 1.0},
        {rectangle(-0.25, -0.25, 0.25, 0.25)},
        square
    ));
    assert(square.components.size() == 1);
    assert(point_in_visible_region(square, z_direction(0.0, 0.0)));
    assert(point_in_visible_region(square, z_direction(0.25, 0.0)));
    assert(!point_in_visible_region(square, z_direction(0.2501, 0.0)));
    assert(!point_in_visible_region(square, {0.0, 0.0, -1.0}));
    assert(point_in_visible_region_with_margin(
        square,
        z_direction(0.19, 0.0),
        0.05
    ));
    assert(!point_in_visible_region_with_margin(
        square,
        z_direction(0.21, 0.0),
        0.05
    ));
    Vec3 closest_safe{};
    assert(closest_safe_direction_in_visible_region(
        square,
        z_direction(0.4, 0.0),
        0.05,
        closest_safe
    ));
    assert(project_target_direction(identity, closest_safe, projected));
    assert(projected.x < 0.2);
    assert(projected.x > 0.19);
    assert(std::abs(projected.y) < 1.0e-12);

    VisibleDirectionComponent clockwise =
        rectangle(-0.25, -0.25, 0.25, 0.25);
    std::reverse(clockwise.begin(), clockwise.end());
    ProjectedTargetRegion reversed{};
    assert(project_visible_target_region(
        {0.0, 0.0, 1.0},
        {clockwise},
        reversed
    ));
    assert(point_in_visible_region(reversed, z_direction(0.0, 0.0)));
    assert(!point_in_visible_region(reversed, z_direction(0.3, 0.0)));

    ProjectedTargetRegion disconnected{};
    assert(project_visible_target_region(
        {0.0, 0.0, 1.0},
        {
            rectangle(-0.5, -0.2, -0.2, 0.2),
            rectangle(0.2, -0.2, 0.5, 0.2),
        },
        disconnected
    ));
    assert(point_in_visible_region(disconnected, z_direction(-0.3, 0.0)));
    assert(point_in_visible_region(disconnected, z_direction(0.3, 0.0)));
    assert(!point_in_visible_region(disconnected, z_direction(0.0, 0.0)));

    TargetProjection vertical_basis{};
    assert(make_target_projection({0.0, 1.0, 1.0e-14}, vertical_basis));
    VisibleDirectionComponent vertical_component{
        local_direction(vertical_basis, -0.1, -0.1),
        local_direction(vertical_basis, 0.1, -0.1),
        local_direction(vertical_basis, 0.1, 0.1),
        local_direction(vertical_basis, -0.1, 0.1),
    };
    ProjectedTargetRegion vertical{};
    assert(project_visible_target_region(
        vertical_basis.forward,
        {vertical_component},
        vertical
    ));
    assert(point_in_visible_region(vertical, vertical_basis.forward));
    assert(!point_in_visible_region(
        vertical,
        local_direction(vertical_basis, 0.2, 0.0)
    ));

    ProjectedTargetRegion invalid = square;
    assert(!project_visible_target_region({}, {rectangle(-1.0, -1.0, 1.0, 1.0)}, invalid));
    assert(invalid.components.empty());
    assert(!project_visible_target_region(
        {0.0, 0.0, 1.0},
        {{z_direction(0.0, 0.0), z_direction(0.1, 0.0)}},
        invalid
    ));
    assert(!project_visible_target_region(
        {0.0, 0.0, 1.0},
        {{
            z_direction(0.0, 0.0),
            z_direction(0.1, 0.0),
            z_direction(0.2, 0.0),
        }},
        invalid
    ));
    assert(!project_visible_target_region(
        {0.0, 0.0, 1.0},
        {{
            z_direction(0.0, 0.0),
            z_direction(0.1, 0.0),
            {std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0},
        }},
        invalid
    ));

    const TargetMetrics target{
        0.0,
        0.0,
        2.0,
        2.0,
        4.0,
        2.0,
    };
    SigmaDriftConfig deterministic{};
    deterministic.undershoot_min = 0.5;
    deterministic.undershoot_max = 0.5;
    deterministic.overshoot_prob = 0.0;
    deterministic.curvature_scale = 0.0;
    deterministic.ou_sigma = 0.0;
    deterministic.tremor_amp_min = 0.0;
    deterministic.tremor_amp_max = 0.0;
    deterministic.sdn_k = 0.0;
    const VisibleDirectionComponents narrow_target{
        rectangle(-0.02, -0.02, 0.02, 0.02),
    };
    const GeometryFeedbackSigmaDriftConfig feedback{0.0, 0.5, 3};
    const AimPath corrected_path =
        generate_geometry_feedback_sigmadrift_path(
            {10.0, -2.0},
            target,
            narrow_target,
            0.15,
            deterministic,
            feedback,
            1234
        );
    assert(corrected_path.size() > 2);
    assert(corrected_path.front().yaw == 10.0);
    assert(corrected_path.front().pitch == -2.0);
    assert(corrected_path.front().t_ms == 0.0);
    for (std::size_t index = 1; index < corrected_path.size(); ++index) {
        assert(corrected_path[index].t_ms > corrected_path[index - 1].t_ms);
    }
    const AimSample corrected_end = corrected_path.back();
    ProjectedTargetRegion narrow_region{};
    assert(project_visible_target_region(
        {0.0, 0.0, 1.0},
        narrow_target,
        narrow_region
    ));
    assert(point_in_visible_region(
        narrow_region,
        minecraft_miner::look_direction_from_yaw_pitch(
            corrected_end.yaw,
            corrected_end.pitch
        )
    ));
    const AimPath uncorrected_path =
        generate_geometry_feedback_sigmadrift_path(
            {10.0, -2.0},
            target,
            narrow_target,
            0.15,
            deterministic,
            {0.0, 0.5, 0},
            1234
        );
    assert(!point_in_visible_region(
        narrow_region,
        minecraft_miner::look_direction_from_yaw_pitch(
            uncorrected_path.back().yaw,
            uncorrected_path.back().pitch
        )
    ));
    const AimPath repeated_path = generate_geometry_feedback_sigmadrift_path(
        {10.0, -2.0},
        target,
        narrow_target,
        0.15,
        deterministic,
        feedback,
        1234
    );
    assert(corrected_path.size() == repeated_path.size());
    for (std::size_t index = 0; index < corrected_path.size(); ++index) {
        assert(corrected_path[index].yaw == repeated_path[index].yaw);
        assert(corrected_path[index].pitch == repeated_path[index].pitch);
        assert(corrected_path[index].t_ms == repeated_path[index].t_ms);
    }
    assert(generate_geometry_feedback_sigmadrift_path(
        {10.0, -2.0},
        target,
        {},
        0.15,
        {},
        {},
        1234
    ).empty());
}
