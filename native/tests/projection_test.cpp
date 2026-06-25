#include "minecraft_miner/geometry/constraint_region.hpp"
#include "minecraft_miner/scanner/projection.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

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
    std::int32_t max_x,
    std::int32_t min_y,
    std::int32_t max_y,
    std::int32_t z
) {
    using namespace minecraft_miner;
    return {
        PlaneAxis::Z,
        -1,
        from_sixteenths(z),
        from_sixteenths(min_x),
        from_sixteenths(max_x),
        from_sixteenths(min_y),
        from_sixteenths(max_y),
    };
}

minecraft_miner::HalfPlaneId footprint_on_line(
    const minecraft_miner::ExactGeometryStore &geometry,
    const minecraft_miner::ExactProjectedFace &face,
    minecraft_miner::LineId line
) {
    for (std::uint8_t i = 0; i < face.count; ++i) {
        if (geometry.half_plane(face.footprint[i]).line == line) {
            return face.footprint[i];
        }
    }
    return {};
}

}  // namespace

int main() {
    using namespace minecraft_miner;

    const ExactRational half{1, 2};
    const ExactRational quarter{1, 4};

    const ViewBasis identity_basis{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };

    ExactGeometryStore geometry;
    ExactProjector projector{
        geometry,
        {0.0, 0.0, 0.0},
        identity_basis
    };

    ExactProjectedFace near_face{};
    assert(projector.project_world_face(
        z_face(0, 32, 0, 32, 32),
        near_face
    ));
    assert(near_face.count == 4);
    assert(near_face.inverse_depth.x == 0);
    assert(near_face.inverse_depth.y == 0);
    assert(near_face.inverse_depth.constant == half);

    const std::array<ExactPoint2H, 4> expected_points{
        make_point(ExactRational{0}, ExactRational{0}),
        make_point(ExactRational{1}, ExactRational{0}),
        make_point(ExactRational{1}, ExactRational{1}),
        make_point(ExactRational{0}, ExactRational{1}),
    };
    for (std::uint8_t i = 0; i < near_face.count; ++i) {
        assert(geometry.vertex(near_face.vertices[i]) == expected_points[i]);
        assert(
            projector.inverse_depth_at(
                near_face,
                near_face.vertices[i]
            ) == half
        );
    }

    std::vector<HalfPlaneId> footprint{
        near_face.footprint.begin(),
        near_face.footprint.begin() + near_face.count,
    };
    ConstraintRegionStore regions{geometry};
    const RegionId projected_region =
        regions.intern_bounded_region(footprint);
    assert(!regions.is_empty(projected_region));
    assert(regions.vertices(projected_region).size() == 4);

    ExactProjectedFace far_face{};
    assert(projector.project_world_face(
        z_face(0, 64, 0, 64, 64),
        far_face
    ));
    const HalfPlaneId near_in_front =
        projector.depth_front_half_plane(near_face, far_face);
    const HalfPlaneId far_in_front =
        projector.depth_front_half_plane(far_face, near_face);
    assert(near_in_front);
    assert(far_in_front);
    assert(
        geometry.classify(near_face.vertices[0], near_in_front) ==
        ExactSign::Positive
    );
    assert(
        geometry.classify(near_face.vertices[0], far_in_front) ==
        ExactSign::Negative
    );
    assert(
        !projector.depth_front_half_plane(near_face, near_face)
    );
    assert(!regions.is_empty(
        regions.add_constraint(projected_region, near_in_front)
    ));
    assert(regions.is_empty(
        regions.add_constraint(projected_region, far_in_front)
    ));

    ExactProjectedFace left{};
    ExactProjectedFace right{};
    assert(projector.project_world_face(
        z_face(0, 16, 0, 16, 32),
        left
    ));
    assert(projector.project_world_face(
        z_face(16, 32, 0, 16, 32),
        right
    ));
    const LineId shared_edge = geometry.intern_line(
        line_through(
            make_point(half, ExactRational{0}),
            make_point(half, half)
        )
    );
    const HalfPlaneId left_shared =
        footprint_on_line(geometry, left, shared_edge);
    const HalfPlaneId right_shared =
        footprint_on_line(geometry, right, shared_edge);
    assert(left_shared);
    assert(right_shared);
    assert(left_shared != right_shared);
    assert(geometry.opposite(left_shared) == right_shared);

    ExactGeometryStore clipped_geometry;
    ExactProjector clipped_projector{
        clipped_geometry,
        {0.0, 0.0, 0.0},
        identity_basis,
        0.25
    };
    const WorldRectFace crossing_near{
        PlaneAxis::X,
        -1,
        from_sixteenths(16),
        from_sixteenths(0),
        from_sixteenths(16),
        from_sixteenths(-16),
        from_sixteenths(16),
    };
    ExactProjectedFace clipped{};
    assert(clipped_projector.project_world_face(
        crossing_near,
        clipped
    ));
    assert(clipped.count == 4);
    assert(
        std::count(
            clipped.depths.begin(),
            clipped.depths.begin() + clipped.count,
            quarter
        ) == 2
    );
    for (std::uint8_t i = 0; i < clipped.count; ++i) {
        assert(clipped.depths[i] >= quarter);
        assert(
            clipped_projector.inverse_depth_at(
                clipped,
                clipped.vertices[i]
            ) ==
            ExactRational{1} / clipped.depths[i]
        );
    }

    const ViewBasis tilted_basis{
        {0.8, 0.0, -0.6},
        {0.0, 1.0, 0.0},
        {0.6, 0.0, 0.8},
    };
    ExactGeometryStore tilted_geometry;
    ExactProjector tilted_projector{
        tilted_geometry,
        {0.0, 0.0, 0.0},
        tilted_basis
    };
    ExactProjectedFace tilted{};
    assert(tilted_projector.project_world_face(
        z_face(0, 16, 0, 16, 32),
        tilted
    ));
    assert(tilted.inverse_depth.x != 0);
    for (std::uint8_t i = 0; i < tilted.count; ++i) {
        assert(
            tilted_projector.inverse_depth_at(
                tilted,
                tilted.vertices[i]
            ) ==
            ExactRational{1} / tilted.depths[i]
        );
    }
}
