#pragma once

#include "minescript_miner/exact_geometry_store.hpp"
#include "minescript_miner/tri2.hpp"

#include <cstdint>
#include <map>
#include <vector>

namespace minescript_miner {

using RegionId = GeometryId<struct RegionIdTag>;

enum class ConstraintRegionState : std::uint8_t {
    Empty,
    Bounded,
};

struct RegionConstraint {
    HalfPlaneId half_plane{};
    bool strict = false;

    friend constexpr bool operator==(
        RegionConstraint lhs,
        RegionConstraint rhs
    ) {
        return lhs.half_plane == rhs.half_plane &&
               lhs.strict == rhs.strict;
    }

    friend constexpr bool operator<(
        RegionConstraint lhs,
        RegionConstraint rhs
    ) {
        return lhs.half_plane != rhs.half_plane
            ? lhs.half_plane < rhs.half_plane
            : lhs.strict < rhs.strict;
    }
};

struct ConstraintRegion {
    RegionId parent{};
    RegionConstraint added_constraint{};
    ConstraintRegionState state = ConstraintRegionState::Empty;
    std::vector<RegionConstraint> constraints{};
    std::vector<VertexId> vertices{};
};

class ConstraintRegionStore {
public:
    explicit ConstraintRegionStore(ExactGeometryStore &geometry);

    // The constraints must describe either a bounded two-dimensional region
    // or an empty intersection. Unbounded feasibility is outside this store's
    // contract because B&B always starts from a bounded target face.
    RegionId intern_bounded_region(
        std::vector<HalfPlaneId> constraints
    );
    RegionId add_constraint(
        RegionId parent,
        HalfPlaneId constraint,
        bool strict = false
    );
    // Returns closures of the disjoint visible prefix pieces. Complemented
    // occluder boundaries are stored as strict constraints, and an empty
    // occluder constraint list is treated as a no-op.
    std::vector<RegionId> subtract_convex_region(
        RegionId source,
        const std::vector<HalfPlaneId> &occluder
    );

    const ConstraintRegion &region(RegionId id) const;
    bool is_empty(RegionId id) const;
    bool contains(RegionId id, VertexId point);
    bool contains_interior(RegionId id, VertexId point);
    const std::vector<VertexId> &vertices(RegionId id) const;
    std::vector<Point2> approximate_vertices(RegionId id) const;

    std::size_t region_count() const;

private:
    RegionId intern_region(
        std::vector<RegionConstraint> constraints,
        RegionId parent,
        RegionConstraint added_constraint
    );
    std::vector<VertexId> compute_convex_hull(
        const std::vector<RegionConstraint> &constraints
    );

    ExactGeometryStore &geometry_;
    std::vector<ConstraintRegion> regions_;
    std::map<std::vector<RegionConstraint>, RegionId> region_ids_;
};

}  // namespace minescript_miner
