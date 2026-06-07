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

struct ConstraintRegion {
    RegionId parent{};
    HalfPlaneId added_constraint{};
    ConstraintRegionState state = ConstraintRegionState::Empty;
    std::vector<HalfPlaneId> constraints{};
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
        HalfPlaneId constraint
    );

    const ConstraintRegion &region(RegionId id) const;
    bool is_empty(RegionId id) const;
    const std::vector<VertexId> &vertices(RegionId id) const;
    std::vector<Point2> approximate_vertices(RegionId id) const;

    std::size_t region_count() const;

private:
    RegionId intern_region(
        std::vector<HalfPlaneId> constraints,
        RegionId parent,
        HalfPlaneId added_constraint
    );
    std::vector<VertexId> compute_convex_hull(
        const std::vector<HalfPlaneId> &constraints
    );

    ExactGeometryStore &geometry_;
    std::vector<ConstraintRegion> regions_;
    std::map<std::vector<HalfPlaneId>, RegionId> region_ids_;
};

}  // namespace minescript_miner
