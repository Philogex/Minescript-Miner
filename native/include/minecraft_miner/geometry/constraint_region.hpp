#pragma once

#include "minecraft_miner/geometry/geometry_store.hpp"
#include "minecraft_miner/geometry/tri2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace minecraft_miner {

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

struct VertexIdSpan {
    const VertexId *data = nullptr;
    std::size_t count = 0;

    const VertexId *begin() const {
        return data;
    }

    const VertexId *end() const {
        return count == 0 ? data : data + count;
    }

    bool empty() const {
        return count == 0;
    }

    std::size_t size() const {
        return count;
    }

    VertexId operator[](std::size_t index) const {
        return data[index];
    }

    VertexId front() const {
        return data[0];
    }

    VertexId back() const {
        return data[count - 1];
    }

    friend bool operator==(VertexIdSpan lhs, VertexIdSpan rhs) {
        return lhs.size() == rhs.size() &&
               std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    friend bool operator!=(VertexIdSpan lhs, VertexIdSpan rhs) {
        return !(lhs == rhs);
    }
};

struct StoredVertexSpan {
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
};

struct ConstraintRegion {
    RegionId parent{};
    RegionConstraint added_constraint{};
    ConstraintRegionState state = ConstraintRegionState::Empty;
    std::vector<RegionConstraint> constraints{};
    StoredVertexSpan vertices{};
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
    // Constraints are applied one by one because occluder subtraction needs
    // those intermediate prefix pieces; batch clipping would only fit future
    // static constraints that do not need intermediate regions.
    std::vector<RegionId> subtract_convex_region(
        RegionId source,
        const std::vector<HalfPlaneId> &occluder
    );

    const ConstraintRegion &region(RegionId id) const;
    bool is_empty(RegionId id) const;
    bool contains(RegionId id, VertexId point);
    bool contains_interior(RegionId id, VertexId point);
    VertexIdSpan vertices(RegionId id) const;
    std::vector<Point2> approximate_vertices(RegionId id) const;

    std::size_t region_count() const;

private:
    RegionId intern_region(
        std::vector<RegionConstraint> constraints,
        RegionId parent,
        RegionConstraint added_constraint,
        bool constraints_are_canonical = false,
        std::optional<StoredVertexSpan> known_hull = std::nullopt
    );
    std::vector<VertexId> compute_convex_hull(
        const std::vector<RegionConstraint> &constraints
    );
    std::optional<std::vector<VertexId>> compute_incremental_hull(
        RegionId parent,
        RegionConstraint added_constraint,
        const std::vector<RegionConstraint> &constraints
    );
    bool can_incrementally_clip(
        RegionId parent,
        RegionConstraint added_constraint,
        const std::vector<RegionConstraint> &constraints
    ) const;
    std::optional<VertexId> intersect_edge_with_constraint(
        VertexId from,
        VertexId to,
        HalfPlaneId constraint
    );
    std::vector<VertexId> compact_hull_vertices(
        std::vector<VertexId> vertices
    ) const;
    StoredVertexSpan store_vertices(
        const std::vector<VertexId> &vertices
    );
    VertexIdSpan vertex_span(StoredVertexSpan span) const;

    ExactGeometryStore &geometry_;
    std::vector<ConstraintRegion> regions_;
    std::vector<VertexId> vertex_storage_;
    std::map<std::vector<RegionConstraint>, RegionId> region_ids_;
};

}  // namespace minecraft_miner
