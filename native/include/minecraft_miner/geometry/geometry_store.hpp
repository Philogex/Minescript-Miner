#pragma once

#include "minecraft_miner/geometry/geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace minecraft_miner {

template <typename Tag>
struct GeometryId {
    std::uint32_t value = std::numeric_limits<std::uint32_t>::max();

    explicit constexpr operator bool() const {
        return value != std::numeric_limits<std::uint32_t>::max();
    }

    friend constexpr bool operator==(GeometryId lhs, GeometryId rhs) {
        return lhs.value == rhs.value;
    }

    friend constexpr bool operator!=(GeometryId lhs, GeometryId rhs) {
        return !(lhs == rhs);
    }

    friend constexpr bool operator<(GeometryId lhs, GeometryId rhs) {
        return lhs.value < rhs.value;
    }
};

using LineId = GeometryId<struct LineIdTag>;
using HalfPlaneId = GeometryId<struct HalfPlaneIdTag>;
using VertexId = GeometryId<struct VertexIdTag>;

struct ExactHalfPlane {
    LineId line{};
    bool positive_side = true;
};

class ExactGeometryStore {
public:
    LineId intern_line(ExactLine2 line);
    HalfPlaneId intern_half_plane(ExactLine2 oriented_line);
    VertexId intern_vertex(ExactPoint2H point);
    HalfPlaneId opposite(HalfPlaneId half_plane);

    VertexId intersect(LineId lhs, LineId rhs);
    ExactSign classify(VertexId vertex, HalfPlaneId half_plane);

    const ExactLine2 &line(LineId id) const;
    const ExactHalfPlane &half_plane(HalfPlaneId id) const;
    const ExactPoint2H &vertex(VertexId id) const;

    std::size_t line_count() const;
    std::size_t half_plane_count() const;
    std::size_t vertex_count() const;
    std::size_t intersection_cache_size() const;
    std::size_t classification_cache_size() const;

private:
    struct LineLess {
        bool operator()(const ExactLine2 &lhs, const ExactLine2 &rhs) const;
    };

    struct PointLess {
        bool operator()(const ExactPoint2H &lhs, const ExactPoint2H &rhs) const;
    };

    static ExactLine2 canonical_unoriented_normalized_line(
        ExactLine2 line
    );
    static bool same_orientation(
        const ExactLine2 &oriented,
        const ExactLine2 &canonical
    );
    LineId intern_canonical_line(ExactLine2 line);

    std::vector<ExactLine2> lines_;
    std::vector<ExactHalfPlane> half_planes_;
    std::vector<ExactPoint2H> vertices_;

    std::map<ExactLine2, LineId, LineLess> line_ids_;
    std::map<std::pair<LineId, bool>, HalfPlaneId> half_plane_ids_;
    std::map<ExactPoint2H, VertexId, PointLess> vertex_ids_;
    std::map<std::pair<LineId, LineId>, VertexId> intersections_;
    std::map<std::pair<VertexId, HalfPlaneId>, ExactSign> classifications_;
};

}  // namespace minecraft_miner
